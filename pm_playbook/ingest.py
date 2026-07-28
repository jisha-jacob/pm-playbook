"""
Corpus-level ingestion pipeline for PM Playbook.

This module:

1. Discovers transcript Markdown files.
2. Parses each transcript into an Episode.
3. Converts each Episode into retrieval-ready Chunk objects.
4. Validates the generated chunks.
5. Writes the complete corpus to a Parquet file.

Sponsor chunks are preserved in the output and identified using the
``is_sponsor_read`` field. They can be excluded later when building
the retrieval index.

Run from the repository root:

    uv run python -m pm_playbook.ingest

Examples:

    uv run python -m pm_playbook.ingest --limit 3

    uv run python -m pm_playbook.ingest \
        --limit 20 \
        --output data/chunks-quality-test.parquet

    uv run python -m pm_playbook.ingest --fail-fast
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd
from pydantic import ValidationError

from pm_playbook.chunker import create_chunks
from pm_playbook.constants import MAX_WORDS_PER_CHUNK, OUTPUT_PARQUET
from pm_playbook.models import Chunk
from pm_playbook.parser import parse_transcript
from pm_playbook.utils import generate_episode_id


DEFAULT_TRANSCRIPTS_DIR = Path("data/raw/transcripts/episodes")


@dataclass
class IngestionResult:
    transcript_files_found: int
    transcript_files_selected: int
    episodes_processed: int
    episodes_failed: int
    episodes_skipped: int
    chunks_created: int
    sponsor_chunks: int
    output_path: Path
    elapsed_seconds: float


def discover_transcript_files(
    transcripts_dir: str | Path,
) -> list[Path]:
    """
    Find transcript.md files in the configured episodes directory.

    Parameters
    ----------
    transcripts_dir:
        Directory containing one subdirectory per podcast episode.

    Returns
    -------
    list[Path]
        Transcript paths sorted deterministically.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no transcript files.
    NotADirectoryError
        If the supplied path is not a directory.
    """
    directory = Path(transcripts_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Transcript directory does not exist: {directory}\n"
            "Make sure the dataset submodule has been initialized:\n"
            "  git submodule update --init --recursive"
        )

    if not directory.is_dir():
        raise NotADirectoryError(f"Transcript path is not a directory: {directory}")

    transcript_files = sorted(directory.glob("*/transcript.md"))

    if not transcript_files:
        raise FileNotFoundError(
            f"No transcript.md files were found under {directory}.\n"
            "Try running:\n"
            "  git submodule update --init --recursive"
        )

    return transcript_files


def validate_chunks(
    chunks: Sequence[Chunk],
    max_words: int = MAX_WORDS_PER_CHUNK,
) -> list[str]:
    """
    Validate chunk-level invariants.

    All detected validation problems are returned together so that
    multiple issues can be inspected in one run.

    Parameters
    ----------
    chunks:
        Generated chunks.
    max_words:
        Maximum permitted number of words in one chunk.

    Returns
    -------
    list[str]
        Human-readable validation errors. An empty list means that
        validation succeeded.
    """
    errors: list[str] = []

    if not chunks:
        return ["The ingestion pipeline generated no chunks."]

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    duplicate_ids = [
        chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1
    ]

    if duplicate_ids:
        errors.append(f"Found {len(duplicate_ids)} duplicate chunk IDs.")

    empty_chunks = [chunk.chunk_id for chunk in chunks if not chunk.text.strip()]

    if empty_chunks:
        errors.append(f"Found {len(empty_chunks)} chunks with empty text.")

    incorrect_word_counts = [
        chunk.chunk_id
        for chunk in chunks
        if chunk.word_count != len(chunk.text.split())
    ]

    if incorrect_word_counts:
        errors.append(
            f"Found {len(incorrect_word_counts)} chunks with "
            "incorrect word_count values."
        )

    oversized_chunks = [chunk for chunk in chunks if chunk.word_count > max_words]

    if oversized_chunks:
        largest_chunk = max(
            oversized_chunks,
            key=lambda chunk: chunk.word_count,
        )

        errors.append(
            f"Found {len(oversized_chunks)} chunks larger than "
            f"{max_words} words. The largest chunk has "
            f"{largest_chunk.word_count} words and ID "
            f"{largest_chunk.chunk_id}."
        )

    invalid_global_indexes = [
        chunk.chunk_index
        for expected_index, chunk in enumerate(chunks)
        if chunk.chunk_index != expected_index
    ]

    if invalid_global_indexes:
        errors.append("Global chunk_index values are not continuous from zero.")

    invalid_episode_ids = find_invalid_chunk_positions(chunks)

    if invalid_episode_ids:
        errors.append(
            "Chunk positions are not continuous from zero in "
            f"{len(invalid_episode_ids)} episodes."
        )

    return errors


def find_invalid_chunk_positions(
    chunks: Sequence[Chunk],
) -> list[str]:
    """
    Find episodes whose chunk positions are not continuous from zero.

    Parameters
    ----------
    chunks:
        Generated chunks.

    Returns
    -------
    list[str]
        Episode IDs with invalid chunk-position sequences.
    """
    positions_by_episode: dict[str, list[int]] = {}

    for chunk in chunks:
        positions_by_episode.setdefault(
            chunk.episode_id,
            [],
        ).append(chunk.chunk_position)

    invalid_episode_ids: list[str] = []

    for episode_id, positions in positions_by_episode.items():
        expected_positions = list(range(len(positions)))

        if sorted(positions) != expected_positions:
            invalid_episode_ids.append(episode_id)

    return invalid_episode_ids


def chunks_to_dataframe(
    chunks: Sequence[Chunk],
) -> pd.DataFrame:
    """
    Convert Chunk models into a Pandas DataFrame.

    Date values are serialized as ISO-formatted strings to keep the
    resulting Parquet schema predictable across environments.

    Parameters
    ----------
    chunks:
        Validated chunks.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per chunk.
    """
    records: list[dict[str, object]] = []

    for chunk in chunks:
        record = chunk.model_dump()

        if chunk.publish_date is not None:
            record["publish_date"] = chunk.publish_date.isoformat()

        records.append(record)

    dataframe = pd.DataFrame.from_records(records)

    if dataframe.empty:
        raise ValueError("Cannot create a Parquet file from an empty chunk collection.")

    column_order = [
        "chunk_id",
        "episode_id",
        "chunk_index",
        "chunk_position",
        "guest",
        "episode_title",
        "publish_date",
        "youtube_url",
        "video_id",
        "speaker_name",
        "start_time",
        "end_time",
        "text",
        "word_count",
        "is_sponsor_read",
    ]

    missing_columns = [
        column for column in column_order if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Generated chunk data is missing expected columns: {missing_columns}"
        )

    return dataframe[column_order]


def write_chunks_parquet(
    chunks: Sequence[Chunk],
    output_path: str | Path,
) -> Path:
    """
    Write chunks to a compressed Parquet file.

    The output directory is created automatically when necessary.

    Parameters
    ----------
    chunks:
        Validated chunks.
    output_path:
        Destination Parquet path.

    Returns
    -------
    Path
        Path to the written Parquet file.
    """
    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = chunks_to_dataframe(chunks)

    dataframe.to_parquet(
        destination,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    return destination


def ingest_corpus(
    transcripts_dir: str | Path = DEFAULT_TRANSCRIPTS_DIR,
    output_path: str | Path = OUTPUT_PARQUET,
    *,
    limit: int | None = None,
    fail_fast: bool = False,
) -> IngestionResult:
    """
    Run the complete corpus ingestion pipeline.

    Sponsor chunks are retained in the generated Parquet file and
    identified through the ``is_sponsor_read`` column.

    Parameters
    ----------
    transcripts_dir:
        Root directory containing episode folders.
    output_path:
        Destination Parquet file.
    limit:
        Optional number of transcripts to process. This is useful for
        smoke tests and manual quality checks.
    fail_fast:
        Stop immediately when an episode fails. When False, failed
        episodes are reported and processing continues.

    Returns
    -------
    IngestionResult
        Summary statistics for the completed ingestion run.
    """
    started_at = perf_counter()

    transcript_files = discover_transcript_files(
        transcripts_dir=transcripts_dir,
    )

    total_files_found = len(transcript_files)

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        selected_files = transcript_files[:limit]
    else:
        selected_files = transcript_files

    all_chunks: list[Chunk] = []
    seen_episode_ids: set[str] = set()

    episodes_processed = 0
    episodes_failed = 0
    episodes_skipped = 0
    sponsor_chunks_detected = 0

    print(f"Found {total_files_found} transcript files.")

    if limit is not None:
        print(
            "Smoke-test limit enabled: processing "
            f"{len(selected_files)} transcript files."
        )

    for file_number, transcript_path in enumerate(
        selected_files,
        start=1,
    ):
        try:
            episode = parse_transcript(transcript_path)

            episode_id = generate_episode_id(
                guest=episode.guest,
                title=episode.title,
                video_id=episode.video_id,
                transcript_path=episode.transcript_path,
            )

            if episode_id in seen_episode_ids:
                episodes_skipped += 1

                print(
                    f"[{file_number}/{len(selected_files)}] "
                    f"SKIPPED DUPLICATE: {episode.guest} "
                    f"({transcript_path})"
                )
                continue

            seen_episode_ids.add(episode_id)
            episode_chunks = create_chunks(
                episode=episode,
                starting_chunk_index=len(all_chunks),
            )

            sponsor_count = sum(chunk.is_sponsor_read for chunk in episode_chunks)

            sponsor_chunks_detected += sponsor_count
            all_chunks.extend(episode_chunks)
            episodes_processed += 1

            print(
                f"[{file_number}/{len(selected_files)}] "
                f"{episode.guest}: "
                f"{len(episode_chunks)} chunks "
                f"({sponsor_count} sponsor-flagged)"
            )

        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            episodes_failed += 1

            print(
                f"[{file_number}/{len(selected_files)}] FAILED: {transcript_path}",
                file=sys.stderr,
            )
            print(
                f"  {type(error).__name__}: {error}",
                file=sys.stderr,
            )

            if fail_fast:
                raise

    validation_errors = validate_chunks(all_chunks)

    if validation_errors:
        formatted_errors = "\n".join(f"- {error}" for error in validation_errors)

        raise ValueError(f"Chunk validation failed:\n{formatted_errors}")

    saved_path = write_chunks_parquet(
        chunks=all_chunks,
        output_path=output_path,
    )

    elapsed_seconds = perf_counter() - started_at

    result = IngestionResult(
        transcript_files_found=total_files_found,
        transcript_files_selected=len(selected_files),
        episodes_processed=episodes_processed,
        episodes_failed=episodes_failed,
        episodes_skipped=episodes_skipped,
        chunks_created=len(all_chunks),
        sponsor_chunks=sponsor_chunks_detected,
        output_path=saved_path,
        elapsed_seconds=elapsed_seconds,
    )

    print_ingestion_summary(result)

    return result


def print_ingestion_summary(
    result: IngestionResult,
) -> None:
    """
    Print a readable summary of an ingestion run.
    """
    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Transcript files found:    {result.transcript_files_found}")
    print(f"Transcript files selected: {result.transcript_files_selected}")
    print(f"Episodes processed:        {result.episodes_processed}")
    print(f"Episodes failed:           {result.episodes_failed}")
    print(f"Episodes skipped:          {result.episodes_skipped}")
    print(f"Chunks written:            {result.chunks_created}")
    print(f"Sponsor chunks flagged:    {result.sponsor_chunks}")
    print(f"Output:                    {result.output_path}")
    print(f"Elapsed time:              {result.elapsed_seconds:.2f} seconds")
    print("=" * 60)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Parse Lenny's Podcast transcripts and create "
            "a retrieval-ready Parquet dataset."
        )
    )

    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help=(
            f"Directory containing episode folders. Default: {DEFAULT_TRANSCRIPTS_DIR}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_PARQUET),
        help=(f"Destination Parquet file. Default: {OUTPUT_PARQUET}"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("Process only the first N transcripts. Useful for smoke testing."),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=("Stop on the first failed transcript instead of continuing."),
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()

    try:
        ingest_corpus(
            transcripts_dir=arguments.transcripts_dir,
            output_path=arguments.output,
            limit=arguments.limit,
            fail_fast=arguments.fail_fast,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Ingestion failed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
