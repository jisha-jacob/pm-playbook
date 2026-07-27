"""
Speaker-turn parsing and chunk generation for PM Playbook.

This module transforms a parsed Episode into retrieval-ready Chunk objects.

Pipeline:
    Episode
        -> parse speaker turns
        -> merge short consecutive turns
        -> split long turns
        -> create Chunk objects
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pm_playbook.constants import (
    CONTINUATION_PATTERN_HHMM,
    CONTINUATION_PATTERN_HHMMSS,
    MAX_WORDS_PER_CHUNK,
    MERGE_THRESHOLD,
    SPEAKER_PATTERN_HHMM,
    SPEAKER_PATTERN_HHMMSS,
    SPEAKER_PATTERN_NAME_ONLY,
)
from pm_playbook.models import Chunk, Episode, SpeakerTurn
from pm_playbook.utils import (
    clean_text,
    count_words,
    generate_chunk_id,
    generate_episode_id,
    is_sponsor_read,
    parse_timestamp,
)


# Compile the expressions once when the module is imported.
# This is more efficient and makes the parsing functions easier to read.
SPEAKER_HHMMSS_RE = re.compile(SPEAKER_PATTERN_HHMMSS)
SPEAKER_HHMM_RE = re.compile(SPEAKER_PATTERN_HHMM)
SPEAKER_NAME_ONLY_RE = re.compile(SPEAKER_PATTERN_NAME_ONLY)

CONTINUATION_HHMMSS_RE = re.compile(CONTINUATION_PATTERN_HHMMSS)
CONTINUATION_HHMM_RE = re.compile(CONTINUATION_PATTERN_HHMM)

# A lightweight sentence-boundary pattern.
# It splits after sentence-ending punctuation followed by whitespace.
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def parse_speaker_turns(
    transcript: str,
    default_speaker: str = "Unknown",
) -> list[SpeakerTurn]:
    """
    Parse transcript text into speaker turns.

    Supported speaker formats:
        Name (HH:MM:SS): text
        Name (MM:SS): text
        Name: text

    Supported continuation formats:
        (HH:MM:SS): text
        (MM:SS): text

    Continuation timestamps inherit the most recently identified speaker.

    Ordinary lines that do not contain a speaker marker are appended to the
    current turn. Text before the first detected speaker is preserved under
    ``default_speaker``.

    Parameters
    ----------
    transcript:
        Raw transcript body without YAML frontmatter.
    default_speaker:
        Speaker name used for text found before the first speaker label.

    Returns
    -------
    list[SpeakerTurn]
        Parsed speaker turns in transcript order.
    """
    turns: list[SpeakerTurn] = []

    current_speaker = default_speaker
    current_start_time: int | None = None
    current_text_parts: list[str] = []

    def flush_current_turn() -> None:
        """
        Save the current buffered turn if it contains non-empty text.
        """
        nonlocal current_text_parts

        text = clean_text(" ".join(current_text_parts))

        if text:
            turns.append(
                SpeakerTurn(
                    speaker_name=current_speaker,
                    start_time=current_start_time,
                    text=text,
                )
            )

        current_text_parts = []

    for raw_line in transcript.splitlines():
        line = raw_line.strip()

        # Blank lines separate visual paragraphs but should not create
        # empty speaker turns.
        if not line:
            continue

        match = SPEAKER_HHMMSS_RE.match(line)

        if match:
            flush_current_turn()

            current_speaker = clean_text(match.group(1))
            current_start_time = parse_timestamp(match.group(2))

            initial_text = clean_text(match.group(3))
            current_text_parts = [initial_text] if initial_text else []
            continue

        match = SPEAKER_HHMM_RE.match(line)

        if match:
            flush_current_turn()

            current_speaker = clean_text(match.group(1))
            current_start_time = parse_timestamp(match.group(2))

            initial_text = clean_text(match.group(3))
            current_text_parts = [initial_text] if initial_text else []
            continue

        match = SPEAKER_NAME_ONLY_RE.match(line)

        if match:
            flush_current_turn()

            current_speaker = clean_text(match.group(1))
            current_start_time = None

            initial_text = clean_text(match.group(2))
            current_text_parts = [initial_text] if initial_text else []
            continue

        match = CONTINUATION_HHMMSS_RE.match(line)

        if match:
            # A continuation timestamp starts another segment for the same
            # speaker. Flush the previous text while preserving the speaker.
            flush_current_turn()

            current_start_time = parse_timestamp(match.group(1))

            continuation_text = clean_text(match.group(2))
            current_text_parts = (
                [continuation_text] if continuation_text else []
            )
            continue

        match = CONTINUATION_HHMM_RE.match(line)

        if match:
            flush_current_turn()

            current_start_time = parse_timestamp(match.group(1))

            continuation_text = clean_text(match.group(2))
            current_text_parts = (
                [continuation_text] if continuation_text else []
            )
            continue

        # This is a normal continuation line belonging to the current turn.
        current_text_parts.append(line)

    # Save the final buffered turn.
    flush_current_turn()

    return turns


def merge_short_turns(
    turns: Sequence[SpeakerTurn],
    threshold: int = MERGE_THRESHOLD,
) -> list[SpeakerTurn]:
    """
    Merge short consecutive turns from the same speaker.

    Two consecutive turns are merged when they belong to the same speaker
    and at least one of the two turns is shorter than ``threshold`` words.

    The first turn's start timestamp is retained.

    Parameters
    ----------
    turns:
        Speaker turns in transcript order.
    threshold:
        Word-count threshold used to identify a short turn.

    Returns
    -------
    list[SpeakerTurn]
        Speaker turns after merging.
    """
    if not turns:
        return []

    merged: list[SpeakerTurn] = []

    for turn in turns:
        if not merged:
            merged.append(turn.model_copy())
            continue

        previous = merged[-1]

        same_speaker = (
            previous.speaker_name.strip().casefold()
            == turn.speaker_name.strip().casefold()
        )

        previous_is_short = count_words(previous.text) < threshold
        current_is_short = count_words(turn.text) < threshold

        if same_speaker and (previous_is_short or current_is_short):
            combined_text = clean_text(
                f"{previous.text} {turn.text}"
            )

            merged[-1] = SpeakerTurn(
                speaker_name=previous.speaker_name,
                start_time=previous.start_time,
                text=combined_text,
            )
        else:
            merged.append(turn.model_copy())

    return merged


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation boundaries.

    This intentionally avoids adding an NLP dependency during Phase 1.
    """
    cleaned = clean_text(text)

    if not cleaned:
        return []

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_RE.split(cleaned)
        if sentence.strip()
    ]

    return sentences or [cleaned]


def split_oversized_sentence(
    sentence: str,
    max_words: int,
) -> list[str]:
    """
    Split an unusually long sentence into word-based pieces.

    This is a safety fallback for transcript text that contains no usable
    sentence punctuation. Normal transcript sentences remain intact.
    """
    words = sentence.split()

    return [
        " ".join(words[index : index + max_words])
        for index in range(0, len(words), max_words)
    ]


def split_long_text(
    text: str,
    max_words: int = MAX_WORDS_PER_CHUNK,
) -> list[str]:
    """
    Split long text into chunks without cutting normal sentences.

    Sentences are accumulated until adding another sentence would exceed
    ``max_words``. An individual sentence longer than ``max_words`` is split
    by words as a last-resort fallback.

    Parameters
    ----------
    text:
        Speaker-turn text.
    max_words:
        Maximum target word count for each resulting piece.

    Returns
    -------
    list[str]
        One or more clean text pieces.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1.")

    cleaned = clean_text(text)

    if not cleaned:
        return []

    if count_words(cleaned) <= max_words:
        return [cleaned]

    sentences = split_into_sentences(cleaned)

    pieces: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    def flush_current_piece() -> None:
        nonlocal current_sentences, current_word_count

        if current_sentences:
            pieces.append(clean_text(" ".join(current_sentences)))

        current_sentences = []
        current_word_count = 0

    for sentence in sentences:
        sentence_word_count = count_words(sentence)

        # Handle transcript sections with missing punctuation that appear
        # as one extremely long sentence.
        if sentence_word_count > max_words:
            flush_current_piece()
            pieces.extend(
                split_oversized_sentence(
                    sentence=sentence,
                    max_words=max_words,
                )
            )
            continue

        would_exceed_limit = (
            current_word_count + sentence_word_count > max_words
        )

        if current_sentences and would_exceed_limit:
            flush_current_piece()

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    flush_current_piece()

    return pieces


def calculate_turn_end_time(
    turns: Sequence[SpeakerTurn],
    turn_index: int,
) -> int | None:
    """
    Find the next available timestamp after the current speaker turn.

    The next timestamp is used as an approximate end time. If no later
    timestamp is available, ``None`` is returned.
    """
    for later_turn in turns[turn_index + 1 :]:
        if later_turn.start_time is not None:
            return later_turn.start_time

    return None


def create_chunks(
    episode: Episode,
    starting_chunk_index: int = 0,
    merge_threshold: int = MERGE_THRESHOLD,
    max_words: int = MAX_WORDS_PER_CHUNK,
) -> list[Chunk]:
    """
    Convert an Episode into retrieval-ready Chunk objects.

    Parameters
    ----------
    episode:
        Parsed episode metadata and transcript body.
    starting_chunk_index:
        Starting value for the global chunk index. The ingestion orchestrator
        can use this when processing multiple episodes.
    merge_threshold:
        Maximum word count used when identifying short consecutive turns.
    max_words:
        Maximum target size of each output chunk.

    Returns
    -------
    list[Chunk]
        Validated chunks in episode order.
    """
    episode_id = generate_episode_id(
        guest=episode.guest,
        title=episode.title,
    )

    parsed_turns = parse_speaker_turns(
        transcript=episode.transcript,
        default_speaker=episode.guest,
    )

    merged_turns = merge_short_turns(
        turns=parsed_turns,
        threshold=merge_threshold,
    )

    chunks: list[Chunk] = []
    chunk_position = 0

    for turn_index, turn in enumerate(merged_turns):
        text_pieces = split_long_text(
            text=turn.text,
            max_words=max_words,
        )

        turn_end_time = calculate_turn_end_time(
            turns=merged_turns,
            turn_index=turn_index,
        )

        for text_piece in text_pieces:
            cleaned_piece = clean_text(text_piece)

            if not cleaned_piece:
                continue

            chunk_id = generate_chunk_id(
                episode_id=episode_id,
                chunk_position=chunk_position,
            )

            chunk = Chunk(
                chunk_id=chunk_id,
                episode_id=episode_id,
                chunk_index=starting_chunk_index + len(chunks),
                chunk_position=chunk_position,
                guest=episode.guest,
                episode_title=episode.title,
                publish_date=episode.publish_date,
                youtube_url=episode.youtube_url,
                video_id=episode.video_id,
                speaker_name=turn.speaker_name,
                start_time=turn.start_time,
                end_time=turn_end_time,
                text=cleaned_piece,
                word_count=count_words(cleaned_piece),
                is_sponsor_read=is_sponsor_read(cleaned_piece),
            )

            chunks.append(chunk)
            chunk_position += 1

    return chunks