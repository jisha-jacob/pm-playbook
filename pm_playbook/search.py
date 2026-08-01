"""
Text retrieval for PM Playbook.

This module loads the processed transcript chunks from Parquet,
builds a Minsearch TF-IDF index, and exposes a reusable search API.

The baseline retrieval approach intentionally excludes chunks that
were flagged as sponsor or promotional content during ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from minsearch import Index


DEFAULT_CHUNKS_PATH = Path("data/chunks.parquet")

TEXT_FIELDS = [
    "text",
    "episode_title",
    "guest",
    "speaker_name",
]

KEYWORD_FIELDS = [
    "chunk_id",
    "episode_id",
    "guest",
    "speaker_name",
    "video_id",
    "is_sponsor_read",
]

DEFAULT_BOOSTS = {
    "text": 1.0,
    "episode_title": 2.0,
    "guest": 3.0,
    "speaker_name": 2.0,
}


@dataclass
class SearchResult:
    """
    One normalized retrieval result.

    Minsearch returns dictionaries. This model gives the rest of the
    application a stable, explicit result structure.
    """

    chunk_id: str
    episode_id: str
    guest: str
    episode_title: str
    speaker_name: str
    text: str
    publish_date: str | None
    youtube_url: str | None
    video_id: str | None
    start_time: int | None
    end_time: int | None
    chunk_position: int
    word_count: int
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result to a serializable dictionary.
        """
        return {
            "chunk_id": self.chunk_id,
            "episode_id": self.episode_id,
            "guest": self.guest,
            "episode_title": self.episode_title,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "publish_date": self.publish_date,
            "youtube_url": self.youtube_url,
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "chunk_position": self.chunk_position,
            "word_count": self.word_count,
            "score": self.score,
        }


class PMPlaybookSearch:
    """
    Baseline text-retrieval service backed by Minsearch.
    """

    def __init__(
        self,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
        *,
        exclude_sponsors: bool = True,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.exclude_sponsors = exclude_sponsors

        self.documents: list[dict[str, Any]] = []
        self.index: Index | None = None

    def load_documents(self) -> list[dict[str, Any]]:
        """
        Load retrieval documents from the Parquet artifact.

        Sponsor/promotional chunks are excluded by default, but they remain
        available in the source Parquet file for inspection and evaluation.
        """
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file does not exist: {self.chunks_path}\n"
                "Generate it with:\n"
                "  uv run python -m pm_playbook.ingest --fail-fast"
            )

        dataframe = pd.read_parquet(self.chunks_path)

        required_columns = {
            "chunk_id",
            "episode_id",
            "guest",
            "episode_title",
            "speaker_name",
            "text",
            "publish_date",
            "youtube_url",
            "video_id",
            "start_time",
            "end_time",
            "chunk_position",
            "word_count",
            "is_sponsor_read",
        }

        missing_columns = sorted(required_columns - set(dataframe.columns))

        if missing_columns:
            raise ValueError(
                f"Chunk dataset is missing required columns: {missing_columns}"
            )

        if self.exclude_sponsors:
            dataframe = dataframe[~dataframe["is_sponsor_read"].fillna(False)].copy()

        dataframe = dataframe[dataframe["text"].fillna("").str.strip().ne("")].copy()

        dataframe = dataframe[dataframe["word_count"].fillna(0).ge(20)].copy()

        dataframe = dataframe.where(
            pd.notna(dataframe),
            None,
        )

        self.documents = dataframe.to_dict(orient="records")

        if not self.documents:
            raise ValueError(
                "No searchable documents remained after loading and filtering."
            )

        return self.documents

    def build_index(self) -> Index:
        """
        Build and fit the Minsearch text index.
        """
        if not self.documents:
            self.load_documents()

        self.index = Index(
            text_fields=TEXT_FIELDS,
            keyword_fields=KEYWORD_FIELDS,
        )

        self.index.fit(self.documents)

        return self.index

    def ensure_ready(self) -> None:
        """
        Lazily initialize documents and the index.
        """
        if self.index is None:
            self.build_index()

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        guest: str | None = None,
        speaker_name: str | None = None,
        boost_dict: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """
        Search transcript chunks using TF-IDF similarity.

        Parameters
        ----------
        query:
            Natural-language user query.
        num_results:
            Maximum number of results to return.
        guest:
            Optional exact guest filter.
        speaker_name:
            Optional exact speaker filter.
        boost_dict:
            Optional Minsearch field boosts. Defaults to DEFAULT_BOOSTS.

        Returns
        -------
        list[SearchResult]
            Ranked retrieval results.
        """
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        if num_results < 1:
            raise ValueError("num_results must be at least 1.")

        self.ensure_ready()

        if self.index is None:
            raise RuntimeError("Search index was not initialized.")

        filter_dict: dict[str, Any] = {}

        if guest:
            filter_dict["guest"] = guest

        if speaker_name:
            filter_dict["speaker_name"] = speaker_name

        raw_results = self.index.search(
            cleaned_query,
            filter_dict=filter_dict or None,
            boost_dict=boost_dict or DEFAULT_BOOSTS,
            num_results=num_results,
        )

        return [self._normalize_result(result) for result in raw_results]

    @staticmethod
    def _normalize_result(
        result: dict[str, Any],
    ) -> SearchResult:
        """
        Convert one Minsearch result dictionary into SearchResult.
        """
        return SearchResult(
            chunk_id=str(result["chunk_id"]),
            episode_id=str(result["episode_id"]),
            guest=str(result["guest"]),
            episode_title=str(result["episode_title"]),
            speaker_name=str(result["speaker_name"]),
            text=str(result["text"]),
            publish_date=_optional_string(result.get("publish_date")),
            youtube_url=_optional_string(result.get("youtube_url")),
            video_id=_optional_string(result.get("video_id")),
            start_time=_optional_int(result.get("start_time")),
            end_time=_optional_int(result.get("end_time")),
            chunk_position=int(result["chunk_position"]),
            word_count=int(result["word_count"]),
            score=_extract_score(result),
        )


def _optional_string(value: Any) -> str | None:
    """
    Normalize nullable values to strings or None.

    Pandas may represent missing values as NaN.
    """
    if value is None or pd.isna(value):
        return None

    return str(value)


def _optional_int(value: Any) -> int | None:
    """
    Normalize nullable numeric values to int or None.

    Pandas may represent missing numeric values as NaN.
    """
    if value is None or pd.isna(value):
        return None

    return int(value)


def _extract_score(
    result: dict[str, Any],
) -> float | None:
    """
    Extract a relevance score when exposed by the installed Minsearch version.

    Minsearch result dictionaries may not always include a public score field,
    so score remains optional.
    """
    for key in ("score", "_score", "relevance_score"):
        value = result.get(key)

        if value is not None:
            return float(value)

    return None


def format_timestamp(seconds: int | None) -> str | None:
    """
    Convert seconds to HH:MM:SS for display.
    """
    if seconds is None:
        return None

    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def search_documents(
    query: str,
    *,
    num_results: int = 5,
    guest: str | None = None,
    speaker_name: str | None = None,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
) -> list[dict[str, Any]]:
    """
    Convenience function for scripts and notebooks.

    For repeated application queries, instantiate PMPlaybookSearch once
    instead of rebuilding the index for every call.
    """
    search_engine = PMPlaybookSearch(
        chunks_path=chunks_path,
    )

    results = search_engine.search(
        query=query,
        num_results=num_results,
        guest=guest,
        speaker_name=speaker_name,
    )

    return [result.to_dict() for result in results]


def main() -> None:
    """
    Run a small command-line smoke test.
    """
    query = "How do I know if I have product-market fit?"

    search_engine = PMPlaybookSearch()
    results = search_engine.search(
        query=query,
        num_results=5,
    )

    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print()

    for position, result in enumerate(
        results,
        start=1,
    ):
        timestamp = format_timestamp(result.start_time)

        print("=" * 80)
        print(f"RESULT {position}")
        print(f"Guest: {result.guest}")
        print(f"Episode: {result.episode_title}")
        print(f"Speaker: {result.speaker_name}")
        print(f"Timestamp: {timestamp or 'Unknown'}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Words: {result.word_count}")
        print()
        print(result.text[:800])
        print()


if __name__ == "__main__":
    main()
