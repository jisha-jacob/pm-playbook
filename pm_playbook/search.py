"""
Text and hybrid retrieval for PM Playbook.

This module loads processed transcript chunks from Parquet and exposes:

1. PMPlaybookSearch
   Minsearch TF-IDF text retrieval.

2. PMPlaybookHybridSearch
   Minsearch text retrieval combined with SentenceTransformer vector
   retrieval using Reciprocal Rank Fusion.

Sponsor/promotional chunks and fragments shorter than 20 words are excluded
from the searchable corpus by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from minsearch import Index
from sentence_transformers import SentenceTransformer


DEFAULT_CHUNKS_PATH = Path("data/chunks.parquet")

DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings/all-MiniLM-L6-v2-text-embeddings.npy")
DEFAULT_EMBEDDING_CHUNK_IDS_PATH = Path(
    "data/embeddings/all-MiniLM-L6-v2-text-chunk-ids.json"
)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TEXT_CANDIDATES = 20
DEFAULT_VECTOR_CANDIDATES = 20
DEFAULT_RRF_K = 60

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

    This model gives the rest of the application a stable result structure
    regardless of whether text or hybrid retrieval produced the result.
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
        Load searchable documents from the Parquet artifact.

        Sponsor/promotional chunks are excluded by default, but remain in the
        source Parquet file for inspection and evaluation.
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
        Lazily initialize documents and the Minsearch index.
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
        *,
        score: float | None = None,
    ) -> SearchResult:
        """
        Convert one retrieval dictionary into SearchResult.
        """
        normalized_score = score if score is not None else _extract_score(result)

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
            score=normalized_score,
        )


class PMPlaybookHybridSearch:
    """
    Hybrid retrieval using text search, vector search, and RRF.

    This implementation uses the production configuration selected during
    Phase 3B evaluation:

    - Minsearch text candidates: 20
    - SentenceTransformer vector candidates: 20
    - Embedding model: all-MiniLM-L6-v2
    - Reciprocal Rank Fusion constant: 60
    """

    def __init__(
        self,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
        embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
        embedding_chunk_ids_path: str | Path = DEFAULT_EMBEDDING_CHUNK_IDS_PATH,
        *,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        exclude_sponsors: bool = True,
    ) -> None:
        self.text_search = PMPlaybookSearch(
            chunks_path=chunks_path,
            exclude_sponsors=exclude_sponsors,
        )

        self.embeddings_path = Path(embeddings_path)
        self.embedding_chunk_ids_path = Path(embedding_chunk_ids_path)
        self.embedding_model_name = embedding_model_name

        self.embedding_model: SentenceTransformer | None = None
        self.document_embeddings: np.ndarray | None = None
        self.embedding_chunk_ids: list[str] = []

        self.documents_by_chunk_id: dict[str, dict[str, Any]] = {}
        self.embedding_index_by_chunk_id: dict[str, int] = {}

    def load_vector_artifacts(self) -> None:
        """
        Load the saved embedding matrix and aligned chunk-ID list.
        """
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embedding file does not exist: {self.embeddings_path}\n"
                "Generate it with:\n"
                "  uv run python scripts/build_embeddings.py"
            )

        if not self.embedding_chunk_ids_path.exists():
            raise FileNotFoundError(
                "Embedding chunk-ID file does not exist: "
                f"{self.embedding_chunk_ids_path}\n"
                "Generate it with:\n"
                "  uv run python scripts/build_embeddings.py"
            )

        self.document_embeddings = np.load(
            self.embeddings_path,
            mmap_mode="r",
        )

        with self.embedding_chunk_ids_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_chunk_ids = json.load(file)

        if not isinstance(raw_chunk_ids, list):
            raise ValueError("Embedding chunk-ID artifact must contain a JSON list.")

        self.embedding_chunk_ids = [str(chunk_id) for chunk_id in raw_chunk_ids]

        if self.document_embeddings.ndim != 2:
            raise ValueError(
                "Embedding matrix must be two-dimensional. "
                f"Received shape: {self.document_embeddings.shape}"
            )

        if self.document_embeddings.shape[0] != len(self.embedding_chunk_ids):
            raise ValueError(
                "Embedding matrix and chunk-ID artifact are misaligned: "
                f"{self.document_embeddings.shape[0]} embedding rows versus "
                f"{len(self.embedding_chunk_ids)} chunk IDs."
            )

        if len(set(self.embedding_chunk_ids)) != len(self.embedding_chunk_ids):
            raise ValueError(
                "Embedding chunk-ID artifact contains duplicate chunk IDs."
            )

        self.embedding_index_by_chunk_id = {
            chunk_id: index for index, chunk_id in enumerate(self.embedding_chunk_ids)
        }

    def ensure_ready(self) -> None:
        """
        Lazily initialize text retrieval, vector artifacts, and the model.
        """
        self.text_search.ensure_ready()

        if not self.documents_by_chunk_id:
            self.documents_by_chunk_id = {
                str(document["chunk_id"]): document
                for document in self.text_search.documents
            }

        if self.document_embeddings is None:
            self.load_vector_artifacts()

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        missing_document_ids = [
            chunk_id
            for chunk_id in self.embedding_chunk_ids
            if chunk_id not in self.documents_by_chunk_id
        ]

        if missing_document_ids:
            preview = missing_document_ids[:5]

            raise ValueError(
                "Some embedding chunk IDs are missing from the searchable "
                f"document corpus. Example IDs: {preview}"
            )

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        guest: str | None = None,
        speaker_name: str | None = None,
        boost_dict: dict[str, float] | None = None,
        text_candidates: int = DEFAULT_TEXT_CANDIDATES,
        vector_candidates: int = DEFAULT_VECTOR_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[SearchResult]:
        """
        Search transcript chunks using hybrid RRF retrieval.
        """
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        if num_results < 1:
            raise ValueError("num_results must be at least 1.")

        if text_candidates < 1:
            raise ValueError("text_candidates must be at least 1.")

        if vector_candidates < 1:
            raise ValueError("vector_candidates must be at least 1.")

        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        self.ensure_ready()

        if self.embedding_model is None:
            raise RuntimeError("Embedding model was not initialized.")

        if self.document_embeddings is None:
            raise RuntimeError("Document embeddings were not loaded.")

        effective_text_candidates = max(
            text_candidates,
            num_results,
        )
        effective_vector_candidates = max(
            vector_candidates,
            num_results,
        )

        text_results = self.text_search.search(
            cleaned_query,
            num_results=effective_text_candidates,
            guest=guest,
            speaker_name=speaker_name,
            boost_dict=boost_dict,
        )

        text_chunk_ids = [result.chunk_id for result in text_results]

        query_embedding = self.embedding_model.encode(
            [cleaned_query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        vector_chunk_ids = self._vector_search_chunk_ids(
            query_embedding,
            num_results=effective_vector_candidates,
            guest=guest,
            speaker_name=speaker_name,
        )

        fused_chunk_ids, fused_scores = reciprocal_rank_fusion_with_scores(
            text_chunk_ids,
            vector_chunk_ids,
            rrf_k=rrf_k,
            limit=num_results,
        )

        results: list[SearchResult] = []

        for chunk_id in fused_chunk_ids:
            document = self.documents_by_chunk_id.get(chunk_id)

            if document is None:
                continue

            results.append(
                PMPlaybookSearch._normalize_result(
                    document,
                    score=fused_scores[chunk_id],
                )
            )

        return results

    def _vector_search_chunk_ids(
        self,
        query_embedding: np.ndarray,
        *,
        num_results: int,
        guest: str | None,
        speaker_name: str | None,
    ) -> list[str]:
        """
        Return vector-ranked chunk IDs with optional exact metadata filters.
        """
        if self.document_embeddings is None:
            raise RuntimeError("Document embeddings were not loaded.")

        scores = self.document_embeddings @ query_embedding

        eligible_indices: list[int] = []

        for index, chunk_id in enumerate(self.embedding_chunk_ids):
            document = self.documents_by_chunk_id[chunk_id]

            if guest and str(document["guest"]) != guest:
                continue

            if speaker_name and str(document["speaker_name"]) != speaker_name:
                continue

            eligible_indices.append(index)

        if not eligible_indices:
            return []

        eligible_indices_array = np.asarray(
            eligible_indices,
            dtype=np.int64,
        )
        eligible_scores = scores[eligible_indices_array]

        result_count = min(
            num_results,
            len(eligible_indices),
        )

        ranked_local_indices = np.argsort(
            eligible_scores,
        )[::-1][:result_count]

        ranked_embedding_indices = eligible_indices_array[ranked_local_indices]

        return [
            self.embedding_chunk_ids[int(index)] for index in ranked_embedding_indices
        ]


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
    Extract a relevance score when exposed by Minsearch.

    Minsearch result dictionaries may not include a public score field,
    so the score remains optional.
    """
    for key in ("score", "_score", "relevance_score"):
        value = result.get(key)

        if value is not None:
            return float(value)

    return None


def reciprocal_rank_fusion_with_scores(
    text_chunk_ids: list[str],
    vector_chunk_ids: list[str],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int = 5,
) -> tuple[list[str], dict[str, float]]:
    """
    Combine two ranked chunk-ID lists using Reciprocal Rank Fusion.

    Returns both the ranked IDs and their fused scores.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    fused_scores: dict[str, float] = {}

    for ranked_chunk_ids in (
        text_chunk_ids,
        vector_chunk_ids,
    ):
        for rank, chunk_id in enumerate(
            ranked_chunk_ids,
            start=1,
        ):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            fused_scores[chunk_id],
            chunk_id,
        ),
        reverse=True,
    )

    return ranked_chunk_ids[:limit], fused_scores


def reciprocal_rank_fusion(
    text_chunk_ids: list[str],
    vector_chunk_ids: list[str],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int = 5,
) -> list[str]:
    """
    Combine two ranked chunk-ID lists using Reciprocal Rank Fusion.
    """
    ranked_chunk_ids, _ = reciprocal_rank_fusion_with_scores(
        text_chunk_ids,
        vector_chunk_ids,
        rrf_k=rrf_k,
        limit=limit,
    )

    return ranked_chunk_ids


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
    Convenience function for baseline text search.

    For repeated queries, instantiate PMPlaybookSearch once rather than
    rebuilding the index for every call.
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


def hybrid_search_documents(
    query: str,
    *,
    num_results: int = 5,
    guest: str | None = None,
    speaker_name: str | None = None,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
    embedding_chunk_ids_path: str | Path = DEFAULT_EMBEDDING_CHUNK_IDS_PATH,
) -> list[dict[str, Any]]:
    """
    Convenience function for hybrid retrieval.

    For an application, instantiate PMPlaybookHybridSearch once so the index,
    embedding matrix, and model are reused across queries.
    """
    search_engine = PMPlaybookHybridSearch(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        embedding_chunk_ids_path=embedding_chunk_ids_path,
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
    Run a hybrid-retrieval command-line smoke test.
    """
    query = "How do I know if I have product-market fit?"

    search_engine = PMPlaybookHybridSearch()

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
        print(f"RRF score: {result.score}")
        print(f"Words: {result.word_count}")
        print()
        print(result.text[:800])
        print()


if __name__ == "__main__":
    main()
