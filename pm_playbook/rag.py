"""
Hybrid retrieval-augmented generation flow for PM Playbook.

Pipeline:
    user question
        -> retrieve relevant transcript chunks using hybrid RRF search
        -> prioritize guest-spoken evidence from the strongest episode
        -> build speaker-aware grounded context
        -> call OpenAI
        -> return answer, sources, usage, and latency
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from pm_playbook.search import (
    DEFAULT_CHUNKS_PATH,
    PMPlaybookHybridSearch,
    PMPlaybookSearch,
    SearchResult,
    format_timestamp,
)


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_NUM_RESULTS = 5
DEFAULT_MAX_CONTEXT_CHARACTERS = 18_000


SYSTEM_INSTRUCTIONS = """
You are PM Playbook, a product-management assistant grounded in
transcripts from Lenny's Podcast.

Answer the user's question using only the supplied transcript excerpts.

Requirements:
- Synthesize a clear and actionable answer.
- Do not use outside knowledge.
- Cite supporting excerpts using labels such as [Source 1].
- Pay close attention to the Actual speaker field for every source.
- Attribute a claim to a guest only when that guest is the actual speaker.
- Treat statements by Lenny or another host as questions, framing, or
  summaries unless the excerpt clearly presents independent advice.
- Never turn a host's question or paraphrase into a direct guest claim.
- When host and guest excerpts differ, prioritize the guest's own words.
- Use only excerpts that directly help answer the question.
- You do not need to cite or mention every supplied source.
- Ignore tangential, promotional, introductory, or closing remarks.
- Do not add a recommendation merely because it appears in one excerpt.
- If the excerpts do not contain enough information, say so clearly.
- Do not invent quotations, sources, facts, or recommendations.
- Keep the answer focused and reasonably concise.
""".strip()


@dataclass
class Source:
    """Source metadata returned with a RAG answer."""

    source_number: int
    chunk_id: str
    episode_id: str
    guest: str
    episode_title: str
    speaker_name: str
    text: str
    youtube_url: str | None
    video_id: str | None
    start_time: int | None
    end_time: int | None
    timestamp: str | None
    chunk_position: int
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the source to a serializable dictionary."""
        return {
            "source_number": self.source_number,
            "chunk_id": self.chunk_id,
            "episode_id": self.episode_id,
            "guest": self.guest,
            "episode_title": self.episode_title,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "youtube_url": self.youtube_url,
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timestamp": self.timestamp,
            "chunk_position": self.chunk_position,
            "word_count": self.word_count,
        }


@dataclass
class RAGResult:
    """Complete result from one RAG request."""

    question: str
    answer: str
    sources: list[Source]
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    retrieval_results: int
    latency_ms: int
    response_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert the RAG result to a serializable dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "retrieval_results": self.retrieval_results,
            "latency_ms": self.latency_ms,
            "response_id": self.response_id,
        }


class PMPlaybookRAG:
    """Grounded RAG service using hybrid retrieval and OpenAI."""

    def __init__(
        self,
        *,
        search_engine: PMPlaybookSearch | PMPlaybookHybridSearch | None = None,
        client: OpenAI | None = None,
        model: str = DEFAULT_MODEL,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
        num_results: int = DEFAULT_NUM_RESULTS,
        max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS,
    ) -> None:
        """
        Initialize the RAG service.

        Parameters
        ----------
        search_engine:
            Optional prebuilt text or hybrid search engine. When omitted,
            hybrid RRF retrieval is used.
        client:
            Optional OpenAI client.
        model:
            OpenAI model used for answer generation.
        chunks_path:
            Path to the processed Parquet dataset.
        num_results:
            Default number of final sources per question.
        max_context_characters:
            Maximum number of context characters sent to the model.
        """
        if num_results < 1:
            raise ValueError("num_results must be at least 1.")

        if max_context_characters < 1:
            raise ValueError("max_context_characters must be at least 1.")

        load_dotenv()

        self.model = model
        self.num_results = num_results
        self.max_context_characters = max_context_characters

        self.search_engine = search_engine or PMPlaybookHybridSearch(
            chunks_path=chunks_path,
            exclude_sponsors=True,
        )

        self.client = client or self._create_openai_client()

    @staticmethod
    def _create_openai_client() -> OpenAI:
        """Create an OpenAI client after validating the API key."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to a local "
                ".env file or export it in the shell environment."
            )

        return OpenAI()

    @staticmethod
    def select_retrieval_results(
        retrieved_chunks: list[SearchResult],
        *,
        num_results: int,
    ) -> list[SearchResult]:
        """
        Select grounded sources while preserving hybrid relevance.

        The highest-ranked result establishes the primary episode. When that
        episode already provides at least three guest-spoken excerpts, only
        those excerpts are used. This avoids padding a strong answer with
        tangential material from weaker episodes.

        When the primary episode has fewer than three guest-spoken excerpts,
        other guest-spoken results are added in hybrid rank order. Host or
        other-speaker excerpts are used only as a final fallback.
        """
        if num_results < 1:
            raise ValueError("num_results must be at least 1.")

        if not retrieved_chunks:
            return []

        primary_episode_id = retrieved_chunks[0].episode_id

        primary_guest_spoken: list[SearchResult] = []
        other_guest_spoken: list[SearchResult] = []
        other_speakers: list[SearchResult] = []

        for result in retrieved_chunks:
            speaker_matches_guest = (
                result.speaker_name.strip().casefold()
                == result.guest.strip().casefold()
            )

            if speaker_matches_guest and result.episode_id == primary_episode_id:
                primary_guest_spoken.append(result)
            elif speaker_matches_guest:
                other_guest_spoken.append(result)
            else:
                other_speakers.append(result)

        if len(primary_guest_spoken) >= 3:
            return primary_guest_spoken[:num_results]

        selected = primary_guest_spoken + other_guest_spoken + other_speakers

        return selected[:num_results]

    def answer(
        self,
        question: str,
        *,
        num_results: int | None = None,
        guest: str | None = None,
        speaker_name: str | None = None,
    ) -> RAGResult:
        """
        Retrieve transcript chunks and generate a grounded answer.

        Hybrid retrieval supplies an expanded candidate set. Source selection
        prioritizes guest-spoken excerpts from the strongest matching episode,
        then preserves hybrid rank among the remaining guest-spoken evidence.
        """
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("question cannot be empty.")

        requested_results = num_results if num_results is not None else self.num_results

        if requested_results < 1:
            raise ValueError("num_results must be at least 1.")

        started_at = perf_counter()

        candidate_count = requested_results * 3

        retrieved_candidates = self.search_engine.search(
            query=cleaned_question,
            num_results=candidate_count,
            guest=guest,
            speaker_name=speaker_name,
        )

        retrieved_chunks = self.select_retrieval_results(
            retrieved_candidates,
            num_results=requested_results,
        )

        sources = self._create_sources(retrieved_chunks)
        context = self.build_context(sources)

        user_input = self.build_user_input(
            question=cleaned_question,
            context=context,
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=user_input,
            )
        except OpenAIError as error:
            raise RuntimeError(f"OpenAI request failed: {error}") from error

        latency_ms = round((perf_counter() - started_at) * 1000)

        input_tokens, output_tokens, total_tokens = self._extract_usage(response)

        answer_text = response.output_text.strip()

        if not answer_text:
            raise RuntimeError("OpenAI returned an empty text response.")

        return RAGResult(
            question=cleaned_question,
            answer=answer_text,
            sources=sources,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            retrieval_results=len(retrieved_chunks),
            latency_ms=latency_ms,
            response_id=getattr(response, "id", None),
        )

    def build_context(
        self,
        sources: list[Source],
    ) -> str:
        """
        Format retrieved chunks as labeled prompt context.

        Context is limited at source boundaries. Individual source
        excerpts are not cut in the middle.
        """
        if not sources:
            return "No transcript excerpts were retrieved."

        context_sections: list[str] = []
        current_length = 0

        for source in sources:
            timestamp = source.timestamp or "Unknown"

            speaker_matches_guest = (
                source.speaker_name.strip().casefold()
                == source.guest.strip().casefold()
            )

            speaker_role = (
                "episode guest" if speaker_matches_guest else "host or other speaker"
            )

            section = (
                f"[Source {source.source_number}]\n"
                f"Episode guest: {source.guest}\n"
                f"Episode: {source.episode_title}\n"
                f"Actual speaker: {source.speaker_name}\n"
                f"Speaker role: {speaker_role}\n"
                f"Timestamp: {timestamp}\n"
                f"Excerpt:\n{source.text}"
            )

            separator_length = 2 if context_sections else 0

            would_exceed_limit = (
                current_length + separator_length + len(section)
                > self.max_context_characters
            )

            if context_sections and would_exceed_limit:
                break

            context_sections.append(section)
            current_length += separator_length + len(section)

        return "\n\n".join(context_sections)

    @staticmethod
    def build_user_input(
        question: str,
        context: str,
    ) -> str:
        """Build the user portion of the grounded prompt."""
        return (
            "Transcript excerpts:\n\n"
            f"{context}\n\n"
            "User question:\n"
            f"{question}\n\n"
            "Provide a grounded answer with inline source citations."
        )

    @staticmethod
    def _create_sources(
        retrieved_chunks: list[SearchResult],
    ) -> list[Source]:
        """Convert search results into numbered source objects."""
        sources: list[Source] = []

        for index, result in enumerate(retrieved_chunks, start=1):
            source = Source(
                source_number=index,
                chunk_id=result.chunk_id,
                episode_id=result.episode_id,
                guest=result.guest,
                episode_title=result.episode_title,
                speaker_name=result.speaker_name,
                text=result.text,
                youtube_url=result.youtube_url,
                video_id=result.video_id,
                start_time=result.start_time,
                end_time=result.end_time,
                timestamp=format_timestamp(result.start_time),
                chunk_position=result.chunk_position,
                word_count=result.word_count,
            )

            sources.append(source)

        return sources

    @staticmethod
    def _extract_usage(
        response: Any,
    ) -> tuple[int, int, int]:
        """
        Extract token usage from an OpenAI response.

        Missing usage information is represented as zero.
        """
        usage = getattr(response, "usage", None)

        if usage is None:
            return 0, 0, 0

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        total_tokens = int(
            getattr(
                usage,
                "total_tokens",
                input_tokens + output_tokens,
            )
            or input_tokens + output_tokens
        )

        return input_tokens, output_tokens, total_tokens


def print_rag_result(result: RAGResult) -> None:
    """Print a readable development view of a RAG result."""
    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result.answer)

    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    if not result.sources:
        print("No sources were retrieved.")
        print()
    else:
        for source in result.sources:
            print(
                f"[Source {source.source_number}] "
                f"{source.guest} — {source.episode_title}"
            )
            print(
                f"Speaker: {source.speaker_name} | "
                f"Timestamp: {source.timestamp or 'Unknown'}"
            )
            print(f"Chunk ID: {source.chunk_id}")
            print()

    print("=" * 80)
    print("REQUEST METADATA")
    print("=" * 80)
    print(f"Model: {result.model}")
    print(f"Retrieved sources: {result.retrieval_results}")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Latency: {result.latency_ms} ms")
    print(f"Response ID: {result.response_id}")


def main() -> None:
    """Run an end-to-end hybrid RAG smoke test."""
    question = "How do I know if I have product-market fit?"

    rag = PMPlaybookRAG()

    result = rag.answer(
        question=question,
        num_results=5,
    )

    print(f"Question: {question}")
    print_rag_result(result)


if __name__ == "__main__":
    main()
