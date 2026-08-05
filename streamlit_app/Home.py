"""
Streamlit chat interface for PM Playbook.

Run from the repository root with:

    uv run streamlit run streamlit_app/Home.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PMPlaybookRAG = import_module("pm_playbook.rag").PMPlaybookRAG


st.set_page_config(
    page_title="PM Playbook",
    page_icon="📘",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading PM Playbook...")
def get_rag_service() -> Any:
    """
    Create and cache the RAG service.

    Caching prevents the Minsearch index, embedding matrix, and
    SentenceTransformer model from being loaded again on every rerun.
    """
    return PMPlaybookRAG()


def initialize_session_state() -> None:
    """Initialize chat history for the current Streamlit session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_source(source: dict[str, Any]) -> None:
    """Render one expandable source citation."""
    source_number = source["source_number"]
    guest = source["guest"]
    episode_title = source["episode_title"]
    speaker_name = source["speaker_name"]
    timestamp = source.get("timestamp") or "Unknown"
    youtube_url = source.get("youtube_url")
    text = source["text"]

    label = f"Source {source_number}: {guest} — {episode_title}"

    with st.expander(label):
        st.write(f"**Actual speaker:** {speaker_name}")
        st.write(f"**Timestamp:** {timestamp}")

        if youtube_url:
            st.link_button(
                "Open episode on YouTube",
                youtube_url,
            )

        st.write("**Transcript excerpt**")
        st.write(text)


def render_assistant_message(message: dict[str, Any]) -> None:
    """Render a saved assistant message and its supporting sources."""
    st.markdown(message["content"])

    sources = message.get("sources", [])

    if sources:
        st.markdown("#### Sources")

        for source in sources:
            render_source(source)

    metadata = message.get("metadata")

    if metadata:
        with st.expander("Response details"):
            st.write(f"**Model:** {metadata['model']}")
            st.write(f"**Sources used:** {metadata['retrieval_results']}")
            st.write(f"**Input tokens:** {metadata['input_tokens']}")
            st.write(f"**Output tokens:** {metadata['output_tokens']}")
            st.write(f"**Total tokens:** {metadata['total_tokens']}")
            st.write(f"**Latency:** {metadata['latency_ms']} ms")


def render_chat_history() -> None:
    """Render all messages stored in the current session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])


def main() -> None:
    """Render the PM Playbook Streamlit application."""
    initialize_session_state()

    st.title("📘 PM Playbook")
    st.caption(
        "Ask product-management questions and get answers grounded "
        "in Lenny's Podcast transcripts."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "PM Playbook uses hybrid text and vector retrieval, "
            "Reciprocal Rank Fusion, and GPT-4o mini."
        )
        st.write(
            "Answers are limited to the retrieved podcast excerpts "
            "and include their supporting sources."
        )

        if st.button(
            "Clear conversation",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

    render_chat_history()

    question = st.chat_input("Ask a product-management question...")

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the playbook and generating an answer..."):
            try:
                rag_service = get_rag_service()
                result = rag_service.answer(question=question)
            except Exception as error:
                st.error(
                    "PM Playbook could not generate an answer. "
                    "Check the terminal for details."
                )
                st.exception(error)
                return

        result_dict = result.to_dict()

        assistant_message = {
            "role": "assistant",
            "content": result.answer,
            "sources": result_dict["sources"],
            "metadata": {
                "model": result.model,
                "retrieval_results": result.retrieval_results,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
                "response_id": result.response_id,
            },
        }

        render_assistant_message(assistant_message)

    st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
