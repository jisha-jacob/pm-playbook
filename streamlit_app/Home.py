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

rag_module = import_module("pm_playbook.rag")
db_module = import_module("pm_playbook.db")

PMPlaybookRAG = rag_module.PMPlaybookRAG
create_tables = db_module.create_tables
estimate_cost_usd = db_module.estimate_cost_usd
save_conversation = db_module.save_conversation
save_feedback = db_module.save_feedback


st.set_page_config(
    page_title="PM Playbook",
    page_icon="📘",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading PM Playbook...")
def get_rag_service() -> Any:
    """
    Create and cache the RAG service.

    Caching prevents the search index, embedding matrix, and embedding model
    from being loaded again on every Streamlit rerun.
    """
    return PMPlaybookRAG()


@st.cache_resource(show_spinner=False)
def initialize_database() -> bool:
    """
    Create database tables if needed.

    Returns True when the database is available. Database failures are handled
    in the UI so the chat interface can still display a useful error.
    """
    create_tables()
    return True


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


def submit_feedback(
    *,
    message_index: int,
    conversation_id: int,
    feedback_value: int,
) -> None:
    """Save one feedback value and mark the message as rated."""
    try:
        save_feedback(
            conversation_id=conversation_id,
            feedback_value=feedback_value,
        )
    except Exception as error:
        st.error(f"Could not save feedback: {error}")
        return

    st.session_state.messages[message_index]["feedback"] = feedback_value
    st.rerun()


def render_feedback(
    *,
    message: dict[str, Any],
    message_index: int,
) -> None:
    """Render thumbs-up and thumbs-down buttons for one saved answer."""
    conversation_id = message.get("conversation_id")

    if conversation_id is None:
        return

    submitted_feedback = message.get("feedback")

    if submitted_feedback == 1:
        st.success("Thanks for the positive feedback.")
        return

    if submitted_feedback == -1:
        st.info("Thanks for the feedback.")
        return

    st.caption("Was this answer helpful?")

    positive_column, negative_column = st.columns(2)

    with positive_column:
        if st.button(
            "👍 Helpful",
            key=f"positive-feedback-{conversation_id}",
            use_container_width=True,
        ):
            submit_feedback(
                message_index=message_index,
                conversation_id=conversation_id,
                feedback_value=1,
            )

    with negative_column:
        if st.button(
            "👎 Not helpful",
            key=f"negative-feedback-{conversation_id}",
            use_container_width=True,
        ):
            submit_feedback(
                message_index=message_index,
                conversation_id=conversation_id,
                feedback_value=-1,
            )


def render_assistant_message(
    message: dict[str, Any],
    *,
    message_index: int,
) -> None:
    """Render a saved assistant message and its supporting information."""
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
            st.write(f"**Estimated API cost:** ${metadata.get('cost_usd', 0.0):.6f}")

    render_feedback(
        message=message,
        message_index=message_index,
    )


def render_chat_history() -> None:
    """Render all messages stored in the current session."""
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(
                    message,
                    message_index=message_index,
                )
            else:
                st.markdown(message["content"])


def main() -> None:
    """Render the PM Playbook Streamlit application."""
    initialize_session_state()

    try:
        initialize_database()
    except Exception as error:
        st.error(
            "The PostgreSQL database is unavailable. "
            "Start it with `docker compose up -d postgres`."
        )
        st.exception(error)
        st.stop()

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

        estimated_cost_usd = estimate_cost_usd(
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

        try:
            conversation_id = save_conversation(
                question=result.question,
                answer=result.answer,
                sources=result_dict["sources"],
                model=result.model,
                tokens_input=result.input_tokens,
                tokens_output=result.output_tokens,
                tokens_total=result.total_tokens,
                cost_usd=estimated_cost_usd,
                latency_ms=result.latency_ms,
            )
        except Exception as error:
            st.error(
                "The answer was generated, but it could not be saved "
                f"to PostgreSQL: {error}"
            )
            conversation_id = None

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
                "cost_usd": estimated_cost_usd,
                "response_id": result.response_id,
            },
            "conversation_id": conversation_id,
            "feedback": None,
        }

        st.session_state.messages.append(assistant_message)

        render_assistant_message(
            assistant_message,
            message_index=len(st.session_state.messages) - 1,
        )


if __name__ == "__main__":
    main()
