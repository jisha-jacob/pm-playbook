"""
Streamlit monitoring dashboard for PM Playbook.

Run the full app from the repository root with:

    uv run streamlit run streamlit_app/Home.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

db_module = import_module("pm_playbook.db")
get_engine = db_module.get_engine


st.set_page_config(
    page_title="PM Playbook Monitoring",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=30, show_spinner=False)
def load_conversations() -> pd.DataFrame:
    """Load all saved conversations from PostgreSQL."""
    query = """
        SELECT
            id,
            question,
            answer,
            model,
            tokens_input,
            tokens_output,
            tokens_total,
            cost_usd,
            latency_ms,
            relevance,
            created_at
        FROM conversations
        ORDER BY created_at ASC
    """

    return pd.read_sql(
        query,
        get_engine(),
        parse_dates=["created_at"],
    )


@st.cache_data(ttl=30, show_spinner=False)
def load_feedback() -> pd.DataFrame:
    """Load all saved feedback from PostgreSQL."""
    query = """
        SELECT
            id,
            conversation_id,
            feedback,
            created_at
        FROM feedback
        ORDER BY created_at ASC
    """

    return pd.read_sql(
        query,
        get_engine(),
        parse_dates=["created_at"],
    )


def prepare_feedback_summary(
    conversations_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build positive, negative, and no-feedback counts."""
    positive_count = int((feedback_df["feedback"] == 1).sum())
    negative_count = int((feedback_df["feedback"] == -1).sum())

    rated_conversations = (
        feedback_df["conversation_id"].nunique() if not feedback_df.empty else 0
    )

    no_feedback_count = max(
        len(conversations_df) - rated_conversations,
        0,
    )

    return pd.DataFrame(
        {
            "Feedback": [
                "Positive",
                "Negative",
                "No feedback",
            ],
            "Count": [
                positive_count,
                negative_count,
                no_feedback_count,
            ],
        }
    ).set_index("Feedback")


def render_summary_metrics(
    conversations_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> None:
    """Render top-level monitoring metrics."""
    total_conversations = len(conversations_df)
    total_feedback = len(feedback_df)

    positive_feedback = int((feedback_df["feedback"] == 1).sum())

    positive_rate = positive_feedback / total_feedback * 100 if total_feedback else 0.0

    average_latency = (
        conversations_df["latency_ms"].mean() if total_conversations else 0.0
    )

    average_tokens = (
        conversations_df["tokens_total"].mean() if total_conversations else 0.0
    )

    column_1, column_2, column_3, column_4 = st.columns(4)

    column_1.metric(
        "Conversations",
        total_conversations,
    )
    column_2.metric(
        "Feedback responses",
        total_feedback,
    )
    column_3.metric(
        "Positive feedback",
        f"{positive_rate:.1f}%",
    )
    column_4.metric(
        "Average latency",
        f"{average_latency:,.0f} ms",
    )

    st.caption(f"Average total tokens per answer: {average_tokens:,.0f}")


def render_charts(
    conversations_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> None:
    """Render the five monitoring charts required by the project."""
    chart_column_1, chart_column_2 = st.columns(2)

    daily_conversations = (
        conversations_df.assign(date=conversations_df["created_at"].dt.date)
        .groupby("date")
        .size()
        .rename("Conversations")
        .to_frame()
    )

    with chart_column_1:
        st.subheader("1. Conversations over time")
        st.bar_chart(daily_conversations)

    feedback_summary = prepare_feedback_summary(
        conversations_df,
        feedback_df,
    )

    with chart_column_2:
        st.subheader("2. Feedback breakdown")
        st.bar_chart(feedback_summary)

    token_usage = conversations_df[
        [
            "created_at",
            "tokens_input",
            "tokens_output",
        ]
    ].copy()

    token_usage = token_usage.set_index("created_at")

    with chart_column_1:
        st.subheader("3. Token usage over time")
        st.line_chart(token_usage)

    latency_over_time = conversations_df[
        [
            "created_at",
            "latency_ms",
        ]
    ].copy()

    latency_over_time = latency_over_time.set_index("created_at")

    with chart_column_2:
        st.subheader("4. Response latency over time")
        st.line_chart(latency_over_time)

    model_usage = (
        conversations_df.groupby("model").size().rename("Conversations").to_frame()
    )

    with chart_column_1:
        st.subheader("5. Model usage")
        st.bar_chart(model_usage)

    cost_over_time = conversations_df[
        [
            "created_at",
            "cost_usd",
        ]
    ].copy()

    cost_over_time = cost_over_time.set_index("created_at")

    with chart_column_2:
        st.subheader("6. Recorded cost over time")
        st.line_chart(cost_over_time)


def render_recent_conversations(
    conversations_df: pd.DataFrame,
) -> None:
    """Render the latest conversation records."""
    st.subheader("Recent conversations")

    recent_df = conversations_df.sort_values(
        "created_at",
        ascending=False,
    ).head(50)[
        [
            "id",
            "created_at",
            "question",
            "answer",
            "model",
            "tokens_total",
            "latency_ms",
            "relevance",
        ]
    ]

    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Render the monitoring dashboard."""
    st.title("📊 PM Playbook Monitoring")
    st.caption(
        "Live usage, feedback, token, latency, and model metrics from PostgreSQL."
    )

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    try:
        conversations_df = load_conversations()
        feedback_df = load_feedback()
    except Exception as error:
        st.error(
            "Could not load monitoring data. Make sure PostgreSQL "
            "is running with `docker compose up -d postgres`."
        )
        st.exception(error)
        st.stop()

    if conversations_df.empty:
        st.info(
            "No conversations have been logged yet. "
            "Use the chat page to generate an answer."
        )
        st.stop()

    render_summary_metrics(
        conversations_df,
        feedback_df,
    )

    st.divider()

    render_charts(
        conversations_df,
        feedback_df,
    )

    st.divider()

    render_recent_conversations(conversations_df)


if __name__ == "__main__":
    main()
