"""
PostgreSQL persistence helpers for PM Playbook.

This module stores RAG conversations and user feedback using SQLAlchemy Core.
It also estimates text-generation cost from model token usage.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine


metadata = MetaData()


MODEL_PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
}


conversations = Table(
    "conversations",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("sources", JSONB, nullable=False),
    Column("model", String(100), nullable=False),
    Column("tokens_input", Integer, nullable=False, default=0),
    Column("tokens_output", Integer, nullable=False, default=0),
    Column("tokens_total", Integer, nullable=False, default=0),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("latency_ms", Integer, nullable=False, default=0),
    Column("relevance", String(50), nullable=True),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)


feedback = Table(
    "feedback",
    metadata,
    Column("id", Integer, primary_key=True),
    Column(
        "conversation_id",
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("feedback", Integer, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)


def get_database_url() -> str:
    """
    Build the PostgreSQL connection URL from environment variables.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "pm_playbook")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Create and cache the SQLAlchemy engine.
    """
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )


def create_tables() -> None:
    """
    Create the conversations and feedback tables if they do not exist.
    """
    metadata.create_all(get_engine())


def estimate_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate standard API text-token cost for a supported model.

    The result is rounded to eight decimal places because individual
    gpt-4o-mini requests often cost much less than one cent.
    """
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(model)

    if pricing is None:
        return 0.0

    input_cost = max(input_tokens, 0) / 1_000_000 * pricing["input"]
    output_cost = max(output_tokens, 0) / 1_000_000 * pricing["output"]

    return round(input_cost + output_cost, 8)


def save_conversation(
    *,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    model: str,
    tokens_input: int,
    tokens_output: int,
    tokens_total: int,
    cost_usd: float,
    latency_ms: int,
    relevance: str | None = None,
) -> int:
    """
    Insert one conversation and return its database ID.
    """
    statement = (
        insert(conversations)
        .values(
            question=question,
            answer=answer,
            sources=sources,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_total=tokens_total,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            relevance=relevance,
        )
        .returning(conversations.c.id)
    )

    with get_engine().begin() as connection:
        conversation_id = connection.execute(statement).scalar_one()

    return int(conversation_id)


def save_feedback(
    *,
    conversation_id: int,
    feedback_value: int,
) -> int:
    """
    Insert thumbs-up or thumbs-down feedback.

    feedback_value must be 1 for positive feedback or -1 for negative feedback.
    """
    if feedback_value not in (-1, 1):
        raise ValueError("feedback_value must be either -1 or 1.")

    statement = (
        insert(feedback)
        .values(
            conversation_id=conversation_id,
            feedback=feedback_value,
        )
        .returning(feedback.c.id)
    )

    with get_engine().begin() as connection:
        feedback_id = connection.execute(statement).scalar_one()

    return int(feedback_id)
