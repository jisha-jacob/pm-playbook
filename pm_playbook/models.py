"""
Data models for the PM Playbook ingestion pipeline.
"""

from typing import Optional

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """
    Represents a single podcast episode after parsing
    the markdown transcript.
    """

    guest: str
    title: str
    publish_date: Optional[str] = None
    youtube_url: Optional[str] = None
    video_id: Optional[str] = None

    transcript: str


class SpeakerTurn(BaseModel):
    """
    Represents one continuous speaker turn.
    """

    speaker_name: str
    start_time: Optional[int] = None
    text: str


class Chunk(BaseModel):
    """
    Represents one searchable chunk that will be stored
    in the knowledge base.
    """

    chunk_id: str

    episode_id: str

    chunk_index: int
    chunk_position: int

    guest: str
    episode_title: str

    publish_date: Optional[str] = None

    youtube_url: Optional[str] = None
    video_id: Optional[str] = None

    speaker_name: str

    start_time: Optional[int] = None
    end_time: Optional[int] = None

    text: str

    word_count: int = Field(..., ge=1)

    is_sponsor_read: bool = False