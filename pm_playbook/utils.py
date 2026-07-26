"""
Utility functions for the PM Playbook ingestion pipeline.
"""

import hashlib
import re
from typing import Optional

from pm_playbook.constants import SPONSOR_PATTERNS


def parse_timestamp(timestamp: Optional[str]) -> Optional[int]:
    """
    Convert a timestamp string into total seconds.

    Supported formats:
        HH:MM:SS
        MM:SS

    Returns:
        int: total seconds
        None: if timestamp is None or invalid
    """
    if not timestamp:
        return None

    parts = timestamp.split(":")

    try:
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds

        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds

    except ValueError:
        return None

    return None


def count_words(text: str) -> int:
    """
    Count the number of words in a text.
    """
    return len(text.split())


def is_sponsor_read(text: str) -> bool:
    """
    Determine whether a chunk is likely a sponsor read.
    """
    text = text.lower()

    return any(pattern in text for pattern in SPONSOR_PATTERNS)


def generate_episode_id(guest: str, title: str) -> str:
    """
    Generate a deterministic episode ID.
    """
    key = f"{guest}|{title}"

    return hashlib.md5(key.encode("utf-8")).hexdigest()


def generate_chunk_id(
    episode_id: str,
    chunk_position: int,
) -> str:
    """
    Generate a deterministic chunk ID.
    """
    key = f"{episode_id}|{chunk_position}"

    return hashlib.md5(key.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """
    Normalize whitespace.

    Removes duplicate spaces and blank lines.
    """
    text = re.sub(r"\s+", " ", text)

    return text.strip()