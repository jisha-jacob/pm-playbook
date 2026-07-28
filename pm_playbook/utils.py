"""
Utility functions for the PM Playbook ingestion pipeline.
"""

import hashlib
import re
from typing import Optional

from pm_playbook.constants import (
    BACKCHANNEL_PHRASES,
    PROMOTIONAL_PATTERNS,
    SPONSOR_ANCHOR_PATTERNS,
)


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


def is_backchannel(text: str) -> bool:
    """
    Return True when text is only a short acknowledgement.

    Longer statements beginning with words such as "Exactly" or
    "Right" are preserved.
    """
    normalized = clean_text(text).casefold()

    # Remove punctuation but preserve word characters, spaces,
    # and hyphens used in phrases such as "mm-hmm".
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    normalized = clean_text(normalized)

    return normalized in BACKCHANNEL_PHRASES


def is_sponsor_read(text: str) -> bool:
    """
    Detect likely sponsor-read content.

    A strong sponsorship phrase is sufficient by itself. Weaker
    promotional phrases require at least two matches to reduce
    false positives.
    """
    normalized = clean_text(text).casefold()

    has_sponsor_anchor = any(
        pattern in normalized for pattern in SPONSOR_ANCHOR_PATTERNS
    )

    if has_sponsor_anchor:
        return True

    promotional_matches = sum(pattern in normalized for pattern in PROMOTIONAL_PATTERNS)

    return promotional_matches >= 2


def generate_episode_id(
    guest: str,
    title: str,
    video_id: str | None = None,
    transcript_path: str | None = None,
) -> str:
    """
    Generate a deterministic episode ID.

    Prefer the YouTube video ID because it uniquely identifies an episode.
    Fall back to the transcript path when the video ID is missing.
    """
    if video_id:
        key = f"video_id|{video_id}"
    elif transcript_path:
        key = f"path|{transcript_path}"
    else:
        key = f"metadata|{guest}|{title}"

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
