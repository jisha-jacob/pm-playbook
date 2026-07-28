"""
Parser for Lenny's Podcast transcript files.

Reads a markdown transcript, extracts the YAML frontmatter,
and returns an Episode object.
"""

from pathlib import Path

import yaml

from pm_playbook.models import Episode


def parse_transcript(file_path: str | Path) -> Episode:
    """
    Parse a transcript markdown file into an Episode object.

    Parameters
    ----------
    file_path : str | Path
        Path to transcript.md

    Returns
    -------
    Episode
    """

    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        content = f.read()

    if not content.startswith("---"):
        raise ValueError(f"{file_path} does not contain YAML frontmatter.")

    parts = content.split("---", maxsplit=2)

    if len(parts) != 3:
        raise ValueError(f"Could not parse frontmatter in {file_path}")

    yaml_text = parts[1]
    transcript = parts[2].strip()

    metadata = yaml.safe_load(yaml_text)

    return Episode(
        guest=metadata["guest"],
        title=metadata["title"],
        publish_date=metadata.get("publish_date"),
        youtube_url=metadata.get("youtube_url"),
        video_id=metadata.get("video_id"),
        transcript_path=str(file_path),
        transcript=transcript,
    )
