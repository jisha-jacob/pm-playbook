"""
Shared constants for the PM Playbook ingestion pipeline.
"""

# ==========================================================
# Chunking Configuration
# ==========================================================

# Target chunk size (words)
MIN_WORDS_PER_CHUNK = 150
MAX_WORDS_PER_CHUNK = 300

# Merge consecutive turns from the same speaker if they
# contain fewer than this many words.
MERGE_THRESHOLD = 50


# ==========================================================
# Sponsor Read Detection
# ==========================================================

SPONSOR_PATTERNS = [
    "brought to you by",
    "this episode is sponsored by",
    "today's episode is brought to you by",
    "this episode is brought to you by",
    "sponsored by",
]


# ==========================================================
# Speaker Detection Regex Patterns
# ==========================================================

# Example:
# Brian Chesky (00:00:00): Hello everyone...
SPEAKER_PATTERN_HHMMSS = (
    r"^([A-Za-z\s\-\']+)\s*\((\d{2}:\d{2}:\d{2})\):\s*(.*)$"
)

# Example:
# Brian Chesky (12:34): Hello everyone...
SPEAKER_PATTERN_HHMM = (
    r"^([A-Za-z\s\-\']+)\s*\((\d{2}:\d{2})\):\s*(.*)$"
)

# Example:
# Brian Chesky: Hello everyone...
SPEAKER_PATTERN_NAME_ONLY = (
    r"^([A-Z][a-zA-Z\s\-\']+):\s*(.*)$"
)

# Continuation line:
# (00:03:45): Continued discussion...
CONTINUATION_PATTERN_HHMMSS = (
    r"^\s*\((\d{2}:\d{2}:\d{2})\):\s*(.*)$"
)

# Continuation line:
# (12:45): Continued discussion...
CONTINUATION_PATTERN_HHMM = (
    r"^\s*\((\d{2}:\d{2})\):\s*(.*)$"
)


# ==========================================================
# Output Configuration
# ==========================================================

OUTPUT_PARQUET = "data/chunks.parquet"