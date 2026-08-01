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

SPONSOR_ANCHOR_PATTERNS = [
    "brought to you by",
    "sponsored by",
    "thanks to our sponsor",
    "thanks to our sponsors",
    "a word from our sponsor",
    "a word from our sponsors",
    "support for this episode comes from",
    "this episode is brought to you by",
    "today's episode is brought to you by",
    "this episode is sponsored by",
    "our first sponsor today is",
    "our next sponsor is",
]

# Weaker promotional signals. A chunk must contain multiple
# signals before it is classified as a sponsor read.
PROMOTIONAL_PATTERNS = [
    "use code",
    "promo code",
    "discount code",
    "special offer",
    "free trial",
    "sign up",
    "learn more at",
    "visit",
    ".com",
    "dot com",
    "percent off",
    "save twenty percent",
    "save 20 percent",
]

# ==========================================================
# Backchannel Detection
# ==========================================================

# Short acknowledgements that contain little or no searchable
# product-management information on their own.
BACKCHANNEL_PHRASES = {
    "absolutely",
    "agreed",
    "correct",
    "exactly",
    "for sure",
    "got it",
    "interesting",
    "makes sense",
    "mhm",
    "mm-hmm",
    "okay",
    "ok",
    "right",
    "sure",
    "totally",
    "uh-huh",
    "yeah",
    "yep",
    "yes",
    "yup",
}


# ==========================================================
# Speaker Detection Regex Patterns
# ==========================================================

# Example:
# Brian Chesky (00:00:00): Hello everyone...
SPEAKER_PATTERN_HHMMSS = r"^([A-Za-z\s\-\']+)\s*\((\d{2}:\d{2}:\d{2})\):\s*(.*)$"

# Example:
# Brian Chesky (12:34): Hello everyone...
SPEAKER_PATTERN_HHMM = r"^([A-Za-z\s\-\']+)\s*\((\d{2}:\d{2})\):\s*(.*)$"

# Example:
# Brian Chesky: Hello everyone...
SPEAKER_PATTERN_NAME_ONLY = (
    r"^("
    r"[A-Z][A-Za-z.'\-]*"
    r"(?:\s+(?:[A-Z][A-Za-z.'\-]*|and|&|\+)){0,5}"
    r"):\s*(.*)$"
)

# Continuation line:
# (00:03:45): Continued discussion...
CONTINUATION_PATTERN_HHMMSS = r"^\s*\((\d{2}:\d{2}:\d{2})\):\s*(.*)$"

# Continuation line:
# (12:45): Continued discussion...
CONTINUATION_PATTERN_HHMM = r"^\s*\((\d{2}:\d{2})\):\s*(.*)$"


# ==========================================================
# Output Configuration
# ==========================================================

OUTPUT_PARQUET = "data/chunks.parquet"
