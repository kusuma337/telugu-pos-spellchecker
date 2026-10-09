"""Telugu-aware text preprocessing utilities."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

TELUGU_RANGE = "\u0C00-\u0C7F"
PUNCTUATION = r""".,;:!?।॥"'`()[]{}<>/\\|@#$%^&*+=~`"""

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(
    rf"([{TELUGU_RANGE}A-Za-z0-9]+|[{re.escape(PUNCTUATION)}])",
    re.UNICODE,
)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC so equivalent Telugu strings match."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text))


def is_telugu_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return 0x0C00 <= code <= 0x0C7F


def contains_telugu(text: str) -> bool:
    return any(is_telugu_char(ch) for ch in text or "")


def clean_text(text: str | None) -> str:
    """Normalize Unicode and collapse whitespace without dropping Telugu letters."""
    if text is None:
        return ""
    normalized = normalize_unicode(text)
    cleaned_chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("C") and char not in "\n\t":
            continue
        cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def tokenize(text: str | None) -> list[str]:
    """Split a Telugu sentence into words and punctuation tokens."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    tokens = _TOKEN_RE.findall(cleaned)
    return [token for token in tokens if token.strip()]


def detokenize(tokens: Iterable[str]) -> str:
    tokens = [str(t) for t in tokens if str(t).strip()]
    if not tokens:
        return ""
    parts: list[str] = []
    for token in tokens:
        if token in PUNCTUATION and parts:
            parts[-1] = parts[-1] + token
        else:
            parts.append(token)
    return " ".join(parts)


def is_punctuation(token: str) -> bool:
    return bool(token) and all(ch in PUNCTUATION for ch in token)


def safe_tokens(tokens: Iterable[str] | None) -> list[str]:
    if not tokens:
        return []
    return [normalize_unicode(str(t)).strip() for t in tokens if str(t).strip()]
