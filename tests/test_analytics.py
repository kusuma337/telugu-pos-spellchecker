# -*- coding: utf-8 -*-
"""Tests for app/analytics.py helper module."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analytics import (
    detect_script,
    get_pos_distribution,
    get_text_analytics,
    introduce_noise,
)


# ---------------------------------------------------------------------------
# detect_script
# ---------------------------------------------------------------------------

def test_detect_script_telugu():
    assert detect_script("నేను పాఠశాలకు వెళ్తున్నాను") == "telugu"


def test_detect_script_english():
    assert detect_script("Hello world") == "english"


def test_detect_script_mixed():
    assert detect_script("నేను Hello") == "mixed"


def test_detect_script_empty():
    assert detect_script("") == "empty"
    assert detect_script("   ") == "empty"
    assert detect_script(None) == "empty"  # type: ignore[arg-type]


def test_detect_script_numeric():
    assert detect_script("123 456") == "numeric"


# ---------------------------------------------------------------------------
# get_text_analytics
# ---------------------------------------------------------------------------

def test_get_text_analytics_basic():
    text = "నేను పాఠశాలకు వెళ్తున్నాను."
    # Simulate a minimal result dict
    result = {
        "original_tokens": ["నేను", "పాఠశాలకు", "వెళ్తున్నాను", "."],
        "misspellings": [],
    }
    analytics = get_text_analytics(text, result)
    assert analytics["word_count"] >= 3
    assert analytics["char_count"] == len(text)
    assert analytics["unique_words"] >= 1
    assert analytics["avg_word_length"] > 0
    assert analytics["misspelling_count"] == 0


def test_get_text_analytics_misspellings():
    text = "నేను పాఠశాలక"
    result = {
        "original_tokens": ["నేను", "పాఠశాలక"],
        "misspellings": [{"original": "పాఠశాలక", "corrected": "పాఠశాలకు"}],
    }
    analytics = get_text_analytics(text, result)
    assert analytics["misspelling_count"] == 1


def test_get_text_analytics_empty_result():
    """Should not crash with minimal/empty result dict."""
    analytics = get_text_analytics("నేను", {})
    assert isinstance(analytics["char_count"], int)
    assert isinstance(analytics["word_count"], int)


# ---------------------------------------------------------------------------
# get_pos_distribution
# ---------------------------------------------------------------------------

def test_get_pos_distribution():
    tagged = [
        {"word": "నేను", "tag": "PRON"},
        {"word": "పాఠశాలకు", "tag": "NOUN"},
        {"word": "వెళ్తున్నాను", "tag": "VERB"},
        {"word": ".", "tag": "PUNCT"},
        {"word": "అతను", "tag": "PRON"},
    ]
    dist = get_pos_distribution(tagged)
    assert isinstance(dist, Counter)
    assert dist["PRON"] == 2
    assert dist["NOUN"] == 1
    assert dist["VERB"] == 1
    assert dist["PUNCT"] == 1


def test_get_pos_distribution_empty():
    dist = get_pos_distribution([])
    assert isinstance(dist, Counter)
    assert len(dist) == 0


def test_get_pos_distribution_none():
    dist = get_pos_distribution(None)  # type: ignore[arg-type]
    assert isinstance(dist, Counter)


# ---------------------------------------------------------------------------
# introduce_noise
# ---------------------------------------------------------------------------

def test_introduce_noise_changes_words():
    """With noise_level=1.0, Telugu words should (almost always) be changed."""
    import random
    random.seed(42)
    tokens = ["నేను", "పాఠశాలకు", "వెళ్తున్నాను"]
    noisy = introduce_noise(tokens, noise_level=1.0)
    assert len(noisy) == len(tokens)
    # At least one token must differ from original with certainty at noise=1.0
    assert any(o != n for o, n in zip(tokens, noisy)), "No tokens were changed at noise_level=1.0"


def test_introduce_noise_preserves_length():
    tokens = ["నేను", "పాఠశాలకు", ".", "వెళ్తున్నాను"]
    noisy = introduce_noise(tokens, noise_level=0.5)
    assert len(noisy) == len(tokens)


def test_introduce_noise_zero():
    """At noise_level=0.0, no tokens should ever change."""
    tokens = ["నేను", "పాఠశాలకు", "వెళ్తున్నాను"]
    noisy = introduce_noise(tokens, noise_level=0.0)
    assert noisy == tokens


def test_introduce_noise_skips_punctuation():
    """Punctuation-only tokens must never be changed."""
    tokens = [".", ",", "!", "నేను"]
    noisy = introduce_noise(tokens, noise_level=1.0)
    assert noisy[0] == "."
    assert noisy[1] == ","
    assert noisy[2] == "!"


def test_introduce_noise_skips_short_tokens():
    """Single-character tokens are too short to corrupt."""
    tokens = ["న"]  # length 1, should be skipped
    noisy = introduce_noise(tokens, noise_level=1.0)
    assert noisy == tokens
