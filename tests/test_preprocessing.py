"""Tests for Telugu preprocessing."""

from src.preprocessing import clean_text, contains_telugu, tokenize


def test_preserves_telugu_characters():
    text = "నేను పాఠశాలకు వెళ్తున్నాను"
    cleaned = clean_text(text)
    assert "నేను" in cleaned
    assert contains_telugu(cleaned)


def test_unicode_normalization():
    composed = "నేను"
    assert clean_text(composed) == composed


def test_whitespace_collapse():
    text = "నేను    పాఠశాలకు\tవెళ్తున్నాను"
    assert clean_text(text) == "నేను పాఠశాలకు వెళ్తున్నాను"


def test_empty_and_none_input():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert tokenize(None) == []
    assert tokenize("   ") == []


def test_punctuation_tokenization():
    tokens = tokenize("నేను వెళ్తున్నాను.")
    assert "నేను" in tokens
    assert "వెళ్తున్నాను" in tokens
    assert "." in tokens
