"""Tests for the probabilistic Telugu spellchecker."""

from src.spellchecker import ProbabilisticSpellChecker, levenshtein


def test_levenshtein_distance():
    assert levenshtein("పాఠశాలకు", "పాఠశాలకు") == 0
    assert levenshtein("పాఠశాలకి", "పాఠశాలకు") == 1


def test_corrects_close_telugu_typo():
    checker = ProbabilisticSpellChecker()
    checker.fit([["నేను", "పాఠశాలకు", "వెళ్తున్నాను"]] * 8)
    result = checker.correct_word("పాఠశాలకి")
    assert result.corrected == "పాఠశాలకు"
    assert result.was_corrected


def test_keeps_known_word():
    checker = ProbabilisticSpellChecker()
    checker.fit([["నేను", "పుస్తకం"]])
    result = checker.correct_word("నేను")
    assert result.corrected == "నేను"
    assert not result.was_corrected


def test_unknown_word_without_candidate():
    checker = ProbabilisticSpellChecker(threshold=0.0)
    checker.fit([["నేను"]])
    result = checker.correct_word("zzzzzz")
    assert result.corrected == "zzzzzz"


def test_correct_sentence_and_empty_input():
    checker = ProbabilisticSpellChecker()
    checker.fit([["నేను", "పాఠశాలకు", "వెళ్తున్నాను"]] * 5)
    output = checker.correct_sentence("నేను పాఠశాలకి వెళ్తున్నాను")
    assert "పాఠశాలకు" in output["corrected_tokens"]
    empty = checker.correct_sentence("")
    assert empty["tokens"] == []
    punct = checker.correct_word(".")
    assert punct.corrected == "."
