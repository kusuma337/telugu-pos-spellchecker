"""Tests for dataset parsing."""

from src.data_loader import parse_conll_like, parse_sentence_csv


def test_parse_word_tab_tag():
    text = "నేను\tPRON\nపుస్తకం\tNOUN\n\nఆమె\tPRON\nఉంది\tVERB\n"
    sentences = parse_conll_like(text, source="sample")
    assert len(sentences) == 2
    assert sentences[0].words == ["నేను", "పుస్తకం"]
    assert sentences[0].tags == ["PRON", "NOUN"]


def test_parse_sentence_csv():
    text = "నేను పుస్తకం,PRON NOUN\n"
    sentences = parse_sentence_csv(text)
    assert len(sentences) == 1
    assert sentences[0].tags == ["PRON", "NOUN"]
