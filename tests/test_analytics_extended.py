
import json
from collections import Counter

import pytest

from app.analytics import (
    detect_script,
    get_pos_distribution,
    get_text_analytics,
    introduce_noise,
    load_eval_metrics,
)


# 1–7: Script detection

def test_detect_script_telugu():
    assert detect_script("నేను పాఠశాలకు వెళ్తున్నాను") == "telugu"


def test_detect_script_english():
    assert detect_script("Hello world") == "english"


def test_detect_script_mixed():
    assert detect_script("నేను Hyderabad వెళ్తున్నాను") == "mixed"


def test_detect_script_numeric():
    assert detect_script("12345") == "numeric"


def test_detect_script_empty():
    assert detect_script("") == "empty"


def test_detect_script_whitespace():
    assert detect_script("   ") == "empty"


def test_detect_script_telugu_with_numbers():
    assert detect_script("రాము 123") == "telugu"


# 8–13: POS distribution

def test_pos_distribution_basic():
    tagged = [
        {"word": "నేను", "tag": "PRON"},
        {"word": "రాము", "tag": "NOUN"},
        {"word": "వెళ్తాను", "tag": "VERB"},
    ]
    result = get_pos_distribution(tagged)
    assert result == Counter({"PRON": 1, "NOUN": 1, "VERB": 1})


def test_pos_distribution_repeated_tags():
    tagged = [
        {"word": "రాము", "tag": "NOUN"},
        {"word": "పుస్తకం", "tag": "NOUN"},
        {"word": "చదివాడు", "tag": "VERB"},
    ]
    result = get_pos_distribution(tagged)
    assert result["NOUN"] == 2
    assert result["VERB"] == 1


def test_pos_distribution_empty():
    assert get_pos_distribution([]) == Counter()


def test_pos_distribution_single_token():
    result = get_pos_distribution([{"word": "నేను", "tag": "PRON"}])
    assert result["PRON"] == 1


def test_pos_distribution_all_same_tag():
    tagged = [{"word": str(i), "tag": "NOUN"} for i in range(4)]
    assert get_pos_distribution(tagged)["NOUN"] == 4


def test_pos_distribution_returns_counter():
    assert isinstance(get_pos_distribution([]), Counter)


# 14–19: Text analytics

def test_text_analytics_returns_dictionary():
    result = get_text_analytics("నేను పాఠశాలకు వెళ్తున్నాను", {})
    assert isinstance(result, dict)


def test_text_analytics_has_word_count():
    result = get_text_analytics("one two three", {})
    assert any("word" in key.lower() for key in result)


def test_text_analytics_empty_input():
    result = get_text_analytics("", {})
    assert isinstance(result, dict)


def test_text_analytics_telugu_input():
    result = get_text_analytics("నేను చదువుతున్నాను", {})
    assert isinstance(result, dict)


def test_text_analytics_punctuation():
    result = get_text_analytics("Hello, world!", {})
    assert isinstance(result, dict)


def test_text_analytics_result_argument_accepted():
    result = get_text_analytics("రాము పుస్తకం చదివాడు", {"tags": []})
    assert isinstance(result, dict)


# 20–25: Noise introduction

def test_introduce_noise_returns_list():
    result = introduce_noise(["రాము", "పుస్తకం"], noise_level=0)
    assert isinstance(result, list)


def test_introduce_noise_zero_preserves_tokens():
    tokens = ["రాము", "పుస్తకం", "చదివాడు"]
    assert introduce_noise(tokens, noise_level=0) == tokens


def test_introduce_noise_empty_list():
    assert introduce_noise([], noise_level=0) == []


def test_introduce_noise_preserves_token_count():
    tokens = ["నేను", "ఇంటికి", "వెళ్తాను"]
    result = introduce_noise(tokens, noise_level=0.2)
    assert len(result) == len(tokens)


def test_introduce_noise_does_not_modify_original():
    tokens = ["రాము", "పుస్తకం"]
    original = tokens.copy()
    introduce_noise(tokens, noise_level=0.5)
    assert tokens == original


def test_introduce_noise_zero_level_is_deterministic():
    tokens = ["తెలుగు", "భాష"]
    assert introduce_noise(tokens, noise_level=0) == introduce_noise(tokens, noise_level=0)


# 26–29: Evaluation metrics loader

def test_load_eval_metrics_returns_dictionary():
    result = load_eval_metrics()
    assert isinstance(result, dict)


def test_load_eval_metrics_contains_accuracy_if_available():
    result = load_eval_metrics()

    if not result:
        pytest.skip("Evaluation metrics file is empty or unavailable")

    def contains_accuracy(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if "accuracy" in key.lower() or key.lower() == "acc":
                    return True
                if contains_accuracy(value):
                    return True
        elif isinstance(data, list):
            return any(contains_accuracy(item) for item in data)
        return False

    assert contains_accuracy(result), (
        "No accuracy field found in the evaluation metrics structure"
    )


def test_load_eval_metrics_values_are_json_compatible():
    result = load_eval_metrics()
    json.dumps(result)


def test_load_eval_metrics_is_repeatable():
    assert load_eval_metrics() == load_eval_metrics()
