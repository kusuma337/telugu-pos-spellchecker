"""Probabilistic Telugu spellchecker based on corpus frequencies and edit distance."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.config import (
    SPELLCHECK_EDIT1_LOG_PROB,
    SPELLCHECK_EDIT2_LOG_PROB,
    SPELLCHECK_KEEP_OOV_LOG_PROB,
    SPELLCHECK_MAX_EDIT_DISTANCE,
    SPELLCHECK_MIN_FREQ,
    SPELLCHECK_THRESHOLD,
)
from src.preprocessing import contains_telugu, is_punctuation, tokenize


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


@dataclass
class Correction:
    original: str
    corrected: str
    was_corrected: bool
    score: float
    candidates: list[tuple[str, float]]


class ProbabilisticSpellChecker:
    """
    Rank candidate corrections with:

        score(c | w) = log P(w | c) + log P(c)

    P(c) is the corpus unigram probability.
    P(w | c) is a simple error model based on edit distance.
    """

    def __init__(
        self,
        word_counts: dict[str, int] | None = None,
        min_freq: int = SPELLCHECK_MIN_FREQ,
        max_edit_distance: int = SPELLCHECK_MAX_EDIT_DISTANCE,
        threshold: float = SPELLCHECK_THRESHOLD,
    ) -> None:
        self.min_freq = min_freq
        self.max_edit_distance = max_edit_distance
        self.threshold = threshold
        self.word_counts: Counter[str] = Counter()
        self.total_count = 0
        self.alphabet: set[str] = set()
        if word_counts:
            self.fit_from_counts(word_counts)

    def fit(self, sentences: Iterable[Sequence[str]]) -> "ProbabilisticSpellChecker":
        counts: Counter[str] = Counter()
        for sentence in sentences:
            for token in sentence:
                if token and not is_punctuation(token):
                    counts[token] += 1
        return self.fit_from_counts(counts)

    def fit_from_counts(self, word_counts: dict[str, int] | Counter) -> "ProbabilisticSpellChecker":
        filtered = {
            word: int(count)
            for word, count in word_counts.items()
            if word and int(count) >= self.min_freq and not is_punctuation(word)
        }
        self.word_counts = Counter(filtered)
        self.total_count = sum(self.word_counts.values())
        self.alphabet = set("".join(self.word_counts.keys()))
        return self

    def log_prior(self, word: str) -> float:
        if self.total_count <= 0:
            return -math.inf
        count = self.word_counts.get(word, 0)
        if count <= 0:
            return -math.inf
        return math.log(count / self.total_count)

    def log_error(self, observed: str, candidate: str) -> float:
        distance = levenshtein(observed, candidate)
        if distance == 0:
            return 0.0
        if distance == 1:
            return SPELLCHECK_EDIT1_LOG_PROB
        if distance == 2:
            return SPELLCHECK_EDIT2_LOG_PROB
        return -math.inf

    def known(self, words: Iterable[str]) -> set[str]:
        return {w for w in words if w in self.word_counts}

    def edits1(self, word: str) -> set[str]:
        letters = self.alphabet or set(word)
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = {left + right[1:] for left, right in splits if right}
        transposes = {
            left + right[1] + right[0] + right[2:]
            for left, right in splits
            if len(right) > 1
        }
        replaces = {
            left + letter + right[1:]
            for left, right in splits
            if right
            for letter in letters
        }
        inserts = {
            left + letter + right
            for left, right in splits
            for letter in letters
        }
        return deletes | transposes | replaces | inserts

    def candidates_for(self, word: str) -> list[str]:
        if word in self.word_counts:
            return [word]
        edit1 = self.known(self.edits1(word))
        if edit1:
            return sorted(edit1)
        if self.max_edit_distance < 2:
            return []
        close: list[str] = []
        for vocab_word in self.word_counts:
            if abs(len(vocab_word) - len(word)) > self.max_edit_distance:
                continue
            if levenshtein(word, vocab_word) <= self.max_edit_distance:
                close.append(vocab_word)
        return close

    def score_candidate(self, observed: str, candidate: str) -> float:
        return self.log_error(observed, candidate) + self.log_prior(candidate)

    def correct_word(self, word: str) -> Correction:
        if not word or is_punctuation(word):
            return Correction(word, word, False, 0.0, [])

        ranked: list[tuple[str, float]] = []
        for candidate in self.candidates_for(word):
            ranked.append((candidate, self.score_candidate(word, candidate)))
        ranked.sort(key=lambda item: item[1], reverse=True)

        if word in self.word_counts:
            keep_score = self.score_candidate(word, word)
            return Correction(word, word, False, keep_score, ranked[:5])

        if not ranked:
            return Correction(word, word, False, SPELLCHECK_KEEP_OOV_LOG_PROB, [])

        best_word, best_score = ranked[0]
        if best_score < self.threshold:
            return Correction(word, word, False, best_score, ranked[:5])
        return Correction(word, best_word, best_word != word, best_score, ranked[:5])

    def correct_sentence(self, sentence: str | Sequence[str]) -> dict:
        tokens = list(sentence) if isinstance(sentence, (list, tuple)) else tokenize(sentence)
        original_text = sentence if isinstance(sentence, str) else " ".join(tokens)
        corrections = [self.correct_word(token) for token in tokens]
        corrected_tokens = [item.corrected for item in corrections]
        detected = [
            {
                "original": item.original,
                "corrected": item.corrected,
                "score": item.score,
                "candidates": item.candidates,
            }
            for item in corrections
            if item.was_corrected
        ]
        return {
            "original": original_text,
            "tokens": tokens,
            "corrected_tokens": corrected_tokens,
            "corrected": " ".join(corrected_tokens),
            "misspellings": detected,
        }

    def to_dict(self) -> dict:
        return {
            "min_freq": self.min_freq,
            "max_edit_distance": self.max_edit_distance,
            "threshold": self.threshold,
            "word_counts": dict(self.word_counts),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ProbabilisticSpellChecker":
        checker = cls(
            min_freq=int(payload.get("min_freq", SPELLCHECK_MIN_FREQ)),
            max_edit_distance=int(payload.get("max_edit_distance", SPELLCHECK_MAX_EDIT_DISTANCE)),
            threshold=float(payload.get("threshold", SPELLCHECK_THRESHOLD)),
        )
        checker.fit_from_counts(payload.get("word_counts", {}))
        return checker

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProbabilisticSpellChecker":
        if not path.exists():
            raise FileNotFoundError(
                f"Spellchecker file not found: {path}. Train first with `python -m src.train`."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def looks_like_telugu_word(word: str) -> bool:
    return contains_telugu(word) and not is_punctuation(word)
