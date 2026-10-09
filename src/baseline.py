"""Most-frequent-tag baseline (simple, no extra CRF dependency)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from src.config import UNK_TOKEN


class MostFrequentTagBaseline:
    """Assign each word the POS tag seen most often in training data."""

    def __init__(self) -> None:
        self.word_tag_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.global_tag_counts: Counter[str] = Counter()

    def fit(self, sentences: Sequence[Sequence[str]], tags: Sequence[Sequence[str]]) -> None:
        for words, word_tags in zip(sentences, tags):
            for word, tag in zip(words, word_tags):
                self.word_tag_counts[word][tag] += 1
                self.global_tag_counts[tag] += 1

    def predict_word(self, word: str) -> str:
        if word in self.word_tag_counts:
            return self.word_tag_counts[word].most_common(1)[0][0]
        if self.global_tag_counts:
            return self.global_tag_counts.most_common(1)[0][0]
        return UNK_TOKEN

    def predict(self, sentences: Sequence[Sequence[str]]) -> list[list[str]]:
        return [[self.predict_word(word) for word in sentence] for sentence in sentences]
