"""Word and POS-tag vocabularies with integer encoding."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from src.config import PAD_TAG, PAD_TOKEN, UNK_TOKEN
from src.preprocessing import safe_tokens


class Vocabulary:
    """Bidirectional mapping between tokens and integer IDs."""

    def __init__(self, tokens: Iterable[str] | None = None, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.token_to_id: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_token: dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}
        if tokens is not None:
            self.build(tokens, min_freq=min_freq)

    def build(self, tokens: Iterable[str], min_freq: int | None = None) -> None:
        min_freq = self.min_freq if min_freq is None else min_freq
        counts = Counter(safe_tokens(tokens))
        for token, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if freq < min_freq:
                continue
            if token in self.token_to_id or token in {PAD_TOKEN, UNK_TOKEN}:
                continue
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

    def encode(self, token: str) -> int:
        token = token if token is not None else ""
        return self.token_to_id.get(token, self.unk_id)

    def encode_sequence(self, tokens: Sequence[str], max_len: int | None = None) -> list[int]:
        encoded = [self.encode(token) for token in tokens]
        if max_len is not None:
            encoded = encoded[:max_len]
        return encoded

    def decode(self, index: int) -> str:
        return self.id_to_token.get(int(index), UNK_TOKEN)

    def decode_sequence(self, indices: Sequence[int], ignore_pad: bool = True) -> list[str]:
        tokens: list[str] = []
        for index in indices:
            token = self.decode(index)
            if ignore_pad and token == PAD_TOKEN:
                continue
            tokens.append(token)
        return tokens

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict:
        return {
            "min_freq": self.min_freq,
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocabulary":
        vocab = cls(min_freq=int(payload.get("min_freq", 1)))
        token_to_id = payload.get("token_to_id", {})
        vocab.token_to_id = {str(k): int(v) for k, v in token_to_id.items()}
        if PAD_TOKEN not in vocab.token_to_id:
            vocab.token_to_id[PAD_TOKEN] = 0
        if UNK_TOKEN not in vocab.token_to_id:
            vocab.token_to_id[UNK_TOKEN] = 1
        vocab.id_to_token = {int(v): str(k) for k, v in vocab.token_to_id.items()}
        return vocab


class TagVocabulary(Vocabulary):
    """POS tag vocabulary. Unknown tags map to UNK rather than crashing."""

    def __init__(self, tags: Iterable[str] | None = None, min_freq: int = 1) -> None:
        super().__init__(tokens=None, min_freq=min_freq)
        self.token_to_id = {PAD_TAG: 0, UNK_TOKEN: 1}
        self.id_to_token = {0: PAD_TAG, 1: UNK_TOKEN}
        if tags is not None:
            self.build(tags, min_freq=1)


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    max_len: int | None = None,
) -> tuple[list[list[int]], list[int], list[list[int]]]:
    """Pad integer sequences and return a binary mask (1 = real token)."""
    lengths = [len(seq) for seq in sequences]
    if not lengths:
        return [], [], []
    target_len = max_len if max_len is not None else max(lengths)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    clipped_lengths: list[int] = []
    for seq in sequences:
        clipped = list(seq)[:target_len]
        length = len(clipped)
        pad_count = target_len - length
        padded.append(clipped + [pad_id] * pad_count)
        masks.append([1] * length + [0] * pad_count)
        clipped_lengths.append(length)
    return padded, clipped_lengths, masks


def save_vocabularies(word_vocab: Vocabulary, tag_vocab: Vocabulary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "word_vocab": word_vocab.to_dict(),
        "tag_vocab": tag_vocab.to_dict(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocabularies(path: Path) -> tuple[Vocabulary, TagVocabulary]:
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {path}. Train the model first with `python -m src.train`."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    word_vocab = Vocabulary.from_dict(payload["word_vocab"])
    tag_vocab = TagVocabulary.from_dict(payload["tag_vocab"])
    return word_vocab, tag_vocab
