"""Load Telugu POS datasets from several common text formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import RAW_DATA_DIR, SAMPLE_DATA_PATH
from src.preprocessing import clean_text, tokenize
from src.sample_corpus import write_sample_corpus


SUPPORTED_SUFFIXES = {".txt", ".tsv", ".conll", ".conllu", ".csv"}


@dataclass
class TaggedSentence:
    words: list[str]
    tags: list[str]
    source: str = ""


def _is_comment(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("#") or stripped.startswith("//")


def _split_token_tag(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or _is_comment(stripped):
        return None
    if "\t" in stripped:
        parts = [p for p in stripped.split("\t") if p != ""]
    else:
        parts = stripped.split()
    if len(parts) < 2:
        return None
    if parts[0].isdigit() and len(parts) >= 3:
        word, tag = parts[1], parts[-1]
    else:
        word, tag = parts[0], parts[-1]
    word = clean_text(word)
    tag = clean_text(tag).upper()
    if not word or not tag:
        return None
    return word, tag


def parse_conll_like(text: str, source: str = "") -> list[TaggedSentence]:
    sentences: list[TaggedSentence] = []
    words: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        nonlocal words, tags
        if words and tags and len(words) == len(tags):
            sentences.append(TaggedSentence(words=words, tags=tags, source=source))
        words, tags = [], []

    for raw_line in text.splitlines():
        if not raw_line.strip():
            flush()
            continue
        parsed = _split_token_tag(raw_line)
        if parsed is None:
            continue
        word, tag = parsed
        words.append(word)
        tags.append(tag)
    flush()
    return sentences


def parse_sentence_csv(text: str, source: str = "") -> list[TaggedSentence]:
    """Parse lines of the form: sentence,tag1 tag2 tag3  OR  sentence<TAB>tags."""
    sentences: list[TaggedSentence] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _is_comment(line) or line.lower().startswith("sentence"):
            continue
        if "\t" in line:
            left, right = line.split("\t", 1)
        elif "," in line:
            left, right = line.split(",", 1)
        else:
            continue
        words = tokenize(left)
        tags = [clean_text(t).upper() for t in right.split() if clean_text(t)]
        if words and tags and len(words) == len(tags):
            sentences.append(TaggedSentence(words=words, tags=tags, source=source))
    return sentences


def parse_file(path: Path) -> list[TaggedSentence]:
    text = path.read_text(encoding="utf-8")
    sentences = parse_conll_like(text, source=str(path.name))
    if sentences:
        return sentences
    return parse_sentence_csv(text, source=str(path.name))


def discover_raw_files(raw_dir: Path | None = None) -> list[Path]:
    raw_dir = raw_dir or RAW_DATA_DIR
    if not raw_dir.exists():
        return []
    files = [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def load_dataset(
    raw_dir: Path | None = None,
    create_sample_if_missing: bool = True,
) -> list[TaggedSentence]:
    raw_dir = raw_dir or RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = discover_raw_files(raw_dir)
    if not files and create_sample_if_missing:
        write_sample_corpus(SAMPLE_DATA_PATH)
        files = discover_raw_files(raw_dir)

    if not files:
        raise FileNotFoundError(
            f"No POS dataset found in {raw_dir}. "
            "Place a Telugu POS file there (word<TAB>tag, blank line between sentences)."
        )

    sentences: list[TaggedSentence] = []
    for path in files:
        try:
            loaded = parse_file(path)
        except OSError as exc:
            raise RuntimeError(f"Could not read dataset file {path.name}: {exc}") from exc
        sentences.extend(loaded)

    sentences = [s for s in sentences if s.words and s.tags and len(s.words) == len(s.tags)]
    if not sentences:
        raise ValueError(
            f"Dataset files were found in {raw_dir}, but no valid tagged sentences could be parsed."
        )
    return sentences


def split_dataset(
    sentences: list[TaggedSentence],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[TaggedSentence], list[TaggedSentence], list[TaggedSentence]]:
    if not sentences:
        raise ValueError("Cannot split an empty dataset.")
    import random

    shuffled = list(sentences)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = int(n * val_ratio)
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    if not train:
        raise ValueError("Training split is empty. Add more tagged sentences.")
    return train, val, test
