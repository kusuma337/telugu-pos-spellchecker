"""Interactive and programmatic Telugu POS inference."""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.config import (
    DROPOUT,
    MODEL_PATH,
    SPELLCHECKER_PATH,
    VOCAB_PATH,
    get_device,
)
from src.pos_model import BiLSTMPOSTagger, build_model
from src.preprocessing import tokenize
from src.spellchecker import ProbabilisticSpellChecker
from src.vocabulary import load_vocabularies


class MissingArtifactError(FileNotFoundError):
    pass


def load_trained_artifacts():
    if not MODEL_PATH.exists():
        raise MissingArtifactError(
            f"Trained model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    if not VOCAB_PATH.exists():
        raise MissingArtifactError(
            f"Vocabulary not found at {VOCAB_PATH}. Run `python -m src.train` first."
        )
    if not SPELLCHECKER_PATH.exists():
        raise MissingArtifactError(
            f"Spellchecker not found at {SPELLCHECKER_PATH}. Run `python -m src.train` first."
        )

    device = get_device()
    word_vocab, tag_vocab = load_vocabularies(VOCAB_PATH)
    spellchecker = ProbabilisticSpellChecker.load(SPELLCHECKER_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = build_model(
        vocab_size=checkpoint.get("vocab_size", len(word_vocab)),
        tagset_size=checkpoint.get("tagset_size", len(tag_vocab)),
        embed_dim=checkpoint.get("embed_dim", 128),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_layers=checkpoint.get("num_layers", 1),
        dropout=DROPOUT,
        pad_idx=checkpoint.get("pad_idx", word_vocab.pad_id),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, word_vocab, tag_vocab, spellchecker, device


def predict_tags(
    model: BiLSTMPOSTagger,
    word_vocab,
    tag_vocab,
    tokens: list[str],
    device,
) -> list[str]:
    if not tokens:
        return []
    word_ids = torch.tensor([word_vocab.encode_sequence(tokens)], dtype=torch.long, device=device)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    with torch.no_grad():
        pred_ids = model.predict(word_ids, lengths)[0].cpu().tolist()
    return [tag_vocab.decode(idx) for idx in pred_ids[: len(tokens)]]


def predict_with_confidence(
    model: BiLSTMPOSTagger,
    word_vocab,
    tag_vocab,
    tokens: List[str],
    device,
) -> List[Tuple[str, float]]:
    """Return per-token (tag, confidence) pairs using softmax on raw logits.

    Args:
        model:      Trained BiLSTMPOSTagger in eval mode.
        word_vocab: Word vocabulary for encoding tokens.
        tag_vocab:  Tag vocabulary for decoding predicted indices.
        tokens:     List of preprocessed word strings.
        device:     Torch device.

    Returns:
        List of (tag_string, confidence_float) tuples, one per input token.
        confidence is the maximum softmax probability (0–1).
    """
    if not tokens:
        return []
    word_ids = torch.tensor(
        [word_vocab.encode_sequence(tokens)], dtype=torch.long, device=device
    )
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(word_ids, lengths)  # [1, seq_len, tagset_size]
    probs = F.softmax(logits[0], dim=-1)  # [seq_len, tagset_size]
    confidences, pred_ids = probs.max(dim=-1)  # each: [seq_len]
    pred_ids = pred_ids.cpu().tolist()
    confidences = confidences.cpu().tolist()
    result: List[Tuple[str, float]] = []
    for i in range(len(tokens)):
        tag = tag_vocab.decode(pred_ids[i])
        conf = float(confidences[i])
        result.append((tag, conf))
    return result


def analyze_sentence(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Please enter a non-empty Telugu sentence.")

    model, word_vocab, tag_vocab, spellchecker, device = load_trained_artifacts()
    original_tokens = tokenize(text)
    if not original_tokens:
        raise ValueError("No tokens were found after preprocessing. Check the input text.")

    spell_result = spellchecker.correct_sentence(original_tokens)
    corrected_tokens = spell_result["corrected_tokens"]
    tags_with_conf = predict_with_confidence(model, word_vocab, tag_vocab, corrected_tokens, device)
    tagged = [
        {"word": word, "tag": tag, "confidence": conf}
        for word, (tag, conf) in zip(corrected_tokens, tags_with_conf)
    ]
    return {
        "original": text,
        "original_tokens": original_tokens,
        "corrected": " ".join(corrected_tokens),
        "corrected_tokens": corrected_tokens,
        "misspellings": spell_result["misspellings"],
        "tagged": tagged,
        "device": str(device),
    }


def format_analysis(result: dict) -> str:
    lines = [
        f"Original:\n{result['original']}",
        "",
        f"Corrected:\n{result['corrected']}",
        "",
    ]
    if result["misspellings"]:
        lines.append("Detected misspellings:")
        for item in result["misspellings"]:
            lines.append(f"  {item['original']} -> {item['corrected']}")
        lines.append("")
    else:
        lines.append("Detected misspellings: none")
        lines.append("")
    lines.append("POS Tags:")
    for item in result["tagged"]:
        conf_str = f" ({item.get('confidence', 0.0)*100:.1f}%)" if "confidence" in item else ""
        lines.append(f"{item['word']} → {item['tag']}{conf_str}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Telugu POS tagging with spellchecking.")
    parser.add_argument("--text", type=str, default="", help="Telugu sentence to analyze")
    args = parser.parse_args()
    try:
        sentence = args.text.strip() if args.text else ""
        if not sentence:
            print("Enter a Telugu sentence (empty line to exit):")
            sentence = sys.stdin.readline()
        sentence = (sentence or "").strip()
        if not sentence:
            raise ValueError("Empty input. Provide --text or type a sentence.")
        result = analyze_sentence(sentence)
        print(format_analysis(result))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
