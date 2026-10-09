"""Evaluate BiLSTM POS tagging with and without probabilistic spellchecking."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.baseline import MostFrequentTagBaseline
from src.config import (
    EVAL_METRICS_PATH,
    FIGURES_DIR,
    MAX_SEQ_LEN,
    PAD_TAG,
    SEED,
    TRAIN_RATIO,
    UNK_TOKEN,
    VAL_RATIO,
    ensure_project_dirs,
)
from src.data_loader import load_dataset, split_dataset
from src.inference import load_trained_artifacts, predict_tags
from src.preprocessing import is_punctuation
from src.utils import save_json, set_seed


def _flatten_eval(
    gold_sentences: list[list[str]],
    pred_sentences: list[list[str]],
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for gold, pred in zip(gold_sentences, pred_sentences):
        limit = min(len(gold), len(pred))
        y_true.extend(gold[:limit])
        y_pred.extend(pred[:limit])
    return y_true, y_pred


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "support": int(len(y_true)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }


def introduce_typos(
    sentences: list[list[str]],
    vocab: set[str],
    rate: float = 0.15,
    seed: int = SEED,
) -> list[list[str]]:
    """Create a noisy copy of test sentences so spellchecking can be measured."""
    rng = np.random.default_rng(seed)
    noisy: list[list[str]] = []
    for sentence in sentences:
        new_sentence = []
        for word in sentence:
            if is_punctuation(word) or word not in vocab or len(word) < 3:
                new_sentence.append(word)
                continue
            if rng.random() > rate:
                new_sentence.append(word)
                continue
            chars = list(word)
            index = int(rng.integers(0, len(chars)))
            if index + 1 < len(chars) and rng.random() < 0.5:
                chars[index], chars[index + 1] = chars[index + 1], chars[index]
            else:
                alphabet = sorted({ch for w in vocab for ch in w})
                chars[index] = alphabet[int(rng.integers(0, len(alphabet)))]
            new_sentence.append("".join(chars))
        noisy.append(new_sentence)
    return noisy


def plot_confusion(y_true: list[str], y_pred: list[str], labels: list[str], path: Path) -> None:
    if not y_true:
        return
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("POS tag confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_metric_comparison(results: dict, path: Path) -> None:
    names = list(results.keys())
    f1_values = [results[name]["f1"] for name in names]
    acc_values = [results[name]["accuracy"] for name in names]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, acc_values, width, label="Accuracy")
    plt.bar(x + width / 2, f1_values, width, label="Macro F1")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model comparison on the test split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def evaluate() -> dict:
    ensure_project_dirs()
    set_seed(SEED)
    try:
        model, word_vocab, tag_vocab, spellchecker, device = load_trained_artifacts()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    sentences = load_dataset()
    train, _, test = split_dataset(sentences, TRAIN_RATIO, VAL_RATIO, SEED)
    if not test:
        raise SystemExit("Test split is empty. Add more tagged sentences before evaluating.")

    gold_words = [s.words[:MAX_SEQ_LEN] for s in test]
    gold_tags = [s.tags[:MAX_SEQ_LEN] for s in test]
    labels = [tag for tag in tag_vocab.id_to_token.values() if tag not in {PAD_TAG, UNK_TOKEN}]

    pred_plain = [predict_tags(model, word_vocab, tag_vocab, words, device) for words in gold_words]
    y_true, y_pred_plain = _flatten_eval(gold_tags, pred_plain)
    metrics_plain = compute_metrics(y_true, y_pred_plain, labels)

    vocab_words = set(spellchecker.word_counts)
    noisy_words = introduce_typos(gold_words, vocab_words)
    pred_noisy = [predict_tags(model, word_vocab, tag_vocab, words, device) for words in noisy_words]
    _, y_pred_noisy = _flatten_eval(gold_tags, pred_noisy)
    metrics_noisy = compute_metrics(y_true, y_pred_noisy, labels)

    corrected_words = [spellchecker.correct_sentence(words)["corrected_tokens"] for words in noisy_words]
    pred_corrected = [
        predict_tags(model, word_vocab, tag_vocab, words, device) for words in corrected_words
    ]
    _, y_pred_corrected = _flatten_eval(gold_tags, pred_corrected)
    metrics_corrected = compute_metrics(y_true, y_pred_corrected, labels)

    baseline = MostFrequentTagBaseline()
    baseline.fit([s.words for s in train], [s.tags for s in train])
    pred_baseline = baseline.predict(gold_words)
    _, y_pred_baseline = _flatten_eval(gold_tags, pred_baseline)
    metrics_baseline = compute_metrics(y_true, y_pred_baseline, labels)

    plot_confusion(y_true, y_pred_plain, labels, FIGURES_DIR / "confusion_matrix.png")
    comparison = {
        "MostFrequentTag": metrics_baseline,
        "BiLSTM": metrics_plain,
        "BiLSTM_on_noisy": metrics_noisy,
        "BiLSTM_plus_spellcheck": metrics_corrected,
    }
    slim_comparison = {
        name: {k: v for k, v in metrics.items() if k != "report"}
        for name, metrics in comparison.items()
    }
    plot_metric_comparison(slim_comparison, FIGURES_DIR / "model_comparison.png")

    payload = {
        "dataset_note": (
            "Metrics below are computed from the dataset currently in data/raw/. "
            "If you are using the bundled sample corpus, these numbers are demonstration-only "
            "and should not be reported as research results."
        ),
        "n_test_sentences": len(test),
        "n_test_tokens": metrics_plain["support"],
        "labels": labels,
        "metrics": comparison,
        "summary": slim_comparison,
    }
    save_json(EVAL_METRICS_PATH, payload)

    print("\n=== Evaluation (actual model predictions) ===")
    for name, metrics in slim_comparison.items():
        print(
            f"{name:24s}  Acc={metrics['accuracy']:.4f}  "
            f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}"
        )
    print(f"\nSaved metrics to {EVAL_METRICS_PATH}")
    print(f"Saved figures to {FIGURES_DIR}")
    return payload


if __name__ == "__main__":
    evaluate()
