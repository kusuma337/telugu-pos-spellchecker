"""Train the BiLSTM POS tagger and persist artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from src.config import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EMBED_DIM,
    EPOCHS,
    FIGURES_DIR,
    GRAD_CLIP_NORM,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    METRICS_DIR,
    MIN_WORD_FREQ,
    MODEL_PATH,
    NUM_LAYERS,
    PROCESSED_DATA_DIR,
    SEED,
    SPELLCHECKER_PATH,
    TRAIN_HISTORY_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    VOCAB_PATH,
    WEIGHT_DECAY,
    ensure_project_dirs,
    get_device,
)
from src.data_loader import TaggedSentence, load_dataset, split_dataset
from src.pos_model import BiLSTMPOSTagger, build_model
from src.spellchecker import ProbabilisticSpellChecker
from src.utils import save_json, set_seed
from src.vocabulary import TagVocabulary, Vocabulary, pad_sequences, save_vocabularies


class POSDataset(Dataset):
    def __init__(self, word_ids: list[list[int]], tag_ids: list[list[int]]) -> None:
        self.word_ids = word_ids
        self.tag_ids = tag_ids

    def __len__(self) -> int:
        return len(self.word_ids)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.word_ids[index], self.tag_ids[index]


def collate_batch(
    batch: list[tuple[list[int], list[int]]],
    word_pad_id: int,
    tag_pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    words, tags = zip(*batch)
    padded_words, lengths, _ = pad_sequences(words, pad_id=word_pad_id)
    padded_tags, _, _ = pad_sequences(tags, pad_id=tag_pad_id, max_len=len(padded_words[0]))
    return (
        torch.tensor(padded_words, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def encode_split(
    split: list[TaggedSentence],
    word_vocab: Vocabulary,
    tag_vocab: TagVocabulary,
    max_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    word_ids = []
    tag_ids = []
    for sentence in split:
        words = sentence.words[:max_len]
        tags = sentence.tags[:max_len]
        word_ids.append(word_vocab.encode_sequence(words, max_len=max_len))
        tag_ids.append(tag_vocab.encode_sequence(tags, max_len=max_len))
    return word_ids, tag_ids


def make_loader(
    word_ids: list[list[int]],
    tag_ids: list[list[int]],
    word_pad_id: int,
    tag_pad_id: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = POSDataset(word_ids, tag_ids)
    if len(dataset) == 0:
        raise ValueError("Cannot create a DataLoader from an empty split.")
    effective_batch = min(batch_size, len(dataset))
    return DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, word_pad_id, tag_pad_id),
    )


def run_epoch(
    model: BiLSTMPOSTagger,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    tag_pad_id: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batches = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    for word_ids, tag_ids, lengths in loader:
        word_ids = word_ids.to(device)
        tag_ids = tag_ids.to(device)
        lengths = lengths.to(device)
        with torch.set_grad_enabled(training):
            logits = model(word_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), tag_ids.view(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        pred = torch.argmax(logits, dim=-1)
        mask = tag_ids != tag_pad_id
        all_true.extend(tag_ids[mask].detach().cpu().tolist())
        all_pred.extend(pred[mask].detach().cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    if not all_true:
        return {"loss": avg_loss, "accuracy": 0.0, "f1": 0.0}
    accuracy = float(accuracy_score(all_true, all_pred))
    f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}


def plot_history(history: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plots = [
        ("loss.png", "Loss", [("train_loss", "Train"), ("val_loss", "Validation")]),
        ("accuracy.png", "Accuracy", [("val_accuracy", "Validation")]),
        ("f1.png", "Macro F1", [("val_f1", "Validation")]),
    ]
    for filename, ylabel, series in plots:
        plt.figure(figsize=(8, 5))
        for key, label in series:
            plt.plot(epochs, history[key], marker="o", label=label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"BiLSTM {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=140)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "train_val_loss.png", dpi=140)
    plt.close()


def save_checkpoint(
    path: Path,
    model: BiLSTMPOSTagger,
    word_vocab: Vocabulary,
    tag_vocab: TagVocabulary,
    history: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": len(word_vocab),
            "tagset_size": len(tag_vocab),
            "embed_dim": model.embed_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "pad_idx": model.pad_idx,
            "history": history,
        },
        path,
    )


def train_model(args: argparse.Namespace | None = None) -> dict:
    ensure_project_dirs()
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    sentences = load_dataset()
    print(f"Loaded {len(sentences)} tagged sentences.")
    train, val, test = split_dataset(sentences, TRAIN_RATIO, VAL_RATIO, SEED)
    if not val:
        val = train[: max(1, len(train) // 5)]
        print("Validation split was empty; using a slice of training data for validation.")
    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    word_tokens = [w for s in train for w in s.words]
    tag_tokens = [t for s in train for t in s.tags]
    word_vocab = Vocabulary(word_tokens, min_freq=MIN_WORD_FREQ)
    tag_vocab = TagVocabulary(tag_tokens)
    save_vocabularies(word_vocab, tag_vocab, VOCAB_PATH)

    spellchecker = ProbabilisticSpellChecker()
    spellchecker.fit([s.words for s in train])
    spellchecker.save(SPELLCHECKER_PATH)

    train_x, train_y = encode_split(train, word_vocab, tag_vocab, MAX_SEQ_LEN)
    val_x, val_y = encode_split(val, word_vocab, tag_vocab, MAX_SEQ_LEN)
    test_ids = [{"words": s.words, "tags": s.tags} for s in test]
    save_json(PROCESSED_DATA_DIR / "test_split.json", test_ids)
    save_json(
        PROCESSED_DATA_DIR / "split_sizes.json",
        {"train": len(train), "val": len(val), "test": len(test), "total": len(sentences)},
    )

    train_loader = make_loader(train_x, train_y, word_vocab.pad_id, tag_vocab.pad_id, BATCH_SIZE, True)
    val_loader = make_loader(val_x, val_y, word_vocab.pad_id, tag_vocab.pad_id, BATCH_SIZE, False)

    model = build_model(
        vocab_size=len(word_vocab),
        tagset_size=len(tag_vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=word_vocab.pad_id,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tag_vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }
    best_f1 = -1.0
    patience_left = EARLY_STOPPING_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, tag_vocab.pad_id)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, tag_vocab.pad_id)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_left = EARLY_STOPPING_PATIENCE
            save_checkpoint(MODEL_PATH, model, word_vocab, tag_vocab, history)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    plot_history(history, FIGURES_DIR)
    save_json(TRAIN_HISTORY_PATH, history)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vocabulary to {VOCAB_PATH}")
    print(f"Saved spellchecker to {SPELLCHECKER_PATH}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Telugu BiLSTM POS tagger.")
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
