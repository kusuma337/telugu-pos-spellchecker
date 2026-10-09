"""BiLSTM POS tagger implemented in PyTorch."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMPOSTagger(nn.Module):
    """
    word IDs -> Embedding -> BiLSTM -> Dropout -> Linear -> tag logits
    """

    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2 (PAD and UNK).")
        if tagset_size < 2:
            raise ValueError("tagset_size must be at least 2 (PAD and UNK).")

        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, word_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_ids: [batch, seq_len]
            lengths: [batch] true lengths without padding
        Returns:
            logits: [batch, seq_len, tagset_size]
        """
        if word_ids.ndim != 2:
            raise ValueError("word_ids must have shape [batch, seq_len].")
        if word_ids.size(0) == 0:
            raise ValueError("Empty batch is not allowed.")

        lengths = lengths.cpu().clamp(min=1)
        embedded = self.embedding(word_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=word_ids.size(1),
        )
        lstm_out = self.dropout(lstm_out)
        return self.classifier(lstm_out)

    def predict(self, word_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        logits = self.forward(word_ids, lengths)
        return torch.argmax(logits, dim=-1)


def build_model(
    vocab_size: int,
    tagset_size: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    pad_idx: int,
    device: Optional[torch.device] = None,
) -> BiLSTMPOSTagger:
    model = BiLSTMPOSTagger(
        vocab_size=vocab_size,
        tagset_size=tagset_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx,
    )
    if device is not None:
        model = model.to(device)
    return model
