"""Tests for the BiLSTM model, vocabulary, padding, and empty-input handling."""

import torch

from src.pos_model import BiLSTMPOSTagger
from src.preprocessing import tokenize
from src.vocabulary import TagVocabulary, Vocabulary, pad_sequences


def test_vocabulary_unk_and_pad():
    vocab = Vocabulary(["నేను", "పుస్తకం", "నేను"], min_freq=1)
    assert vocab.encode("నేను") != vocab.unk_id
    assert vocab.encode("unknown-word") == vocab.unk_id
    assert vocab.decode(vocab.pad_id) == "<PAD>"


def test_padding_and_mask():
    padded, lengths, masks = pad_sequences([[1, 2], [3]], pad_id=0)
    assert padded == [[1, 2], [3, 0]]
    assert lengths == [2, 1]
    assert masks == [[1, 1], [1, 0]]


def test_empty_sequences_padding():
    padded, lengths, masks = pad_sequences([], pad_id=0)
    assert padded == []
    assert lengths == []
    assert masks == []


def test_tag_vocabulary():
    tags = TagVocabulary(["NOUN", "VERB", "NOUN"])
    assert tags.encode("NOUN") != tags.unk_id
    assert tags.encode("XYZ") == tags.unk_id


def test_model_forward_pass_shape():
    model = BiLSTMPOSTagger(
        vocab_size=12,
        tagset_size=6,
        embed_dim=16,
        hidden_dim=8,
        num_layers=1,
        dropout=0.1,
        pad_idx=0,
    )
    word_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([3, 2], dtype=torch.long)
    logits = model(word_ids, lengths)
    assert logits.shape == (2, 4, 6)
    preds = model.predict(word_ids, lengths)
    assert preds.shape == (2, 4)


def test_pad_does_not_change_unpadded_positions():
    model = BiLSTMPOSTagger(10, 5, 8, 8, pad_idx=0)
    model.eval()
    word_ids = torch.tensor([[1, 2, 0]], dtype=torch.long)
    lengths = torch.tensor([2], dtype=torch.long)
    with torch.no_grad():
        logits = model(word_ids, lengths)
    assert torch.isfinite(logits[:, :2, :]).all()


def test_empty_tokenize():
    assert tokenize("") == []
    assert tokenize(None) == []
