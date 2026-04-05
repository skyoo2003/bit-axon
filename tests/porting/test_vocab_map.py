"""Tests for tokenizer vocabulary mapping and truncation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from tokenizers import Tokenizer

from bit_axon.porting.vocab_map import build_vocab_mapping, load_truncated_tokenizer


def _make_mock_tokenizer(vocab_size: int = 100) -> Tokenizer:
    """Create a minimal WordLevel tokenizer with the given vocab size."""
    from tokenizers import models, pre_tokenizers

    vocab = {f"tok_{i}": i for i in range(vocab_size)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok_0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


@pytest.fixture()
def mock_tokenizer_100() -> Tokenizer:
    return _make_mock_tokenizer(100)


@pytest.fixture()
def patched_build(mock_tokenizer_100: Tokenizer):
    """Patch build_vocab_mapping to use the mock tokenizer."""
    with patch("bit_axon.porting.vocab_map.Tokenizer.from_pretrained", return_value=mock_tokenizer_100):
        yield


def _patched_build_and_load(mock_tokenizer: Tokenizer):
    """Return a context manager that patches both build and load to use the mock."""
    return patch("bit_axon.porting.vocab_map.Tokenizer.from_pretrained", return_value=mock_tokenizer)


class TestBuildVocabMapping:
    def test_vocab_mapping_size(self, patched_build):
        mapping = build_vocab_mapping(tokenizer_name="mock", target_size=50)
        assert len(mapping) == 50

    def test_vocab_mapping_keys_unique(self, patched_build):
        mapping = build_vocab_mapping(tokenizer_name="mock", target_size=50)
        assert len(set(mapping.keys())) == len(mapping)

    def test_vocab_mapping_values_range(self, patched_build):
        target = 50
        mapping = build_vocab_mapping(tokenizer_name="mock", target_size=target)
        assert all(0 <= v < target for v in mapping.values())

    def test_vocab_mapping_with_mock_tokenizer(self, patched_build):
        mapping = build_vocab_mapping(tokenizer_name="mock", target_size=50)
        assert all(isinstance(k, int) for k in mapping)
        assert all(isinstance(v, int) for v in mapping.values())

    def test_vocab_mapping_values_are_dense(self, patched_build):
        target = 50
        mapping = build_vocab_mapping(tokenizer_name="mock", target_size=target)
        assert set(mapping.values()) == set(range(target))

    def test_build_vocab_mapping_default_fallback(self, patched_build):
        target = 50
        mapping = build_vocab_mapping(tokenizer_name="mock", target_size=target, corpus_text=None)
        # First-N strategy: old_ids are 0..49, new_ids are also 0..49
        for i in range(target):
            assert mapping[i] == i

    def test_vocab_mapping_preserves_most_frequent(self):
        mock_tok = _make_mock_tokenizer(100)
        # Build corpus that repeats tok_42 heavily
        corpus = " ".join(["tok_42"] * 200 + ["tok_7"] * 100 + ["tok_99"] * 50)
        with _patched_build_and_load(mock_tok):
            mapping = build_vocab_mapping(tokenizer_name="mock", target_size=10, corpus_text=corpus)
        # tok_42 (id=42) should have the lowest new_id since it's most frequent
        most_frequent_new_id = mapping[42]
        assert most_frequent_new_id == 0

    def test_vocab_mapping_target_exceeds_vocab(self, patched_build):
        with pytest.raises(ValueError, match="exceeds tokenizer vocab size"):
            build_vocab_mapping(tokenizer_name="mock", target_size=200)


class TestLoadTruncatedTokenizer:
    def test_load_truncated_tokenizer_vocab_size(self, mock_tokenizer_100):
        vocab_mapping = {i: i for i in range(50)}
        with _patched_build_and_load(mock_tokenizer_100):
            truncated = load_truncated_tokenizer("mock", vocab_mapping)
        assert truncated.get_vocab_size() == 50

    def test_load_truncated_tokenizer_has_mapped_tokens(self, mock_tokenizer_100):
        vocab_mapping = {0: 0, 1: 1, 5: 2}
        with _patched_build_and_load(mock_tokenizer_100):
            truncated = load_truncated_tokenizer("mock", vocab_mapping)
        vocab = truncated.get_vocab()
        assert "tok_0" in vocab
        assert "tok_1" in vocab
        assert "tok_5" in vocab
        assert vocab["tok_0"] == 0
        assert vocab["tok_1"] == 1
        assert vocab["tok_5"] == 2
