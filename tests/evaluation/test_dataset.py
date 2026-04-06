import mlx.core as mx

from bit_axon.evaluation.dataset import WikiTextDataset


def test_dataset_creation():
    ds = WikiTextDataset(seq_length=64, max_tokens=512)
    assert len(ds) > 0


def test_dataset_length():
    seq_length = 64
    max_tokens = 512
    ds = WikiTextDataset(seq_length=seq_length, max_tokens=max_tokens)
    expected = max_tokens // seq_length
    assert len(ds) == expected


def test_dataset_item_shape():
    seq_length = 64
    ds = WikiTextDataset(seq_length=seq_length, max_tokens=512)
    item = ds[0]
    assert isinstance(item, mx.array)
    assert item.shape == (seq_length,)


def test_dataset_no_empty_chunks():
    ds = WikiTextDataset(seq_length=64, max_tokens=512)
    for i in range(len(ds)):
        chunk = ds[i]
        assert chunk.size > 0


def test_with_tokenizer_uses_encode():
    class MockTokenizer:
        vocab_size = 1000

        def encode(self, text):
            return [hash(c) % self.vocab_size for c in text]

    ds = WikiTextDataset(split="test", seq_length=64, max_tokens=128, tokenizer=MockTokenizer())
    assert len(ds) >= 1
    chunk = ds[0]
    tokens = chunk.tolist()
    assert any(t > 255 for t in tokens)


def test_without_tokenizer_backward_compat():
    ds = WikiTextDataset(split="test", seq_length=64, max_tokens=128)
    chunk = ds[0]
    tokens = chunk.tolist()
    assert all(0 <= t <= 255 for t in tokens)


def test_with_tokenizer_chunk_shape():
    class MockTokenizer:
        vocab_size = 500

        def encode(self, text):
            return [i % self.vocab_size for i in range(len(text))]

    ds = WikiTextDataset(split="test", seq_length=32, max_tokens=128, tokenizer=MockTokenizer())
    if len(ds) > 0:
        assert ds[0].shape == (32,)


def test_with_tokenizer_ids_in_vocab_range():
    class MockTokenizer:
        vocab_size = 1000

        def encode(self, text):
            return [i % self.vocab_size for i in range(len(text))]

    ds = WikiTextDataset(split="test", seq_length=64, max_tokens=128, tokenizer=MockTokenizer())
    for i in range(min(len(ds), 3)):
        tokens = ds[i].tolist()
        assert all(0 <= t < 1000 for t in tokens)
