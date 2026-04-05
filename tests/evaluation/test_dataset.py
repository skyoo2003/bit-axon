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
