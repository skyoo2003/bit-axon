"""Tests for ORPO batch collation."""

from __future__ import annotations

import mlx.core as mx

from bit_axon.training.orpo_collate import (
    IGNORE_INDEX,
    collate_orpo_batch,
    create_labels,
    iterate_orpo_batches,
    pad_to_length,
)


class MockORPODataset:
    """Minimal dataset yielding (chosen_ids, chosen_mask, rejected_ids, rejected_mask) tuples."""

    def __init__(self, n: int = 10, seq_len: int = 20) -> None:
        self.data = [
            (
                [i] * (seq_len // 4) + [i + 100] * (seq_len * 3 // 4),
                [0] * (seq_len // 4) + [1] * (seq_len * 3 // 4),
                [i + 50] * (seq_len // 4) + [i + 200] * (seq_len * 3 // 4),
                [0] * (seq_len // 4) + [1] * (seq_len * 3 // 4),
            )
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], list[int], list[int]]:
        return self.data[idx]


class TestPadToLength:
    def test_no_pad_needed(self) -> None:
        assert pad_to_length([1, 2, 3], length=3) == [1, 2, 3]

    def test_right_pad(self) -> None:
        assert pad_to_length([1, 2], length=5, pad_value=0) == [1, 2, 0, 0, 0]

    def test_truncate(self) -> None:
        assert pad_to_length([1, 2, 3, 4, 5], length=3) == [1, 2, 3]


class TestCreateLabels:
    def test_prompt_masked(self) -> None:
        ids = [10, 20, 30, 40]
        mask = [0, 0, 1, 1]
        labels = create_labels(ids, mask, pad_length=4)
        assert labels == [-100, -100, 30, 40]

    def test_response_kept(self) -> None:
        ids = [10, 20, 30, 40]
        mask = [1, 1, 1, 1]
        labels = create_labels(ids, mask, pad_length=4)
        assert labels == [10, 20, 30, 40]

    def test_padding_masked(self) -> None:
        ids = [10, 20]
        mask = [0, 1]
        labels = create_labels(ids, mask, pad_length=4)
        assert labels == [-100, 20, -100, -100]


class TestCollateOrpoBatch:
    def test_output_shapes(self) -> None:
        batch = [((MockORPODataset(n=1, seq_len=20))[0]) for _ in range(3)]
        chosen_ids, chosen_labels, rejected_ids, rejected_labels = collate_orpo_batch(batch, max_seq_len=64)
        assert chosen_ids.shape[0] == 3
        assert chosen_labels.shape[0] == 3
        assert rejected_ids.shape[0] == 3
        assert rejected_labels.shape[0] == 3
        assert chosen_ids.shape == chosen_labels.shape == rejected_ids.shape == rejected_labels.shape

    def test_labels_have_ignore(self) -> None:
        chosen_ids = [1, 2, 3, 4, 5]
        chosen_mask = [0, 0, 1, 1, 1]
        rejected_ids = [10, 20, 30, 40, 50]
        rejected_mask = [0, 1, 1, 1, 1]
        batch = [(chosen_ids, chosen_mask, rejected_ids, rejected_mask)]
        _, chosen_labels, _, rejected_labels = collate_orpo_batch(batch, max_seq_len=64)

        chosen_labels_list = chosen_labels.tolist()[0]
        assert chosen_labels_list[0] == IGNORE_INDEX
        assert chosen_labels_list[1] == IGNORE_INDEX
        assert chosen_labels_list[2] == 3

        rejected_labels_list = rejected_labels.tolist()[0]
        assert rejected_labels_list[0] == IGNORE_INDEX
        assert rejected_labels_list[1] == 20

    def test_truncation(self) -> None:
        long_ids = list(range(100))
        mask = [1] * 100
        batch = [(long_ids, mask, long_ids, mask)]
        chosen_ids, _, rejected_ids, _ = collate_orpo_batch(batch, max_seq_len=32)
        assert chosen_ids.shape[1] == 32
        assert rejected_ids.shape[1] == 32

    def test_single_item(self) -> None:
        batch = [([5, 6, 7], [0, 1, 1], [8, 9, 10], [0, 1, 1])]
        chosen_ids, chosen_labels, rejected_ids, rejected_labels = collate_orpo_batch(batch, max_seq_len=64)
        assert chosen_ids.shape == (1, 3)
        assert chosen_labels.tolist()[0] == [-100, 6, 7]
        assert rejected_ids.tolist()[0] == [8, 9, 10]
        assert rejected_labels.tolist()[0] == [-100, 9, 10]


class TestIterateOrpoBatches:
    def test_yields_tuples(self) -> None:
        ds = MockORPODataset(n=4, seq_len=10)
        gen = iterate_orpo_batches(ds, batch_size=2, shuffle=False, loop=False)
        for item in gen:
            assert isinstance(item, tuple)
            assert len(item) == 4
            for tensor in item:
                assert isinstance(tensor, mx.array)

    def test_shapes(self) -> None:
        ds = MockORPODataset(n=6, seq_len=20)
        gen = iterate_orpo_batches(ds, batch_size=2, max_seq_len=64, shuffle=False, loop=False)
        for chosen_ids, chosen_labels, rejected_ids, rejected_labels in gen:
            assert chosen_ids.shape[0] == 2
            assert chosen_ids.shape == chosen_labels.shape == rejected_ids.shape == rejected_labels.shape

    def test_loop_mode(self) -> None:
        ds = MockORPODataset(n=4, seq_len=10)
        gen = iterate_orpo_batches(ds, batch_size=2, shuffle=False, loop=True, seed=0)
        results = [next(gen) for _ in range(len(ds) + 2)]
        assert len(results) == len(ds) + 2

    def test_no_loop(self) -> None:
        ds = MockORPODataset(n=10, seq_len=10)
        batches = list(iterate_orpo_batches(ds, batch_size=2, shuffle=False, loop=False))
        assert len(batches) == len(ds) // 2

    def test_shuffle(self) -> None:
        ds = MockORPODataset(n=20, seq_len=10)
        gen_a = iterate_orpo_batches(ds, batch_size=5, shuffle=True, loop=False, seed=42)
        gen_b = iterate_orpo_batches(ds, batch_size=5, shuffle=True, loop=False, seed=99)
        a = next(gen_a)[0]
        b = next(gen_b)[0]
        assert not mx.array_equal(a, b), "Different seeds should produce different first batches"
