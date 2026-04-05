import json

import mlx.core as mx
import numpy as np
import pytest

from bit_axon.training.collate import BatchCollator, iterate_batches
from bit_axon.training.data import SFTDataset
from bit_axon.training.packing import PackedBatch


class TestBatchCollator:
    def test_single_batch_b1(self):
        seq_len = 64
        token_ids = list(range(1, seq_len + 1))
        loss_mask = [1] * seq_len
        pb = PackedBatch(token_ids=token_ids, loss_mask=loss_mask)

        collator = BatchCollator(batch_size=1, max_seq_len=seq_len)
        input_ids, labels = collator.collate([pb])

        assert input_ids.shape == (1, seq_len - 1)
        assert labels.shape == (1, seq_len - 1)

    def test_multi_batch_b2(self):
        seq_len = 64
        pb1 = PackedBatch(token_ids=list(range(1, seq_len + 1)), loss_mask=[1] * seq_len)
        pb2 = PackedBatch(token_ids=list(range(100, 100 + seq_len)), loss_mask=[1] * seq_len)

        collator = BatchCollator(batch_size=2, max_seq_len=seq_len)
        input_ids, labels = collator.collate([pb1, pb2])

        assert input_ids.shape == (2, seq_len - 1)
        assert labels.shape == (2, seq_len - 1)

    def test_labels_ignore_index(self):
        seq_len = 16
        token_ids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        loss_mask = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        pb = PackedBatch(token_ids=token_ids, loss_mask=loss_mask)

        collator = BatchCollator(batch_size=1, max_seq_len=seq_len)
        _, labels = collator.collate([pb])

        labels_np = np.array(labels)
        shifted_mask = loss_mask[1:]
        for t in range(seq_len - 1):
            if shifted_mask[t] == 0:
                assert labels_np[0, t] == -100

    def test_labels_valid_positions(self):
        seq_len = 16
        token_ids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        loss_mask = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
        pb = PackedBatch(token_ids=token_ids, loss_mask=loss_mask)

        collator = BatchCollator(batch_size=1, max_seq_len=seq_len)
        _, labels = collator.collate([pb])

        labels_np = np.array(labels)
        shifted_mask = loss_mask[1:]
        for t in range(seq_len - 1):
            if shifted_mask[t] == 1:
                assert labels_np[0, t] == token_ids[t + 1]

    def test_shift_by_one(self):
        seq_len = 32
        token_ids = list(range(5, 5 + seq_len))
        loss_mask = [1] * seq_len
        pb = PackedBatch(token_ids=token_ids, loss_mask=loss_mask)

        collator = BatchCollator(batch_size=1, max_seq_len=seq_len)
        input_ids, labels = collator.collate([pb])

        input_np = np.array(input_ids)
        labels_np = np.array(labels)

        for t in range(seq_len - 1):
            assert input_np[0, t] == token_ids[t], f"input_ids[0, {t}] should be token_ids[{t}]"
            assert labels_np[0, t] == token_ids[t + 1], f"labels[0, {t}] should be token_ids[{t + 1}]"

    def test_output_shapes(self):
        seq_len = 128
        batch_size = 3
        batches = [PackedBatch(token_ids=list(range(i, i + seq_len)), loss_mask=[1] * seq_len) for i in range(batch_size)]

        collator = BatchCollator(batch_size=batch_size, max_seq_len=seq_len)
        input_ids, labels = collator.collate(batches)

        assert input_ids.shape == (batch_size, seq_len - 1)
        assert labels.shape == (batch_size, seq_len - 1)

    def test_output_dtypes(self):
        seq_len = 64
        pb = PackedBatch(token_ids=list(range(seq_len)), loss_mask=[1] * seq_len)

        collator = BatchCollator(batch_size=1, max_seq_len=seq_len)
        input_ids, labels = collator.collate([pb])

        assert input_ids.dtype == mx.int32
        assert labels.dtype == mx.int32


@pytest.fixture
def small_sft_jsonl(tmp_path):
    path = tmp_path / "sft.jsonl"
    data = [
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
        {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]},
        {"messages": [{"role": "user", "content": "What?"}, {"role": "assistant", "content": "Yes!"}]},
    ]
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


class TestIterateBatches:
    def test_single_epoch(self, test_tokenizer, small_sft_jsonl):
        dataset = SFTDataset(data=str(small_sft_jsonl), tokenizer=test_tokenizer, max_seq_len=128)
        batches = list(iterate_batches(dataset, batch_size=1, max_seq_len=128, shuffle=False))
        assert len(batches) >= 1
        for input_ids, labels in batches:
            assert input_ids.shape[0] == 1
            assert input_ids.shape[1] == 127
            assert labels.shape == input_ids.shape

    def test_loop_infinite(self, test_tokenizer, small_sft_jsonl):
        dataset = SFTDataset(data=str(small_sft_jsonl), tokenizer=test_tokenizer, max_seq_len=128)
        gen = iterate_batches(dataset, batch_size=1, max_seq_len=128, shuffle=False, loop=True)
        results = [next(gen) for _ in range(3)]
        assert len(results) == 3
        for input_ids, labels in results:
            assert input_ids.shape[0] == 1
            assert input_ids.shape[1] == 127

    def test_shuffle_changes_order(self, test_tokenizer, small_sft_jsonl):
        dataset = SFTDataset(data=str(small_sft_jsonl), tokenizer=test_tokenizer, max_seq_len=128)
        gen_a = iterate_batches(dataset, batch_size=1, max_seq_len=128, shuffle=True, seed=42)
        gen_b = iterate_batches(dataset, batch_size=1, max_seq_len=128, shuffle=True, seed=99)
        a = np.array(next(gen_a)[0])
        b = np.array(next(gen_b)[0])
        assert not np.array_equal(a, b), "Different seeds should produce different order"

    def test_no_shuffle_preserves_order(self, test_tokenizer, small_sft_jsonl):
        dataset = SFTDataset(data=str(small_sft_jsonl), tokenizer=test_tokenizer, max_seq_len=128)
        gen_a = iterate_batches(dataset, batch_size=1, max_seq_len=128, shuffle=False, seed=42)
        gen_b = iterate_batches(dataset, batch_size=1, max_seq_len=128, shuffle=False, seed=99)
        a = np.array(next(gen_a)[0])
        b = np.array(next(gen_b)[0])
        assert np.array_equal(a, b), "No shuffle should be deterministic regardless of seed"

    def test_batch_size_1(self, test_tokenizer, small_sft_jsonl):
        dataset = SFTDataset(data=str(small_sft_jsonl), tokenizer=test_tokenizer, max_seq_len=128)
        for input_ids, labels in iterate_batches(dataset, batch_size=1, max_seq_len=128, shuffle=False):
            assert input_ids.shape[0] == 1

    def test_small_max_seq_len(self, test_tokenizer, small_sft_jsonl):
        dataset = SFTDataset(data=str(small_sft_jsonl), tokenizer=test_tokenizer, max_seq_len=32)
        batches = list(iterate_batches(dataset, batch_size=1, max_seq_len=32, shuffle=False))
        assert len(batches) >= 1
        for input_ids, labels in batches:
            assert input_ids.shape == (1, 31)
            assert labels.shape == (1, 31)

    def test_empty_dataset(self, test_tokenizer, tmp_path):
        path = tmp_path / "empty.jsonl"
        with open(path, "w") as f:
            pass
        dataset = SFTDataset(data=str(path), tokenizer=test_tokenizer, max_seq_len=128)
        batches = list(iterate_batches(dataset, batch_size=1, max_seq_len=128))
        assert len(batches) == 0
