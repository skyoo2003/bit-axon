import json

import mlx.core as mx

from bit_axon.model import BitAxonModel
from bit_axon.training.collate import iterate_batches
from bit_axon.training.data import AlpacaDataset, CacheDataset, ORPODataset, SFTDataset
from bit_axon.training.loss import cross_entropy_loss


def _write_jsonl(tmp_path, name, data):
    path = tmp_path / name
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


SFT_EXAMPLES = [
    {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
    {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
    {"messages": [{"role": "user", "content": "Name a color"}, {"role": "assistant", "content": "Blue"}]},
    {"messages": [{"role": "user", "content": "Say yes"}, {"role": "assistant", "content": "Yes"}]},
    {"messages": [{"role": "user", "content": "Count to 1"}, {"role": "assistant", "content": "1"}]},
]


class TestDataPipelineEndToEnd:
    def test_sft_pipeline_end_to_end(self, tmp_path, test_tokenizer):
        path = _write_jsonl(tmp_path, "sft.jsonl", SFT_EXAMPLES)
        ds = SFTDataset(path, test_tokenizer)
        ds = CacheDataset(ds)
        batches = list(iterate_batches(ds, batch_size=1, max_seq_len=32, shuffle=False))
        assert len(batches) >= 1

        input_ids, labels = batches[0]
        assert isinstance(input_ids, mx.array)
        assert isinstance(labels, mx.array)
        mx.eval(input_ids, labels)

        assert input_ids.shape[0] == 1
        assert input_ids.shape[1] == 31  # max_seq_len - 1
        assert input_ids.dtype == mx.int32

        assert labels.shape == input_ids.shape
        assert labels.dtype == mx.int32

        labels_list = labels.tolist()[0]
        has_masked = any(v == -100 for v in labels_list)
        has_valid = any(v > 0 for v in labels_list)
        assert has_masked, "Labels should contain -100 (masked prompt positions)"
        assert has_valid, "Labels should contain valid positive token IDs"

    def test_alpaca_pipeline_end_to_end(self, tmp_path, test_tokenizer):
        alpaca_examples = [
            {"instruction": "Translate to French", "input": "", "output": "Bonjour"},
            {"instruction": "What is 2+2?", "input": "", "output": "4"},
            {"instruction": "Say hello", "input": "", "output": "Hello!"},
        ]
        path = _write_jsonl(tmp_path, "alpaca.jsonl", alpaca_examples)
        ds = AlpacaDataset(path, test_tokenizer)
        batches = list(iterate_batches(ds, batch_size=1, max_seq_len=32, shuffle=False))
        assert len(batches) >= 1

        for input_ids, labels in batches:
            assert isinstance(input_ids, mx.array)
            assert isinstance(labels, mx.array)
            mx.eval(input_ids, labels)
            assert input_ids.shape[0] == 1
            assert input_ids.shape[1] == 31
            assert input_ids.dtype == mx.int32
            assert labels.dtype == mx.int32

    def test_loss_computation_on_pipeline_output(self, tmp_path, test_tokenizer):
        path = _write_jsonl(tmp_path, "sft_loss.jsonl", SFT_EXAMPLES[:2])
        ds = SFTDataset(path, test_tokenizer)
        batches = list(iterate_batches(ds, batch_size=1, max_seq_len=32, shuffle=False))
        input_ids, labels = batches[0]
        mx.eval(input_ids, labels)

        T = input_ids.shape[1]
        vocab_size = test_tokenizer.vocab_size
        logits = mx.random.normal(shape=(1, T, vocab_size))
        mx.eval(logits)

        loss, num_valid = cross_entropy_loss(logits, labels)
        mx.eval(loss, num_valid)

        assert isinstance(loss, mx.array)
        assert loss.shape == ()
        assert mx.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
        assert num_valid.item() > 0, "Expected at least one valid token"

    def test_pipeline_with_model_forward_pass(self, tmp_path, test_tokenizer, small_config):
        path = _write_jsonl(tmp_path, "sft_model.jsonl", SFT_EXAMPLES[:2])
        ds = SFTDataset(path, test_tokenizer)
        batches = list(iterate_batches(ds, batch_size=1, max_seq_len=32, shuffle=False))
        input_ids, _ = batches[0]
        mx.eval(input_ids)

        model = BitAxonModel(small_config)
        logits, _caches = model(input_ids)
        mx.eval(logits)

        assert logits.shape[0] == 1
        assert logits.shape[1] == input_ids.shape[1]
        assert logits.shape[2] == small_config.vocab_size
        assert mx.all(mx.isfinite(logits)).item(), "Logits contain NaN or Inf"


class TestDataPipelineORPO:
    def test_orpo_dataset_integration(self, tmp_path, test_tokenizer):
        orpo_examples = [
            {
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "chosen": [{"role": "assistant", "content": "4"}],
                "rejected": [{"role": "assistant", "content": "5"}],
            },
            {
                "prompt": [{"role": "user", "content": "Say hi"}],
                "chosen": [{"role": "assistant", "content": "Hello!"}],
                "rejected": [{"role": "assistant", "content": "Bye!"}],
            },
        ]
        path = _write_jsonl(tmp_path, "orpo.jsonl", orpo_examples)
        ds = ORPODataset(path, test_tokenizer)
        assert len(ds) == 2

        for i in range(len(ds)):
            result = ds[i]
            assert len(result) == 4, f"Expected 4-tuple, got {len(result)}"
            chosen_ids, chosen_mask, rejected_ids, rejected_mask = result

            assert isinstance(chosen_ids, list)
            assert isinstance(chosen_mask, list)
            assert isinstance(rejected_ids, list)
            assert isinstance(rejected_mask, list)
            assert len(chosen_ids) == len(chosen_mask)
            assert len(rejected_ids) == len(rejected_mask)

            assert chosen_ids != rejected_ids, "Chosen and rejected should differ"

            assert all(m in (0, 1) for m in chosen_mask), "Chosen mask must be binary"
            assert all(m in (0, 1) for m in rejected_mask), "Rejected mask must be binary"


class TestDataPipelineEdgeCases:
    def test_single_example_dataset(self, tmp_path, test_tokenizer):
        path = _write_jsonl(tmp_path, "single.jsonl", SFT_EXAMPLES[:1])
        ds = SFTDataset(path, test_tokenizer)
        batches = list(iterate_batches(ds, batch_size=1, max_seq_len=32, shuffle=False))
        assert len(batches) == 1

    def test_very_long_example_truncation(self, tmp_path, test_tokenizer):
        long_content = "This is a very long sentence. " * 200
        data = [
            {"messages": [{"role": "user", "content": long_content}, {"role": "assistant", "content": "Done"}]},
        ]
        path = _write_jsonl(tmp_path, "long.jsonl", data)
        ds = SFTDataset(path, test_tokenizer)
        batches = list(iterate_batches(ds, batch_size=1, max_seq_len=32, shuffle=False))
        assert len(batches) >= 1

        input_ids, labels = batches[0]
        mx.eval(input_ids, labels)
        assert input_ids.shape[1] == 31
