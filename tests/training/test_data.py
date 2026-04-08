from __future__ import annotations

import json

import pytest

from bit_axon.training.data import AlpacaDataset, CacheDataset, ORPODataset, SFTDataset, stream_jsonl


class TestStreamJsonl:
    def test_streams_valid_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n')
        result = list(stream_jsonl(path))
        assert result == [{"a": 1}, {"b": 2}]

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n\n\n{"b": 2}\n')
        result = list(stream_jsonl(path))
        assert result == [{"a": 1}, {"b": 2}]

    def test_handles_invalid_json(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
        with pytest.raises(json.JSONDecodeError):
            list(stream_jsonl(path))


class TestSFTDataset:
    @pytest.fixture
    def sft_data(self):
        return [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
        ]

    @pytest.fixture
    def sft_data_with_system(self):
        return [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
            },
        ]

    @pytest.fixture
    def sft_jsonl(self, tmp_path):
        path = tmp_path / "sft.jsonl"
        data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
            },
        ]
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return path

    def test_messages_format(self, test_tokenizer, sft_data):
        ds = SFTDataset(sft_data, test_tokenizer)
        token_ids, loss_mask = ds[0]
        assert isinstance(token_ids, list)
        assert isinstance(loss_mask, list)
        assert len(token_ids) == len(loss_mask)
        assert all(isinstance(t, int) for t in token_ids)
        assert all(isinstance(m, int) for m in loss_mask)

    def test_messages_format_with_system(self, test_tokenizer, sft_data_with_system):
        ds = SFTDataset(sft_data_with_system, test_tokenizer)
        token_ids, loss_mask = ds[0]
        assert len(token_ids) == len(loss_mask)

    def test_mask_prompt_true(self, test_tokenizer, sft_data):
        ds = SFTDataset(sft_data, test_tokenizer, mask_prompt=True)
        token_ids, loss_mask = ds[0]
        assert loss_mask[0] == 0
        assert loss_mask[-1] == 1

    def test_mask_prompt_false(self, test_tokenizer, sft_data):
        ds = SFTDataset(sft_data, test_tokenizer, mask_prompt=False)
        token_ids, loss_mask = ds[0]
        assert all(m == 1 for m in loss_mask)

    def test_truncation(self, test_tokenizer):
        long_content = "x" * 10000
        data = [
            {"messages": [{"role": "user", "content": long_content}, {"role": "assistant", "content": "short"}]},
        ]
        ds = SFTDataset(data, test_tokenizer, max_seq_len=64)
        token_ids, loss_mask = ds[0]
        assert len(token_ids) <= 64
        assert len(loss_mask) <= 64

    def test_returns_token_ids_and_mask(self, test_tokenizer, sft_data):
        ds = SFTDataset(sft_data, test_tokenizer)
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_len(self, test_tokenizer, sft_data):
        ds = SFTDataset(sft_data, test_tokenizer)
        assert len(ds) == 1

    def test_iter(self, test_tokenizer, sft_data):
        ds = SFTDataset(sft_data, test_tokenizer)
        items = list(ds)
        assert len(items) == 1
        assert len(items[0][0]) == len(items[0][1])

    def test_loads_from_jsonl_path(self, test_tokenizer, sft_jsonl):
        ds = SFTDataset(sft_jsonl, test_tokenizer)
        assert len(ds) == 2

    def test_eos_appended(self, test_tokenizer, sft_data):
        ds = SFTDataset(sft_data, test_tokenizer)
        token_ids, _ = ds[0]
        assert token_ids[-1] == test_tokenizer.eos_token_id

    def test_multiturn_loss_mask(self, test_tokenizer):
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "Fine, thanks!"},
                ],
            },
        ]
        ds = SFTDataset(data, test_tokenizer, mask_prompt=True)
        token_ids, loss_mask = ds[0]
        assert len(token_ids) == len(loss_mask)
        assert 1 in loss_mask
        assert 0 in loss_mask


class TestSFTDatasetOpenHermes:
    @pytest.fixture
    def openhermes_data(self):
        return [
            {
                "conversations": [
                    {"from": "human", "value": "What is 2+2?"},
                    {"from": "gpt", "value": "4"},
                ],
            },
        ]

    @pytest.fixture
    def openhermes_data_with_system(self):
        return [
            {
                "system_prompt": "You are a math tutor.",
                "conversations": [
                    {"from": "human", "value": "What is 2+2?"},
                    {"from": "gpt", "value": "4"},
                ],
            },
        ]

    def test_conversations_format(self, test_tokenizer, openhermes_data):
        ds = SFTDataset(openhermes_data, test_tokenizer)
        token_ids, loss_mask = ds[0]
        assert isinstance(token_ids, list)
        assert isinstance(loss_mask, list)
        assert len(token_ids) == len(loss_mask)

    def test_conversations_with_system_prompt(self, test_tokenizer, openhermes_data_with_system):
        ds = SFTDataset(openhermes_data_with_system, test_tokenizer)
        token_ids, loss_mask = ds[0]
        assert len(token_ids) == len(loss_mask)
        decoded = test_tokenizer.decode(token_ids)
        assert "math tutor" in decoded

    def test_conversations_mask_prompt(self, test_tokenizer, openhermes_data):
        ds = SFTDataset(openhermes_data, test_tokenizer, mask_prompt=True)
        token_ids, loss_mask = ds[0]
        assert loss_mask[0] == 0
        assert loss_mask[-1] == 1

    def test_conversations_eos_appended(self, test_tokenizer, openhermes_data):
        ds = SFTDataset(openhermes_data, test_tokenizer)
        token_ids, _ = ds[0]
        assert token_ids[-1] == test_tokenizer.eos_token_id

    def test_conversations_multiturn(self, test_tokenizer):
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"},
                    {"from": "human", "value": "How are you?"},
                    {"from": "gpt", "value": "Fine, thanks!"},
                ],
            },
        ]
        ds = SFTDataset(data, test_tokenizer, mask_prompt=True)
        token_ids, loss_mask = ds[0]
        assert len(token_ids) == len(loss_mask)
        assert 1 in loss_mask
        assert 0 in loss_mask

    def test_conversations_from_jsonl(self, test_tokenizer, tmp_path):
        path = tmp_path / "openhermes.jsonl"
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"},
                ],
            },
        ]
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        ds = SFTDataset(path, test_tokenizer)
        assert len(ds) == 1

    def test_neither_key_raises(self, test_tokenizer):
        data = [{"text": "just some text"}]
        ds = SFTDataset(data, test_tokenizer)
        with pytest.raises(KeyError, match=r"messages.*conversations"):
            ds[0]

    def test_null_system_prompt_ignored(self, test_tokenizer):
        data = [
            {
                "system_prompt": None,
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"},
                ],
            },
        ]
        ds = SFTDataset(data, test_tokenizer)
        token_ids, loss_mask = ds[0]
        assert len(token_ids) == len(loss_mask)


class TestAlpacaDataset:
    @pytest.fixture
    def alpaca_data(self):
        return [
            {"instruction": "Translate to French", "input": "", "output": "Bonjour"},
        ]

    @pytest.fixture
    def alpaca_data_with_input(self):
        return [
            {"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"},
        ]

    def test_alpaca_format(self, test_tokenizer, alpaca_data):
        ds = AlpacaDataset(alpaca_data, test_tokenizer)
        token_ids, loss_mask = ds[0]
        assert isinstance(token_ids, list)
        assert isinstance(loss_mask, list)
        assert len(token_ids) == len(loss_mask)

    def test_alpaca_with_input(self, test_tokenizer, alpaca_data_with_input):
        ds = AlpacaDataset(alpaca_data_with_input, test_tokenizer)
        token_ids, loss_mask = ds[0]
        decoded = test_tokenizer.decode(token_ids)
        assert "Hello world" in decoded
        assert "Bonjour le monde" in decoded

    def test_alpaca_empty_input(self, test_tokenizer, alpaca_data):
        ds = AlpacaDataset(alpaca_data, test_tokenizer)
        token_ids, loss_mask = ds[0]
        decoded = test_tokenizer.decode(token_ids)
        assert "Translate to French" in decoded
        assert "Bonjour" in decoded

    def test_alpaca_mask_prompt(self, test_tokenizer, alpaca_data):
        ds = AlpacaDataset(alpaca_data, test_tokenizer, mask_prompt=True)
        token_ids, loss_mask = ds[0]
        assert loss_mask[0] == 0
        assert loss_mask[-1] == 1


class TestORPODataset:
    @pytest.fixture
    def orpo_data_messages(self):
        return [
            {
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "chosen": [{"role": "assistant", "content": "4"}],
                "rejected": [{"role": "assistant", "content": "5"}],
            },
        ]

    @pytest.fixture
    def orpo_data_strings(self):
        return [
            {
                "prompt": "What is 2+2?",
                "chosen": "4",
                "rejected": "5",
            },
        ]

    def test_orpo_message_format(self, test_tokenizer, orpo_data_messages):
        ds = ORPODataset(orpo_data_messages, test_tokenizer)
        chosen_ids, chosen_mask, rejected_ids, rejected_mask = ds[0]
        assert isinstance(chosen_ids, list)
        assert isinstance(chosen_mask, list)
        assert isinstance(rejected_ids, list)
        assert isinstance(rejected_mask, list)

    def test_orpo_string_format(self, test_tokenizer, orpo_data_strings):
        ds = ORPODataset(orpo_data_strings, test_tokenizer)
        chosen_ids, chosen_mask, rejected_ids, rejected_mask = ds[0]
        assert len(chosen_ids) == len(chosen_mask)
        assert len(rejected_ids) == len(rejected_mask)

    def test_orpo_chosen_and_rejected_different(self, test_tokenizer, orpo_data_messages):
        ds = ORPODataset(orpo_data_messages, test_tokenizer)
        chosen_ids, _, rejected_ids, _ = ds[0]
        assert chosen_ids != rejected_ids

    def test_orpo_mask_prompt_in_chosen(self, test_tokenizer, orpo_data_messages):
        ds = ORPODataset(orpo_data_messages, test_tokenizer)
        chosen_ids, chosen_mask, _, _ = ds[0]
        assert chosen_mask[0] == 0
        assert chosen_mask[-1] == 1

    def test_orpo_mask_prompt_in_rejected(self, test_tokenizer, orpo_data_messages):
        ds = ORPODataset(orpo_data_messages, test_tokenizer)
        _, _, rejected_ids, rejected_mask = ds[0]
        assert rejected_mask[0] == 0
        assert rejected_mask[-1] == 1

    def test_orpo_truncation(self, test_tokenizer):
        long_text = "x" * 10000
        data = [
            {
                "prompt": [{"role": "user", "content": long_text}],
                "chosen": [{"role": "assistant", "content": "good"}],
                "rejected": [{"role": "assistant", "content": "bad"}],
            },
        ]
        ds = ORPODataset(data, test_tokenizer, max_seq_len=64)
        chosen_ids, _, rejected_ids, _ = ds[0]
        assert len(chosen_ids) <= 64
        assert len(rejected_ids) <= 64

    def test_orpo_len(self, test_tokenizer, orpo_data_messages):
        ds = ORPODataset(orpo_data_messages, test_tokenizer)
        assert len(ds) == 1

    def test_orpo_iter(self, test_tokenizer, orpo_data_messages):
        ds = ORPODataset(orpo_data_messages, test_tokenizer)
        items = list(ds)
        assert len(items) == 1

    def test_orpo_eos_appended(self, test_tokenizer, orpo_data_messages):
        ds = ORPODataset(orpo_data_messages, test_tokenizer)
        chosen_ids, _, rejected_ids, _ = ds[0]
        assert chosen_ids[-1] == test_tokenizer.eos_token_id
        assert rejected_ids[-1] == test_tokenizer.eos_token_id

    def test_orpo_loads_from_jsonl(self, test_tokenizer, tmp_path):
        path = tmp_path / "orpo.jsonl"
        data = [
            {
                "prompt": [{"role": "user", "content": "Hi"}],
                "chosen": [{"role": "assistant", "content": "Hello!"}],
                "rejected": [{"role": "assistant", "content": "Bye!"}],
            },
        ]
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        ds = ORPODataset(path, test_tokenizer)
        assert len(ds) == 1


class TestCacheDataset:
    @pytest.fixture
    def cached_sft(self, test_tokenizer):
        data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
            {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]},
        ]
        inner = SFTDataset(data, test_tokenizer)
        return CacheDataset(inner)

    def test_caches_on_first_access(self, cached_sft):
        result1 = cached_sft[0]
        result2 = cached_sft[0]
        assert result1 is result2

    def test_len_matches_wrapped_dataset(self, cached_sft):
        assert len(cached_sft) == 2

    def test_itemlen_returns_token_length(self, cached_sft):
        item = cached_sft[0]
        assert cached_sft.itemlen(0) == len(item[0])

    def test_caches_orpo(self, test_tokenizer):
        data = [
            {
                "prompt": [{"role": "user", "content": "Hi"}],
                "chosen": [{"role": "assistant", "content": "Hello!"}],
                "rejected": [{"role": "assistant", "content": "Bye!"}],
            },
        ]
        inner = ORPODataset(data, test_tokenizer)
        cached = CacheDataset(inner)
        assert len(cached) == 1
        result1 = cached[0]
        result2 = cached[0]
        assert result1 is result2
        assert cached.itemlen(0) == len(result1[0])
