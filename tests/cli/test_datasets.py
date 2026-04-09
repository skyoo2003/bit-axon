"""Unit tests for dataset resolution (src/bit_axon/cli/_datasets.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bit_axon.cli._datasets import (
    _convert_sft_rows,
    resolve_orpo_data,
    resolve_sft_data,
)


def _make_fake_dataset(rows):
    ds = MagicMock()
    ds.__len__ = lambda self: len(rows)
    ds.__getitem__ = lambda self, idx: rows[idx]

    def _select(indices):
        subset = [rows[i] for i in indices]
        return _make_fake_dataset(subset)

    ds.select = _select
    return ds


# ---------------------------------------------------------------------------
# SFT resolution tests
# ---------------------------------------------------------------------------


class TestDatasetResolution:
    # --- resolve_sft_data --------------------------------------------------

    def test_resolve_sft_none(self):
        assert resolve_sft_data(None) is None

    def test_resolve_sft_local_jsonl(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"messages": []}\n', encoding="utf-8")
        result = resolve_sft_data(str(p))
        assert isinstance(result, str)
        assert result == str(p.resolve())

    def test_resolve_sft_preset_ultrachat(self):
        rows = [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "see ya"}]},
        ]
        fake_ds = _make_fake_dataset(rows)
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = resolve_sft_data("ultrachat")
        assert isinstance(result, list)
        assert len(result) == 2
        for row in result:
            assert "messages" in row

    def test_resolve_sft_preset_alpaca(self):
        rows = [
            {"instruction": "Translate to Korean", "input": "Hello", "output": "안녕하세요"},
            {"instruction": "Summarize", "input": "", "output": "A brief summary"},
        ]
        fake_ds = _make_fake_dataset(rows)
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = resolve_sft_data("alpaca")
        assert isinstance(result, list)
        assert len(result) == 2
        msgs = result[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert "Translate to Korean" in msgs[0]["content"]
        assert msgs[1]["content"] == "안녕하세요"

    def test_resolve_sft_preset_openorca(self):
        rows = [
            {
                "system_prompt": "You are helpful.",
                "question": "What is 2+2?",
                "response": "4",
            },
        ]
        fake_ds = _make_fake_dataset(rows)
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = resolve_sft_data("openorca")
        assert isinstance(result, list)
        assert len(result) == 1
        msgs = result[0]["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "4"

    def test_resolve_sft_limit(self):
        rows = [{"instruction": f"task{i}", "input": "", "output": f"ans{i}"} for i in range(100)]
        fake_ds = _make_fake_dataset(rows)
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = resolve_sft_data("alpaca", limit=5)
        assert isinstance(result, list)
        assert len(result) == 5

    # --- resolve_orpo_data -------------------------------------------------

    def test_resolve_orpo_none(self):
        assert resolve_orpo_data(None) is None

    def test_resolve_orpo_preset_ultrafeedback(self):
        rows = [
            {
                "chosen": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "good"},
                ],
                "rejected": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "bad"},
                ],
            },
        ]
        fake_ds = _make_fake_dataset(rows)
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = resolve_orpo_data("ultrafeedback")
        assert isinstance(result, list)
        assert len(result) == 1
        row = result[0]
        assert "prompt" in row
        assert "chosen" in row
        assert "rejected" in row
        assert row["prompt"] == [{"role": "user", "content": "q"}]
        assert row["chosen"] == [{"role": "assistant", "content": "good"}]
        assert row["rejected"] == [{"role": "assistant", "content": "bad"}]

    def test_resolve_orpo_preset_hh_rlhf(self):
        rows = [
            {
                "chosen": "Human: Hello\n\nAssistant: Hi there",
                "rejected": "Human: Hello\n\nAssistant: Bad response",
            },
        ]
        fake_ds = _make_fake_dataset(rows)
        with patch("datasets.load_dataset", return_value=fake_ds):
            result = resolve_orpo_data("hh-rlhf")
        assert isinstance(result, list)
        assert len(result) == 1
        row = result[0]
        # prompt should have user message "Hello"
        assert row["prompt"] == [{"role": "user", "content": "Hello"}]
        # chosen should be assistant with "Hi there"
        assert row["chosen"] == [{"role": "assistant", "content": "Hi there"}]
        # rejected should be assistant with "Bad response"
        assert row["rejected"] == [{"role": "assistant", "content": "Bad response"}]

    def test_resolve_orpo_local_jsonl(self, tmp_path):
        p = tmp_path / "prefs.jsonl"
        p.write_text('{"prompt": [], "chosen": [], "rejected": []}\n', encoding="utf-8")
        result = resolve_orpo_data(str(p))
        assert isinstance(result, str)
        assert result == str(p.resolve())

    # --- Direct converter unit tests ---------------------------------------

    def test_convert_alpaca_with_empty_input(self):
        rows = [{"instruction": "Do X", "input": "", "output": "Done"}]
        result = _convert_sft_rows(rows, "alpaca")
        user_content = result[0]["messages"][0]["content"]
        # When input is empty, content should be just the instruction (no double newlines)
        assert user_content == "Do X"
        assert "\n\n" not in user_content

    def test_convert_openorca_no_system_prompt(self):
        rows = [{"system_prompt": "", "question": "What?", "response": "Answer"}]
        result = _convert_sft_rows(rows, "openorca")
        msgs = result[0]["messages"]
        # No system message when system_prompt is empty
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

        # Also test missing key entirely
        rows2 = [{"question": "What?", "response": "Answer"}]
        result2 = _convert_sft_rows(rows2, "openorca")
        msgs2 = result2[0]["messages"]
        assert len(msgs2) == 2

    def test_convert_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown SFT format"):
            _convert_sft_rows([], "nonexistent_format")
