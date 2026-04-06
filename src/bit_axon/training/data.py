from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from bit_axon.training.tokenizer import QwenTokenizerWrapper


def stream_jsonl(path: str | Path) -> Iterator[dict[str, object]]:
    """Stream JSONL file line by line without loading into memory.

    Skips empty lines and lines that fail JSON parsing.
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class SFTDataset:
    """Dataset for supervised fine-tuning with chat/messages format.

    Expected JSONL format:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Yields (token_ids, loss_mask) tuples per example.
    If mask_prompt=True, loss is only computed on assistant response tokens.
    """

    def __init__(
        self,
        data: list[dict] | str | Path,
        tokenizer: QwenTokenizerWrapper,
        max_seq_len: int = 2048,
        mask_prompt: bool = True,
    ) -> None:
        if isinstance(data, (str, Path)):
            self._data = list(stream_jsonl(data))
        else:
            self._data = data
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._mask_prompt = mask_prompt

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        raw = self._data[idx]
        messages = raw["messages"]
        token_ids = self._tokenizer.apply_chat_template(messages)

        if token_ids[-1] != self._tokenizer.eos_token_id:
            token_ids.append(self._tokenizer.eos_token_id)

        token_ids = token_ids[: self._max_seq_len]

        loss_mask = self._compute_loss_mask(token_ids, messages)

        return token_ids, loss_mask

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        for i in range(len(self)):
            yield self[i]

    def _compute_loss_mask(self, token_ids: list[int], messages: list[dict[str, str]]) -> list[int]:
        """Compute binary loss mask. 1 for assistant tokens (including appended EOS), 0 otherwise."""
        if not self._mask_prompt:
            return [1] * len(token_ids)

        mask = [0] * len(token_ids)
        accumulated: list[dict[str, str]] = []
        for msg in messages:
            prev_len = len(self._tokenizer.apply_chat_template(accumulated)) if accumulated else 0
            accumulated.append(msg)
            cur_len = len(self._tokenizer.apply_chat_template(accumulated))

            if msg["role"] == "assistant":
                end = min(cur_len, len(token_ids))
                for j in range(prev_len, end):
                    mask[j] = 1

        # Extend last assistant range to include appended EOS token
        if messages and messages[-1]["role"] == "assistant":
            last_assistant_end = len(self._tokenizer.apply_chat_template(messages))
            for j in range(last_assistant_end, len(token_ids)):
                mask[j] = 1

        return mask


class AlpacaDataset(SFTDataset):
    """Dataset for alpaca (instruction/input/output) format.

    Expected JSONL format:
        {"instruction": "...", "input": "...", "output": "..."}

    Converts to messages format internally.
    """

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        raw = self._data[idx]
        messages = self._parse_messages(raw)
        token_ids = self._tokenizer.apply_chat_template(messages)

        if token_ids[-1] != self._tokenizer.eos_token_id:
            token_ids.append(self._tokenizer.eos_token_id)

        token_ids = token_ids[: self._max_seq_len]
        loss_mask = self._compute_loss_mask(token_ids, messages)

        return token_ids, loss_mask

    @staticmethod
    def _parse_messages(raw: dict[str, object]) -> list[dict[str, str]]:
        instruction = raw["instruction"]
        inp = raw.get("input", "")
        output = raw["output"]

        user_content = f"{instruction}\n\n{inp}" if inp else instruction

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]


class ORPODataset:
    """Dataset for ORPO (Odds Ratio Preference Optimization).

    Expected JSONL format:
        {"prompt": [{"role": "user", "content": "..."}], "chosen": [{"role": "assistant", "content": "..."}], "rejected": [{"role": "assistant", "content": "..."}]}

    Also supports string format:
        {"prompt": "...", "chosen": "...", "rejected": "..."}

    Yields (chosen_ids, chosen_mask, rejected_ids, rejected_mask) per example.
    """

    def __init__(
        self,
        data: list[dict[str, object]] | str | Path,
        tokenizer: QwenTokenizerWrapper,
        max_seq_len: int = 2048,
    ) -> None:
        if isinstance(data, (str, Path)):
            self._data = list(stream_jsonl(data))
        else:
            self._data = data
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], list[int], list[int]]:
        raw = self._data[idx]
        prompt_messages = self._parse_prompt(raw["prompt"])
        chosen_messages = self._parse_response(raw["chosen"])
        rejected_messages = self._parse_response(raw["rejected"])

        chosen_ids = self._tokenize_pair(prompt_messages, chosen_messages)
        rejected_ids = self._tokenize_pair(prompt_messages, rejected_messages)

        prompt_len = len(self._tokenizer.apply_chat_template(prompt_messages))

        chosen_ids = chosen_ids[: self._max_seq_len]
        rejected_ids = rejected_ids[: self._max_seq_len]

        chosen_mask = self._build_mask(chosen_ids, prompt_len)
        rejected_mask = self._build_mask(rejected_ids, prompt_len)

        return chosen_ids, chosen_mask, rejected_ids, rejected_mask

    def __iter__(self) -> Iterator[tuple[list[int], list[int], list[int], list[int]]]:
        for i in range(len(self)):
            yield self[i]

    def _parse_prompt(self, prompt: list[dict[str, str]] | str) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    @staticmethod
    def _parse_response(response: list[dict[str, str]] | str) -> list[dict[str, str]]:
        if isinstance(response, str):
            return [{"role": "assistant", "content": response}]
        return response

    def _tokenize_pair(self, prompt_messages: list[dict[str, str]], response_messages: list[dict[str, str]]) -> list[int]:
        full_messages = prompt_messages + response_messages
        token_ids = self._tokenizer.apply_chat_template(full_messages)
        if token_ids[-1] != self._tokenizer.eos_token_id:
            token_ids.append(self._tokenizer.eos_token_id)
        return token_ids

    @staticmethod
    def _build_mask(token_ids: list[int], prompt_len: int) -> list[int]:
        mask = [0] * len(token_ids)
        for i in range(prompt_len, len(mask)):
            mask[i] = 1
        return mask


class CacheDataset:
    """Wraps a dataset and caches processed examples to avoid re-tokenization.

    Tokenization is done once lazily per sample and cached for subsequent access.
    """

    def __init__(self, dataset: SFTDataset | AlpacaDataset | ORPODataset) -> None:
        self._dataset = dataset
        self._cache: list[tuple | None] = [None] * len(dataset)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple:
        if self._cache[idx] is None:
            self._cache[idx] = self._dataset[idx]
        result = self._cache[idx]
        assert result is not None
        return result

    def itemlen(self, idx: int) -> int:
        item = self[idx]
        return len(item[0])
