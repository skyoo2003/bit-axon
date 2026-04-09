from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from bit_axon.tokenizer import QwenTokenizerWrapper


def stream_jsonl(path: str | Path) -> Iterator[dict[str, object]]:
    """Stream JSONL file line by line without loading into memory.

    Skips empty lines and lines that fail JSON parsing.
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_line_offsets(path: str | Path) -> list[int]:
    """Scan JSONL file and record byte offsets for each valid line.

    Returns a list of byte offsets; seeking to offset[i] and reading one line
    yields the i-th JSON record.  This uses ~8 bytes per line (a Python int
    in a list) instead of materialising the full parsed dict.
    """
    offsets: list[int] = []
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def _read_jsonl_line(path: str | Path, offset: int) -> dict[str, object]:
    """Read and parse a single JSONL line at the given byte offset."""
    with open(path) as f:
        f.seek(offset)
        return json.loads(f.readline())


class _BaseJSONLDataset:
    def __init__(
        self,
        data: list[dict] | str | Path,
        tokenizer: QwenTokenizerWrapper,
        max_seq_len: int = 2048,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        if isinstance(data, (str, Path)):
            self._file_path = str(Path(data).resolve())
            self._offsets = _build_line_offsets(self._file_path)
            self._data: list[dict[str, object]] | None = None
        else:
            self._file_path = None
            self._offsets: list[int] | None = None
            self._data = data

    def __len__(self) -> int:
        if self._offsets is not None:
            return len(self._offsets)
        if self._data is None:
            raise RuntimeError("No in-memory data available")
        return len(self._data)

    def _get_raw(self, idx: int) -> dict[str, object]:
        if self._file_path is not None:
            if self._offsets is None:
                raise RuntimeError("No in-memory data available")
            return _read_jsonl_line(self._file_path, self._offsets[idx])
        if self._data is None:
            raise RuntimeError("No in-memory data available")
        return self._data[idx]


class SFTDataset(_BaseJSONLDataset):
    """Dataset for supervised fine-tuning with chat/messages format.

    Expected JSONL formats:

    OpenAI messages format:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    ShareGPT / OpenHermes format:
        {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}], "system_prompt": "..."}

    Yields (token_ids, loss_mask) tuples per example.
    If mask_prompt=True, loss is only computed on assistant response tokens.
    """

    _SHAREGPT_ROLE_MAP: ClassVar[dict[str, str]] = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    def __init__(
        self,
        data: list[dict] | str | Path,
        tokenizer: QwenTokenizerWrapper,
        max_seq_len: int = 2048,
        mask_prompt: bool = True,
    ) -> None:
        super().__init__(data, tokenizer, max_seq_len)
        self._mask_prompt = mask_prompt

    def _get_messages(self, raw: dict[str, object]) -> list[dict[str, str]]:
        if "messages" in raw:
            return raw["messages"]  # type: ignore[return-value]

        if "conversations" in raw:
            messages: list[dict[str, str]] = []
            system_prompt = raw.get("system_prompt")
            if isinstance(system_prompt, str) and system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for entry in raw["conversations"]:
                role = self._SHAREGPT_ROLE_MAP.get(entry["from"], entry["from"])
                messages.append({"role": role, "content": entry["value"]})

            return messages

        raise KeyError("Data must contain 'messages' (OpenAI) or 'conversations' (ShareGPT/OpenHermes) key")

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        raw = self._get_raw(idx)
        messages = self._get_messages(raw)
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
        prev_len = 0

        for msg in messages:
            accumulated.append(msg)
            cur_len = len(self._tokenizer.apply_chat_template(accumulated))

            if msg["role"] == "assistant":
                end = min(cur_len, len(token_ids))
                for j in range(prev_len, end):
                    mask[j] = 1

            prev_len = cur_len

        # Extend last assistant range to include appended EOS token.
        # prev_len now equals len(apply_chat_template(messages)).
        if messages and messages[-1]["role"] == "assistant":
            for j in range(prev_len, len(token_ids)):
                mask[j] = 1

        return mask


class AlpacaDataset(SFTDataset):
    """Dataset for alpaca (instruction/input/output) format.

    Expected JSONL format:
        {"instruction": "...", "input": "...", "output": "..."}

    Converts to messages format internally.
    """

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        raw = self._get_raw(idx)
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


class ORPODataset(_BaseJSONLDataset):
    """Dataset for ORPO (Odds Ratio Preference Optimization).

    Expected JSONL format:
        {"prompt": [{"role": "user", "content": "..."}], "chosen": [{"role": "assistant", "content": "..."}], "rejected": [{"role": "assistant", "content": "..."}]}

    Also supports string format:
        {"prompt": "...", "chosen": "...", "rejected": "..."}

    Yields (chosen_ids, chosen_mask, rejected_ids, rejected_mask) per example.
    """

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], list[int], list[int]]:
        raw = self._get_raw(idx)
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
        if result is None:
            raise RuntimeError("Failed to load dataset example")
        return result

    def itemlen(self, idx: int) -> int:
        item = self[idx]
        return len(item[0])
