"""Resolve dataset arguments into data for SFTDataset / ORPODataset.

Supports three data sources, selected by the ``--data`` CLI argument:

1. **Preset names** — short aliases that expand to HuggingFace dataset IDs
   with pre-configured splits and format converters.
   See ``SFT_PRESETS`` and ``ORPO_PRESETS`` for the available presets.

2. **HuggingFace IDs** — any ``org/dataset`` string that is not a local file
   and not a preset name is loaded directly from the HuggingFace Hub.

3. **Local JSONL paths** — if the argument points to an existing file on disk,
   its resolved path is returned as a string so downstream classes can stream it.

Return types:
    - ``None`` when no data argument is provided (signals SimpleDataset fallback).
    - ``str`` when the argument resolves to a local JSONL file path.
    - ``list[dict]`` when data is loaded and converted from HuggingFace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bit_axon.utils import _require_datasets

SFT_PRESETS: dict[str, dict[str, str]] = {
    "ultrachat": {"hf_id": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "format": "messages"},
    "alpaca": {"hf_id": "tatsu-lab/alpaca", "split": "train", "format": "alpaca"},
    "openorca": {"hf_id": "Open-Orca/OpenOrca", "split": "train", "format": "openorca"},
}

ORPO_PRESETS: dict[str, dict[str, str]] = {
    "ultrafeedback": {"hf_id": "HuggingFaceH4/ultrafeedback_binarized_cleaned", "split": "train_prefs", "format": "ultrafeedback"},
    "hh-rlhf": {"hf_id": "Anthropic/hh-rlhf", "split": "train", "format": "hh-rlhf"},
}


def _load_hf_rows(hf_id: str, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load rows from a HuggingFace dataset as a list of dicts."""
    datasets = _require_datasets()
    ds = datasets.load_dataset(hf_id, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [ds[i] for i in range(len(ds))]


def _convert_sft_rows(rows: list[dict[str, Any]], fmt: str) -> list[dict[str, Any]]:
    """Convert raw HF rows to the unified ``messages`` format.

    Each output row contains a ``"messages"`` key whose value is a list of
    ``{"role": ..., "content": ...}`` dicts.
    """
    if fmt == "messages":
        return rows

    if fmt == "alpaca":
        converted: list[dict[str, Any]] = []
        for row in rows:
            instruction = row.get("instruction", "")
            inp = row.get("input", "")
            output = row.get("output", "")
            content = f"{instruction}\n\n{inp}" if inp else instruction
            converted.append(
                {
                    "messages": [
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": output},
                    ],
                }
            )
        return converted

    if fmt == "openorca":
        converted = []
        for row in rows:
            system_prompt = row.get("system_prompt", "")
            question = row.get("question", "")
            response = row.get("response", "")
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": response})
            converted.append({"messages": messages})
        return converted

    msg = f"Unknown SFT format: {fmt!r}"
    raise ValueError(msg)


def _parse_hh_text(text: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse ``Human: ...\\nAssistant: ...`` text into (prompt, response).

    Returns a tuple of ``(prompt_messages, response_messages)`` where each is a
    list of ``{"role": ..., "content": ...}`` dicts.
    """
    # Find the last "Assistant:" marker — everything after it is the response.
    # Everything before (excluding a leading "Human: " tag) forms the prompt.
    assistant_marker = "\n\nAssistant: "
    last_idx = text.rfind(assistant_marker)

    if last_idx == -1:
        # Fallback: treat entire text as a single user message.
        return [{"role": "user", "content": text}], [{"role": "assistant", "content": ""}]

    prompt_text = text[:last_idx]
    response_text = text[last_idx + len(assistant_marker) :]

    # Strip leading "Human: " from the prompt if present.
    if prompt_text.startswith("Human: "):
        prompt_text = prompt_text[len("Human: ") :]

    prompt_messages: list[dict[str, str]] = [{"role": "user", "content": prompt_text}]
    response_messages: list[dict[str, str]] = [{"role": "assistant", "content": response_text}]
    return prompt_messages, response_messages


def _convert_orpo_rows(rows: list[dict[str, Any]], fmt: str) -> list[dict[str, Any]]:
    """Convert raw HF rows to ORPO format with prompt/chosen/rejected.

    Each output row contains ``"prompt"``, ``"chosen"``, and ``"rejected"``
    keys.  Each value is a list of ``{"role": ..., "content": ...}`` dicts.
    """
    if fmt == "ultrafeedback":
        converted: list[dict[str, Any]] = []
        for row in rows:
            chosen = row["chosen"]
            rejected = row["rejected"]
            prompt = chosen[:-1]
            chosen_response = [{"role": "assistant", "content": chosen[-1]["content"]}]
            rejected_response = [{"role": "assistant", "content": rejected[-1]["content"]}]
            converted.append(
                {
                    "prompt": prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                }
            )
        return converted

    if fmt == "hh-rlhf":
        converted = []
        for row in rows:
            chosen_text = row["chosen"]
            rejected_text = row["rejected"]
            prompt_msgs, chosen_resp = _parse_hh_text(chosen_text)
            _, rejected_resp = _parse_hh_text(rejected_text)
            converted.append(
                {
                    "prompt": prompt_msgs,
                    "chosen": chosen_resp,
                    "rejected": rejected_resp,
                }
            )
        return converted

    msg = f"Unknown ORPO format: {fmt!r}"
    raise ValueError(msg)


def resolve_sft_data(
    data_arg: str | None,
    split: str = "train",
    limit: int | None = None,
) -> list[dict[str, Any]] | str | None:
    """Resolve an SFT data argument to data that ``SFTDataset`` can consume.

    Args:
        data_arg: Preset name, HuggingFace ID, local JSONL path, or ``None``.
        split: Dataset split to load (used for HF datasets).
        limit: Maximum number of rows to load.

    Returns:
        - ``None`` if *data_arg* is ``None``.
        - ``str`` resolved local path if *data_arg* points to an existing file.
        - ``list[dict]`` with ``"messages"`` keys when loaded from HuggingFace.
    """
    if data_arg is None:
        return None

    if Path(data_arg).exists():
        return str(Path(data_arg).resolve())

    if data_arg in SFT_PRESETS:
        preset = SFT_PRESETS[data_arg]
        rows = _load_hf_rows(preset["hf_id"], preset["split"], limit)
        return _convert_sft_rows(rows, preset["format"])

    rows = _load_hf_rows(data_arg, split, limit)
    return _convert_sft_rows(rows, "messages")


def resolve_orpo_data(
    data_arg: str | None,
    split: str = "train",
    limit: int | None = None,
) -> list[dict[str, Any]] | str | None:
    """Resolve an ORPO data argument to data that ``ORPODataset`` can consume.

    Args:
        data_arg: Preset name, HuggingFace ID, local JSONL path, or ``None``.
        split: Dataset split to load (used for HF datasets).
        limit: Maximum number of rows to load.

    Returns:
        - ``None`` if *data_arg* is ``None``.
        - ``str`` resolved local path if *data_arg* points to an existing file.
        - ``list[dict]`` with ``"prompt"``, ``"chosen"``, ``"rejected"`` keys
          when loaded from HuggingFace.
    """
    if data_arg is None:
        return None

    if Path(data_arg).exists():
        return str(Path(data_arg).resolve())

    if data_arg in ORPO_PRESETS:
        preset = ORPO_PRESETS[data_arg]
        rows = _load_hf_rows(preset["hf_id"], preset["split"], limit)
        return _convert_orpo_rows(rows, preset["format"])

    rows = _load_hf_rows(data_arg, split, limit)
    return _convert_orpo_rows(rows, "ultrafeedback")
