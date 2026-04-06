"""Implementation of the `bit-axon prepare` command."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from bit_axon.cli._console import print_error, print_success

console = Console()

VALID_FORMATS = ("alpaca", "messages", "orpo")


def prepare_cmd(
    dataset: str,
    format: str,
    output: str,
    split: str,
    limit: int | None,
) -> None:
    """Convert HuggingFace dataset to JSONL for training."""
    try:
        from datasets import load_dataset
    except ImportError:
        print_error("`datasets` package is not installed. Install it with: pip install 'bit-axon[evaluation]'")
        raise SystemExit(1) from None

    if format not in VALID_FORMATS:
        from typer import BadParameter

        raise BadParameter(f"Format must be one of {VALID_FORMATS}, got '{format}'")

    if not output:
        output = f"{dataset.replace('/', '_')}_{split}.jsonl"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with console.status(f"[bold green]Loading dataset {dataset} ({split})...", spinner="dots"):
        ds = load_dataset(dataset, split=split)

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows = [ds[i] for i in range(len(ds))]

    if format == "messages":
        for row in rows:
            if "messages" not in row:
                instruction = row.get("instruction", "")
                input_text = row.get("input", "")
                assistant_output = row.get("output", "")
                content = f"{instruction}\n\n{input_text}" if input_text else instruction
                row["messages"] = [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": assistant_output},
                ]

    with open(output, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_success(f"Wrote {len(rows)} rows to {output}")
