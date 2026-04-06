"""Implementation of the `bit-axon merge` command."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from bit_axon.cli._console import print_success

console = Console()


def merge_cmd(
    base_model: Annotated[str, typer.Argument(help="Base model directory")],
    adapter: Annotated[str, typer.Option("--adapter", "-a", help="Adapter weights path (.safetensors)")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "",
    no_re_quantize: Annotated[bool, typer.Option("--no-re-quantize", help="Skip re-quantization after merge")] = False,
    bits: Annotated[int, typer.Option("--bits", "-b", help="Quantization bit-width")] = 4,
    group_size: Annotated[int, typer.Option("--group-size", "-g", help="Quantization group size")] = 64,
    lora_rank: Annotated[int, typer.Option("--lora-rank", "-r", help="LoRA rank")] = 8,
) -> None:
    """Merge LoRA/DoRA adapter weights into a base model."""
    from bit_axon.training.merging import load_and_merge

    if not output:
        output = f"{base_model}-merged"

    with console.status("[bold green]Merging adapter into base model...", spinner="dots"):
        result_path = load_and_merge(
            base_model_path=base_model,
            adapter_path=adapter,
            output_dir=output,
            quantize_after_merge=not no_re_quantize,
            bits=bits,
            group_size=group_size,
            lora_rank=lora_rank,
        )

    print_success(f"Merged model saved to {result_path}")
