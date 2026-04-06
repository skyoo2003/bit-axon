"""Implementation of the `bit-axon quantize` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from bit_axon.cli._console import print_success

console = Console()


def quantize_cmd(
    model_path: Annotated[str, typer.Argument(help="Path to model directory")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "",
    bits: Annotated[int, typer.Option("--bits", "-b", help="Quantization bit-width")] = 4,
    group_size: Annotated[int, typer.Option("--group-size", "-g", help="Quantization group size")] = 64,
    config_small: Annotated[bool, typer.Option("--config-small", help="Use small model for testing")] = False,
) -> None:
    """Quantize model weights to lower bit-width."""
    from bit_axon.training.merging import save_merged_model

    if not output:
        output = str(Path(model_path) / f"q{bits}")

    if config_small:
        from bit_axon.config import BitAxonConfig
        from bit_axon.model import BitAxonModel

        config = BitAxonConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            d_source_model=128,
            vocab_size=1024,
        )
        with console.status("[bold green]Initializing small model...", spinner="dots"):
            model = BitAxonModel(config)
            import mlx.core as mx

            mx.eval(model.parameters())
    else:
        from bit_axon.inference.loader import load_model

        with console.status(f"[bold green]Loading model from {model_path}...", spinner="dots"):
            model = load_model(Path(model_path))
        config = model.config if hasattr(model, "config") else None

    with console.status(f"[bold green]Quantizing to {bits}-bit (group_size={group_size})...", spinner="dots"):
        from bit_axon.quantization.nf4 import replace_linear_with_quantized

        replace_linear_with_quantized(model, group_size=group_size, bits=bits)

    with console.status(f"[bold green]Saving to {output}...", spinner="dots"):
        save_merged_model(model, output, config=config)

    print_success(f"Quantized model saved to {output}")
