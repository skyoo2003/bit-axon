"""Implementation of the `bit-axon benchmark` command."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_SEQ_LENGTHS = "128,512,1024,2048"


def _parse_seq_lengths(value: str) -> list[int]:
    return [int(s.strip()) for s in value.split(",")]


def benchmark_cmd(
    seq_lengths: Annotated[str, typer.Option("--seq-lengths", "-s", help="Comma-separated sequence lengths")] = DEFAULT_SEQ_LENGTHS,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 1,
    warmup: Annotated[int, typer.Option("--warmup", "-w", help="Warmup iterations")] = 2,
    iterations: Annotated[int, typer.Option("--iterations", "-i", help="Timed iterations")] = 5,
    config_small: Annotated[bool, typer.Option("--config-small", help="Use small config for testing")] = False,
) -> None:
    """Benchmark model performance across sequence lengths."""
    from bit_axon.config import BitAxonConfig
    from bit_axon.profiling.benchmark import BenchmarkSuite

    parsed_lengths = _parse_seq_lengths(seq_lengths)

    if config_small:
        config = BitAxonConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            d_source_model=128,
            vocab_size=1024,
        )
    else:
        config = BitAxonConfig()

    console.print("[bold]Bit-Axon Benchmark Suite[/bold]")
    console.print(f"Config: hidden_dim={config.hidden_dim}, layers={config.num_layers}, vocab={config.vocab_size}")
    console.print(f"Sequence lengths: {parsed_lengths}")
    console.print(f"Iterations: {warmup} warmup + {iterations} timed")
    console.print()

    suite = BenchmarkSuite(config)

    with console.status("[bold green]Running benchmarks...", spinner="dots"):
        results = suite.benchmark_sequence_lengths(
            seq_lengths=parsed_lengths,
            batch_size=batch_size,
            num_warmup=warmup,
            num_iterations=iterations,
        )

    table = Table(title="Benchmark Results")
    table.add_column("Benchmark", style="cyan", no_wrap=True)
    table.add_column("tok/s", justify="right", style="green")
    table.add_column("latency_ms", justify="right")
    table.add_column("peak_GB", justify="right")
    table.add_column("active_GB", justify="right")
    table.add_column("SoC °C", justify="right")

    for name, m in results.results.items():
        temp_str = f"{m['soc_temp_c']:.1f}" if m.get("soc_temp_c") is not None else "N/A"
        table.add_row(
            name,
            f"{m['tokens_per_sec']:.1f}",
            f"{m['latency_ms']:.2f}",
            f"{m['peak_memory_gb']:.3f}",
            f"{m['active_memory_gb']:.3f}",
            temp_str,
        )

    console.print(table)
