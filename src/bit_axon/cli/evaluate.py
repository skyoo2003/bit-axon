"""Evaluate model perplexity on WikiText-103."""

from __future__ import annotations

import mlx.core as mx
from rich.console import Console
from rich.table import Table

from bit_axon.cli._console import print_info, print_success

console = Console()


def evaluate_cmd(
    model_path: str,
    config_small: bool,
    max_tokens: int,
    seq_length: int,
    tokenizer: str | None,
    batch_size: int,
) -> None:
    """Evaluate model perplexity on WikiText-103 test set."""
    from bit_axon.config import BitAxonConfig
    from bit_axon.evaluation.dataset import WikiTextDataset
    from bit_axon.evaluation.perplexity import evaluate_ppl
    from bit_axon.model import BitAxonModel

    if config_small:
        config = BitAxonConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            d_source_model=128,
            vocab_size=1024,
        )
        with console.status("[bold green]Creating small model..."):
            model = BitAxonModel(config)
            mx.eval(model.parameters())
        print_info("Using random weights (config-small mode)")
    else:
        with console.status(f"[bold green]Loading model from {model_path}..."):
            from bit_axon.inference.loader import load_model

            config = BitAxonConfig()
            model = load_model(model_path, config=config)
        print_success(f"Model loaded from {model_path}")

    tok = None
    if tokenizer:
        with console.status(f"[bold green]Loading tokenizer: {tokenizer}..."):
            from bit_axon.tokenizer import QwenTokenizerWrapper

            tok = QwenTokenizerWrapper(tokenizer)
        print_success(f"Tokenizer loaded: {tokenizer}")

    with console.status("[bold green]Loading WikiText-103 test set..."):
        ds = WikiTextDataset(split="test", seq_length=seq_length, max_tokens=max_tokens, tokenizer=tok)
        all_tokens = mx.concatenate([ds[i] for i in range(len(ds))])
    print_success(f"Loaded {all_tokens.shape[0]} tokens in {len(ds)} chunks")

    with console.status("[bold green]Computing perplexity..."):
        ppl, se = evaluate_ppl(model, all_tokens, batch_size=batch_size, seq_length=seq_length)

    table = Table(title="Perplexity Evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Perplexity", f"{ppl:.2f}")
    table.add_row("Std Error", f"{se:.2f}")
    table.add_row("Tokens", f"{all_tokens.shape[0]:,}")
    table.add_row("Chunks", str(len(ds)))
    if tok:
        table.add_row("Tokenizer", tokenizer)
    else:
        table.add_row("Tokenizer", "char-level (ord(c) % 256)")
    console.print(table)
