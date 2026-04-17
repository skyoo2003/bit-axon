"""Evaluate model perplexity on WikiText-103."""

from __future__ import annotations

import json
from pathlib import Path

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
    config_medium: bool = False,
) -> None:
    """Evaluate model perplexity on WikiText-103 test set."""
    from bit_axon.config import BitAxonConfig
    from bit_axon.evaluation.dataset import WikiTextDataset
    from bit_axon.evaluation.perplexity import evaluate_ppl
    from bit_axon.model import BitAxonModel

    if config_small:
        config = BitAxonConfig.small()
        with console.status("[bold green]Creating small model..."):
            model = BitAxonModel(config)
            mx.eval(model.parameters())
        print_info("Using random weights (config-small mode)")
    elif config_medium:
        config = BitAxonConfig.medium()
        with console.status("[bold green]Loading model..."):
            from bit_axon.inference.loader import load_model

            model = load_model(model_path, config=config)
        print_success(f"Model loaded from {model_path}")
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

    # Auto-resize model vocab if tokenizer has larger vocab
    if tok is not None and model.config.vocab_size < tok.vocab_size:
        from bit_axon.inference.loader import resize_model_vocab

        resize_model_vocab(model, tok.vocab_size)

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

    ppl_json = {"perplexity": round(ppl, 4), "std_error": round(se, 4), "tokens": int(all_tokens.shape[0]), "chunks": len(ds)}
    if tok:
        ppl_json["tokenizer"] = tokenizer
    ppl_path = Path(model_path) / "evaluation_results.json"
    ppl_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if ppl_path.exists():
        with open(ppl_path) as f:
            existing = json.load(f)
    existing["perplexity"] = ppl_json
    with open(ppl_path, "w") as f:
        json.dump(existing, f, indent=2)
    print_success(f"Perplexity results saved to {ppl_path}")


def evaluate_benchmarks_cmd(
    model_path: str,
    config_small: bool,
    tokenizer: str,
    benchmarks: list[str],
    benchmark_limit: int | None,
    max_tokens: int,
    scoring_method: str = "generate",
    config_medium: bool = False,
) -> None:
    """Evaluate model on standard LLM benchmarks."""
    import mlx.core as mx

    from bit_axon.config import BitAxonConfig
    from bit_axon.evaluation.benchmark import evaluate_benchmarks
    from bit_axon.evaluation.tasks import BENCHMARK_REGISTRY, BenchmarkConfig
    from bit_axon.model import BitAxonModel

    if config_small:
        config = BitAxonConfig.small()
        with console.status("[bold green]Creating small model..."):
            model = BitAxonModel(config)
            mx.eval(model.parameters())
        print_info("Using random weights (config-small mode)")
    elif config_medium:
        config = BitAxonConfig.medium()
        with console.status("[bold green]Loading model..."):
            from bit_axon.inference.loader import load_model

            model = load_model(model_path, config=config)
        print_success(f"Model loaded from {model_path}")
    else:
        with console.status(f"[bold green]Loading model from {model_path}..."):
            from bit_axon.inference.loader import load_model

            config = BitAxonConfig()
            model = load_model(model_path, config=config)
        print_success(f"Model loaded from {model_path}")

    with console.status(f"[bold green]Loading tokenizer: {tokenizer}..."):
        from bit_axon.tokenizer import QwenTokenizerWrapper

        tok = QwenTokenizerWrapper(tokenizer)
    print_success(f"Tokenizer loaded: {tokenizer}")

    # Auto-resize model vocab if tokenizer has larger vocab
    if model.config.vocab_size < tok.vocab_size:
        from bit_axon.inference.loader import resize_model_vocab

        resize_model_vocab(model, tok.vocab_size)

    bench_config = BenchmarkConfig(benchmarks=benchmarks, limit=benchmark_limit, max_tokens=max_tokens, scoring_method=scoring_method)
    unknown = [b for b in benchmarks if b not in BENCHMARK_REGISTRY]
    if unknown:
        from bit_axon.cli._console import print_error as _err

        _err(f"Unknown benchmarks: {unknown}. Available: {list(BENCHMARK_REGISTRY.keys())}")
        raise SystemExit(1)
    results = evaluate_benchmarks(model, tok, config=bench_config, console=console)

    table = Table(title="Benchmark Evaluation")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Std Error", style="dim")
    table.add_column("Total", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Time", justify="right")
    for r in results:
        table.add_row(r.benchmark_name, f"{r.accuracy:.1%}", f"{r.std_error:.1%}", str(r.total), str(r.correct), f"{r.time_seconds:.1f}s")
    console.print(table)

    results_json = {
        "benchmarks": [
            {
                "name": r.benchmark_name,
                "accuracy": round(r.accuracy, 6),
                "std_error": round(r.std_error, 6),
                "correct": r.correct,
                "total": r.total,
                "time_seconds": round(r.time_seconds, 1),
            }
            for r in results
        ],
    }
    results_path = Path(model_path) / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print_success(f"Results saved to {results_path}")
