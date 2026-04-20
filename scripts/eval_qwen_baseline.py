"""Run the Bit-Axon benchmark harness against Qwen2.5-3B for a reference baseline.

Why this exists
---------------
Our evaluation harness (``src/bit_axon/evaluation/benchmark.py``) expects the
model to present Bit-Axon's forward interface: ``model(input_ids, cache=None)``
returning ``(logits, caches)``. mlx-lm's Qwen models return just logits and use
``make_prompt_cache`` for KV state. ``MlxLmBitAxonAdapter`` bridges the two so
we can run the exact same harness, tasks, prompts, and scoring against Qwen
and later against our Bit-Axon checkpoints — apples-to-apples.

Usage
-----
::

    python scripts/eval_qwen_baseline.py \\
        --model Qwen/Qwen2.5-3B \\
        --output-dir pipeline_output/qwen_baseline \\
        --mmlu-limit 2500 --hellaswag-limit 2500

Outputs ``qwen2.5-3b_eval.json`` with per-benchmark accuracy, totals, and timing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_lm_load
from mlx_lm.models.cache import make_prompt_cache
from rich.console import Console

from bit_axon.evaluation.benchmark import evaluate_benchmark
from bit_axon.evaluation.tasks import BenchmarkConfig
from bit_axon.tokenizer import QwenTokenizerWrapper


class MlxLmBitAxonAdapter(nn.Module):
    """Expose an mlx-lm model via Bit-Axon's ``(logits, caches)`` contract.

    The Bit-Axon harness passes ``cache=caches`` back in on each incremental
    step. mlx-lm mutates its cache list in place, so we return the same list
    the caller handed us (or a freshly allocated one on prefill).
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def __call__(self, input_ids: mx.array, cache: list | None = None):
        if cache is None:
            cache = make_prompt_cache(self.inner)
        logits = self.inner(input_ids, cache=cache)
        return logits, cache


# Benchmarks fall into two buckets:
#   - "full" runs the entire eval split (small enough to complete quickly).
#   - "capped" caps MMLU / HellaSwag at --mmlu-limit / --hellaswag-limit
#     because their eval splits (14k / 10k) would otherwise dominate wall time.
_FULL_BENCHMARKS = ["gsm8k", "arc_challenge", "arc_easy", "winogrande"]
_CAPPED_BENCHMARKS = ["mmlu", "hellaswag"]


def _run_one(
    adapter: MlxLmBitAxonAdapter,
    tokenizer: QwenTokenizerWrapper,
    name: str,
    limit: int | None,
    console: Console,
) -> dict:
    cfg = BenchmarkConfig(limit=limit, max_tokens=256, scoring_method="generate")
    console.print(f"[bold cyan]Running {name}[/] (limit={limit})")
    t0 = time.perf_counter()
    result = evaluate_benchmark(adapter, tokenizer, name, cfg, console=console)
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "accuracy": round(result.accuracy, 6),
        "std_error": round(result.std_error, 6),
        "correct": result.correct,
        "total": result.total,
        "failed": result.failed,
        "time_seconds": round(elapsed, 1),
        "limit": limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B", help="HF repo id of the mlx-lm model to evaluate")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B", help="HF repo id providing tokenizer.json")
    parser.add_argument("--output-dir", default="pipeline_output/qwen_baseline", help="Directory for the results JSON")
    parser.add_argument(
        "--default-limit",
        type=int,
        default=None,
        help="Default item cap applied to every benchmark unless overridden by a per-benchmark flag. Pass 0 or omit for no cap.",
    )
    # Per-benchmark overrides. When set, they win over --default-limit.
    parser.add_argument("--gsm8k-limit", type=int, default=None, help="Item cap for GSM8K (overrides --default-limit)")
    parser.add_argument("--arc-challenge-limit", type=int, default=None, help="Item cap for ARC-Challenge (overrides --default-limit)")
    parser.add_argument("--arc-easy-limit", type=int, default=None, help="Item cap for ARC-Easy (overrides --default-limit)")
    parser.add_argument("--winogrande-limit", type=int, default=None, help="Item cap for Winogrande (overrides --default-limit)")
    parser.add_argument("--mmlu-limit", type=int, default=None, help="Item cap for MMLU (overrides --default-limit)")
    parser.add_argument("--hellaswag-limit", type=int, default=None, help="Item cap for HellaSwag (overrides --default-limit)")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these benchmarks (subset of the 6)")
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold green]Loading {args.model} via mlx-lm...")
    inner_model, _ = mlx_lm_load(args.model)
    adapter = MlxLmBitAxonAdapter(inner_model)
    tokenizer = QwenTokenizerWrapper(args.tokenizer)

    # --default-limit=0 means "no cap", matching the behaviour the user gets
    # by omitting the flag. This keeps the two inputs symmetric and prevents
    # a 0 from accidentally short-circuiting every benchmark to zero items.
    default_lim = args.default_limit if args.default_limit and args.default_limit > 0 else None
    per_bench_overrides: dict[str, int | None] = {
        "gsm8k": args.gsm8k_limit,
        "arc_challenge": args.arc_challenge_limit,
        "arc_easy": args.arc_easy_limit,
        "winogrande": args.winogrande_limit,
        "mmlu": args.mmlu_limit,
        "hellaswag": args.hellaswag_limit,
    }

    def _resolve_limit(name: str) -> int | None:
        override = per_bench_overrides.get(name)
        return override if override is not None else default_lim

    plan: list[tuple[str, int | None]] = []
    for name in _FULL_BENCHMARKS + _CAPPED_BENCHMARKS:
        plan.append((name, _resolve_limit(name)))
    if args.only:
        plan = [(n, lim) for (n, lim) in plan if n in set(args.only)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for name, lim in plan:
        res = _run_one(adapter, tokenizer, name, lim, console)
        results.append(res)
        console.print(f"[bold yellow]{name}[/]: {res['accuracy']:.4f} ({res['correct']}/{res['total']})  {res['time_seconds']}s")
        # Persist after each benchmark so partial progress survives a crash.
        payload = {
            "model": args.model,
            "tokenizer": args.tokenizer,
            "benchmarks": results,
        }
        (out_dir / "qwen2.5-3b_eval.json").write_text(json.dumps(payload, indent=2))

    console.print("[bold green]Done. Summary:")
    for r in results:
        console.print(f"  {r['name']}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")


if __name__ == "__main__":
    main()
