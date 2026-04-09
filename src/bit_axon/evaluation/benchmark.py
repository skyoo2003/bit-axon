"""Benchmark evaluation orchestrator for Bit-Axon."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from bit_axon.evaluation.tasks import BENCHMARK_REGISTRY, BenchmarkConfig
from bit_axon.inference.generate import GenerateConfig, GenerateResult, generate

if TYPE_CHECKING:
    from rich.console import Console

_TASK_MAX_TOKENS: dict[str, int] = {
    "mmlu": 5,
    "arc_challenge": 5,
    "arc_easy": 5,
    "hellaswag": 5,
    "winogrande": 5,
    "gsm8k": 512,
}


@dataclass
class BenchmarkResult:
    benchmark_name: str
    accuracy: float
    std_error: float
    total: int
    correct: int
    time_seconds: float
    category_scores: dict[str, tuple[float, int, int]] = field(default_factory=dict)


def evaluate_benchmark(
    model,
    tokenizer,
    benchmark_name: str,
    config: BenchmarkConfig,
    console: Console | None = None,
) -> BenchmarkResult:
    if benchmark_name not in BENCHMARK_REGISTRY:
        msg = f"Unknown benchmark: {benchmark_name!r}. Available: {list(BENCHMARK_REGISTRY)}"
        raise ValueError(msg)

    task_cls = BENCHMARK_REGISTRY[benchmark_name]
    task = task_cls()

    if console is not None:
        with console.status("[bold green]Loading benchmark data...") as status:

            def _loading_cb(msg: str) -> None:
                status.update(f"[bold green]{msg}")

            items = task.load_data(limit=config.limit, status_callback=_loading_cb)
    else:
        items = task.load_data(limit=config.limit)

    max_tokens = _TASK_MAX_TOKENS.get(benchmark_name, config.max_tokens)
    gen_config = GenerateConfig(max_tokens=max_tokens, temperature=0.0, top_k=0, top_p=1.0)

    correct = 0
    category_correct: dict[str, int] = {}
    category_total: dict[str, int] = {}

    t_start = time.perf_counter()

    if console is not None:
        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("  [cyan]{task.fields[acc]:.1%}[/]"),
            console=console,
        ) as progress:
            task_id = progress.add_task(benchmark_name, total=len(items), acc=0.0)
            for i, item in enumerate(items):
                try:
                    result = generate(model, tokenizer, task.format_prompt(item), gen_config)
                except Exception:
                    correct += 0
                    progress.update(task_id, completed=i + 1, acc=correct / (i + 1))
                    continue
                if not isinstance(result, GenerateResult):
                    raise TypeError(f"Expected GenerateResult, got {type(result).__name__}")
                predicted = task.extract_answer(result.text)
                if task.check_answer(predicted, item.answer):
                    correct += 1
                if item.category:
                    category_total[item.category] = category_total.get(item.category, 0) + 1
                    if task.check_answer(predicted, item.answer):
                        category_correct[item.category] = category_correct.get(item.category, 0) + 1
                progress.update(task_id, completed=i + 1, acc=correct / (i + 1))
    else:
        for item in items:
            try:
                result = generate(model, tokenizer, task.format_prompt(item), gen_config)
            except Exception:
                continue
            if not isinstance(result, GenerateResult):
                raise TypeError(f"Expected GenerateResult, got {type(result).__name__}")
            predicted = task.extract_answer(result.text)
            if task.check_answer(predicted, item.answer):
                correct += 1
            if item.category:
                category_total[item.category] = category_total.get(item.category, 0) + 1
                if task.check_answer(predicted, item.answer):
                    category_correct[item.category] = category_correct.get(item.category, 0) + 1

    elapsed = time.perf_counter() - t_start
    total = len(items)
    accuracy = correct / total if total > 0 else 0.0
    std_error = math.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0

    category_scores: dict[str, tuple[float, int, int]] = {}
    for cat, cat_total in category_total.items():
        cat_correct = category_correct.get(cat, 0)
        cat_acc = cat_correct / cat_total if cat_total > 0 else 0.0
        category_scores[cat] = (cat_acc, cat_correct, cat_total)

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        accuracy=accuracy,
        std_error=std_error,
        total=total,
        correct=correct,
        time_seconds=elapsed,
        category_scores=category_scores,
    )


def evaluate_benchmarks(
    model,
    tokenizer,
    benchmarks: list[str] | None = None,
    config: BenchmarkConfig | None = None,
    console: Console | None = None,
) -> list[BenchmarkResult]:
    cfg = config or BenchmarkConfig()
    names = benchmarks if benchmarks is not None else cfg.benchmarks
    results: list[BenchmarkResult] = []
    for name in names:
        results.append(evaluate_benchmark(model, tokenizer, name, cfg, console))
    return results
