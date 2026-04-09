from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from bit_axon.evaluation.benchmark import BenchmarkResult, evaluate_benchmark, evaluate_benchmarks
from bit_axon.evaluation.tasks import BenchmarkConfig, BenchmarkItem, MMLUTask
from bit_axon.inference.generate import GenerateResult


def _mock_generate_return(text: str = "A") -> GenerateResult:
    return GenerateResult(
        text=text,
        token_ids=[1],
        prompt_tokens=10,
        completion_tokens=1,
        tokens_per_sec=100.0,
    )


def _mock_items(n: int, answer: str = "B", category: str = "") -> list[BenchmarkItem]:
    return [BenchmarkItem(id=f"test_{i}", prompt=f"Question {i}\nA. X\nB. Y\nAnswer:", answer=answer, category=category) for i in range(n)]


class TestEvaluateBenchmark:
    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            evaluate_benchmark(MagicMock(), MagicMock(), "nonexistent_benchmark", BenchmarkConfig())

    @patch("bit_axon.evaluation.benchmark.generate", return_value=_mock_generate_return("B"))
    def test_perfect_model_100_percent(self, mock_gen):
        items = _mock_items(5, answer="B")
        with patch.object(MMLUTask, "load_data", return_value=items):
            result = evaluate_benchmark(MagicMock(), MagicMock(), "mmlu", BenchmarkConfig(limit=5))
        assert result.accuracy == 1.0
        assert result.total == 5
        assert result.correct == 5

    @patch("bit_axon.evaluation.benchmark.generate", return_value=_mock_generate_return("X"))
    def test_random_model_finite_results(self, mock_gen):
        items = _mock_items(3, answer="B")
        with patch.object(MMLUTask, "load_data", return_value=items):
            result = evaluate_benchmark(MagicMock(), MagicMock(), "mmlu", BenchmarkConfig(limit=3))
        assert result.accuracy == 0.0
        assert result.total == 3
        assert result.correct == 0
        assert result.std_error >= 0.0

    @patch("bit_axon.evaluation.benchmark.generate", return_value=_mock_generate_return("B"))
    def test_limit_parameter(self, mock_gen):
        all_items = _mock_items(100, answer="B")
        with patch.object(MMLUTask, "load_data", return_value=all_items[:10]):
            result = evaluate_benchmark(MagicMock(), MagicMock(), "mmlu", BenchmarkConfig(limit=10))
        assert result.total == 10
        assert mock_gen.call_count == 10

    @patch("bit_axon.evaluation.benchmark.generate", return_value=_mock_generate_return("B"))
    def test_category_scores_computed(self, mock_gen):
        items = [
            BenchmarkItem(id="t0", prompt="Q0", answer="B", category="math"),
            BenchmarkItem(id="t1", prompt="Q1", answer="B", category="math"),
            BenchmarkItem(id="t2", prompt="Q2", answer="C", category="history"),
        ]
        with patch.object(MMLUTask, "load_data", return_value=items):
            result = evaluate_benchmark(MagicMock(), MagicMock(), "mmlu", BenchmarkConfig(limit=3))
        assert "math" in result.category_scores
        assert result.category_scores["math"] == (1.0, 2, 2)
        assert result.category_scores["history"] == (0.0, 0, 1)

    @patch("bit_axon.evaluation.benchmark.generate", return_value=_mock_generate_return("B"))
    def test_empty_dataset_returns_zero(self, mock_gen):
        with patch.object(MMLUTask, "load_data", return_value=[]):
            result = evaluate_benchmark(MagicMock(), MagicMock(), "mmlu", BenchmarkConfig())
        assert result.accuracy == 0.0
        assert result.total == 0
        assert result.correct == 0
        assert result.std_error == 0.0

    def test_std_error_computation(self):
        result = BenchmarkResult(
            benchmark_name="test",
            accuracy=0.8,
            std_error=0.0,
            total=100,
            correct=80,
            time_seconds=1.0,
        )
        import math

        expected_se = math.sqrt(0.8 * 0.2 / 100)
        actual_se = math.sqrt(result.accuracy * (1 - result.accuracy) / result.total)
        assert abs(actual_se - expected_se) < 1e-10

    @patch("bit_axon.evaluation.benchmark.generate", return_value=_mock_generate_return("B"))
    def test_with_rich_console(self, mock_gen):
        items = _mock_items(2, answer="B")
        console = Console(file=io.StringIO(), width=80, legacy_windows=False)
        with patch.object(MMLUTask, "load_data", return_value=items):
            result = evaluate_benchmark(MagicMock(), MagicMock(), "mmlu", BenchmarkConfig(), console=console)
        assert result.accuracy == 1.0
        assert result.total == 2


class TestEvaluateBenchmarks:
    @patch("bit_axon.evaluation.benchmark.evaluate_benchmark")
    def test_runs_multiple_benchmarks(self, mock_eval):
        mock_eval.return_value = BenchmarkResult("mmlu", 0.5, 0.01, 100, 50, 1.0)
        results = evaluate_benchmarks(MagicMock(), MagicMock(), benchmarks=["mmlu", "gsm8k"])
        assert len(results) == 2
        assert mock_eval.call_count == 2

    @patch("bit_axon.evaluation.benchmark.evaluate_benchmark")
    def test_empty_benchmarks_list(self, mock_eval):
        results = evaluate_benchmarks(MagicMock(), MagicMock(), benchmarks=[])
        assert results == []
        assert mock_eval.call_count == 0

    @patch("bit_axon.evaluation.benchmark.evaluate_benchmark")
    def test_default_benchmarks(self, mock_eval):
        mock_eval.return_value = BenchmarkResult("x", 0.5, 0.01, 10, 5, 1.0)
        config = BenchmarkConfig(benchmarks=["mmlu"])
        results = evaluate_benchmarks(MagicMock(), MagicMock(), config=config)
        assert len(results) == 1
        mock_eval.assert_called_once()


class TestBenchmarkResult:
    def test_fields_populated(self):
        r = BenchmarkResult(
            benchmark_name="mmlu",
            accuracy=0.75,
            std_error=0.043,
            total=100,
            correct=75,
            time_seconds=12.3,
            category_scores={"math": (0.8, 40, 50)},
        )
        assert r.benchmark_name == "mmlu"
        assert r.accuracy == 0.75
        assert r.std_error == 0.043
        assert r.total == 100
        assert r.correct == 75
        assert r.time_seconds == 12.3
        assert r.category_scores["math"] == (0.8, 40, 50)

    def test_default_category_scores(self):
        r = BenchmarkResult("test", 0.5, 0.01, 10, 5, 1.0)
        assert r.category_scores == {}
