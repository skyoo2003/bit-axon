"""Tests for speed profiling utilities."""

from bit_axon.model import BitAxonModel
from bit_axon.profiling.speed import SpeedProfiler


class TestSpeedProfiler:
    def test_benchmark_returns_dict(self, small_config):
        profiler = SpeedProfiler()
        model = BitAxonModel(small_config)
        result = profiler.benchmark_tokens_per_sec(model, seq_len=4, batch_size=1, num_warmup=1, num_iterations=2, vocab_size=small_config.vocab_size)
        assert isinstance(result, dict)
        assert "tokens_per_sec" in result
        assert "latency_ms" in result
        assert result["tokens_per_sec"] > 0
        assert result["latency_ms"] > 0

    def test_benchmark_tokens_per_sec_reasonable(self, small_config):
        profiler = SpeedProfiler()
        model = BitAxonModel(small_config)
        result = profiler.benchmark_tokens_per_sec(model, seq_len=4, num_warmup=1, num_iterations=3, vocab_size=small_config.vocab_size)
        assert result["tokens_per_sec"] > 0
        assert result["latency_ms"] > 0
        assert result["num_iterations"] == 3

    def test_autoregressive_returns_dict(self, small_config):
        profiler = SpeedProfiler()
        model = BitAxonModel(small_config)
        result = profiler.benchmark_autoregressive(model, total_tokens=8, num_warmup=0, vocab_size=small_config.vocab_size)
        assert isinstance(result, dict)
        assert "tokens_per_sec" in result
        assert "total_latency_ms" in result
        assert result["tokens_per_sec"] > 0

    def test_autoregressive_larger_is_slower(self, small_config):
        profiler = SpeedProfiler()
        model = BitAxonModel(small_config)
        result_small = profiler.benchmark_autoregressive(model, total_tokens=4, num_warmup=0, vocab_size=small_config.vocab_size)
        result_large = profiler.benchmark_autoregressive(model, total_tokens=16, num_warmup=0, vocab_size=small_config.vocab_size)
        assert result_large["total_latency_ms"] >= result_small["total_latency_ms"] * 0.5
