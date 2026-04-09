"""Comprehensive benchmark suite for Bit-Axon model."""

from __future__ import annotations

import time

import mlx.core as mx

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel
from bit_axon.profiling.memory import MemoryProfiler
from bit_axon.profiling.thermal import ThermalMonitor


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self):
        self.results: dict[str, dict] = {}

    def add(self, name: str, metrics: dict) -> None:
        self.results[name] = metrics

    def to_table(self) -> str:
        """Format all results as ASCII table."""
        if not self.results:
            return "No benchmark results."

        header = f"{'Benchmark':<20} {'tok/s':>10} {'latency_ms':>12} {'peak_GB':>10} {'active_GB':>12} {'SoC_°C':>8}"
        separator = "-" * len(header)
        lines = [header, separator]

        for name, m in self.results.items():
            temp_str = f"{m['soc_temp_c']:.1f}" if m.get("soc_temp_c") is not None else "N/A"
            lines.append(
                f"{name:<20} {m['tokens_per_sec']:>10.1f} {m['latency_ms']:>12.2f} {m['peak_memory_gb']:>10.3f} {m['active_memory_gb']:>12.3f} {temp_str:>8}"
            )

        return "\n".join(lines)


class BenchmarkSuite:
    """Run comprehensive benchmarks across sequence lengths."""

    def __init__(self, config: BitAxonConfig | None = None):
        self.config = config or BitAxonConfig()
        self.memory_profiler = MemoryProfiler()
        self.thermal_monitor = ThermalMonitor()

    def benchmark_sequence_lengths(
        self,
        seq_lengths: list[int] | None = None,
        batch_size: int = 1,
        num_warmup: int = 2,
        num_iterations: int = 5,
    ) -> BenchmarkResult:
        """Benchmark model across multiple sequence lengths.

        For each sequence length, measures:
        - tokens/sec (prefill)
        - latency_ms
        - peak_memory_gb
        - active_memory_gb
        - soc_temperature (if available)
        """
        if seq_lengths is None:
            seq_lengths = [128, 512, 1024, 2048]

        result = BenchmarkResult()
        model = BitAxonModel(self.config)

        for seq_len in seq_lengths:
            for _ in range(num_warmup):
                input_ids = mx.random.randint(0, self.config.vocab_size, shape=(batch_size, seq_len), dtype=mx.uint32)
                model(input_ids)
                mx.synchronize()

            latencies = []
            self.memory_profiler.reset_peak()
            for _ in range(num_iterations):
                input_ids = mx.random.randint(0, self.config.vocab_size, shape=(batch_size, seq_len), dtype=mx.uint32)
                t0 = time.perf_counter()
                model(input_ids)
                mx.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            tokens_per_sec = (seq_len * batch_size) / (avg_latency / 1000)
            peak_mem = self.memory_profiler.peak_memory_gb()
            active_mem = self.memory_profiler.active_memory_gb()
            temp = self.thermal_monitor.get_soc_temperature()

            result.add(
                f"seq_{seq_len}",
                {
                    "seq_len": seq_len,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "latency_ms": round(avg_latency, 2),
                    "peak_memory_gb": round(peak_mem, 3),
                    "active_memory_gb": round(active_mem, 3),
                    "soc_temp_c": temp,
                },
            )

        return result
