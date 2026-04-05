"""Speed profiling utilities for MLX model inference."""

import statistics
import time

import mlx.core as mx


class SpeedProfiler:
    """Measure inference speed (tokens/sec) and latency."""

    def benchmark_tokens_per_sec(
        self,
        model,
        seq_len: int = 128,
        batch_size: int = 1,
        num_warmup: int = 2,
        num_iterations: int = 5,
        vocab_size: int = 1024,
    ) -> dict:
        """Benchmark tokens/sec for prefill.

        Returns dict with:
        - 'tokens_per_sec': mean tokens/sec across iterations
        - 'latency_ms': mean latency in ms per forward pass
        - 'std_latency_ms': standard deviation of latency
        - 'num_iterations': number of timed iterations
        """
        input_ids = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len), dtype=mx.uint32)

        # Warmup iterations
        for _ in range(num_warmup):
            model(input_ids)
            mx.synchronize()

        # Timed iterations
        latencies = []
        for _ in range(num_iterations):
            mx.synchronize()
            start = time.perf_counter()
            model(input_ids)
            mx.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        total_tokens = batch_size * seq_len
        tokens_per_sec = total_tokens / (mean_latency / 1000)

        return {
            "tokens_per_sec": tokens_per_sec,
            "latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "num_iterations": num_iterations,
        }

    def benchmark_autoregressive(
        self,
        model,
        total_tokens: int = 64,
        num_warmup: int = 2,
        vocab_size: int = 1024,
    ) -> dict:
        """Benchmark autoregressive token generation speed.

        Simulates decode-phase: generates one token at a time using KV cache.

        Returns dict with:
        - 'tokens_per_sec': total tokens / total time
        - 'total_latency_ms': total generation time
        - 'mean_per_token_ms': average latency per token
        """
        # Warmup
        for _ in range(num_warmup):
            warmup_ids = mx.array([[0]], dtype=mx.uint32)
            _, cache = model(warmup_ids)
            for __ in range(4):
                _, cache = model(mx.array([[0]], dtype=mx.uint32), cache=cache)
            mx.synchronize()

        # Timed generation
        current_token = mx.array([[0]], dtype=mx.uint32)
        mx.synchronize()
        start = time.perf_counter()

        # Initial forward to get first cache state
        logits, cache = model(current_token)
        mx.synchronize()

        for _ in range(total_tokens - 1):
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            logits, cache = model(next_token, cache=cache)
            mx.synchronize()

        end = time.perf_counter()
        total_latency_ms = (end - start) * 1000
        tokens_per_sec = total_tokens / (total_latency_ms / 1000)
        mean_per_token_ms = total_latency_ms / total_tokens

        return {
            "tokens_per_sec": tokens_per_sec,
            "total_latency_ms": total_latency_ms,
            "mean_per_token_ms": mean_per_token_ms,
        }
