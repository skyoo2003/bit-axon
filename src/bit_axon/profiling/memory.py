"""Memory profiling utilities for MLX models."""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten


def format_memory(bytes: int) -> str:
    """Convert bytes to human-readable string: '2.53 GB'."""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024**2:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1024**3:
        return f"{bytes / 1024**2:.2f} MB"
    else:
        return f"{bytes / 1024**3:.2f} GB"


class MemoryProfiler:
    """Wraps MLX memory API for profiling model inference."""

    def reset_peak(self) -> None:
        """Reset peak memory counter."""
        mx.reset_peak_memory()

    def active_memory_gb(self) -> float:
        """Currently active GPU memory in GB."""
        return mx.get_active_memory() / (1024**3)

    def peak_memory_gb(self) -> float:
        """Peak GPU memory since last reset, in GB."""
        return mx.get_peak_memory() / (1024**3)

    def cache_memory_gb(self) -> float:
        """Cached memory not yet returned, in GB."""
        return mx.get_cache_memory() / (1024**3)

    def device_info(self) -> dict:
        """Return device information from mx.device_info()."""
        return mx.device_info(mx.gpu)

    def profile_model(self, model) -> dict:
        """Measure model weight memory.

        Returns dict with:
        - 'weight_memory_bytes': total weight memory
        - 'weight_memory_gb': human-readable
        - 'param_count': total number of parameters
        - 'num_arrays': number of distinct arrays
        """
        params = tree_flatten(model.parameters())
        weight_memory_bytes = sum(v.nbytes for _, v in params)
        param_count = sum(v.size for _, v in params)
        return {
            "weight_memory_bytes": weight_memory_bytes,
            "weight_memory_gb": weight_memory_bytes / (1024**3),
            "param_count": param_count,
            "num_arrays": len(params),
        }

    def profile_forward(self, model, seq_len: int, batch_size: int = 1) -> dict:
        """Run a forward pass and measure memory.

        Returns dict with:
        - 'active_memory_gb': memory after forward
        - 'peak_memory_gb': peak memory during forward
        - 'activation_memory_gb': estimated activation memory (peak - weight)
        """
        weight_memory_gb = self.profile_model(model)["weight_memory_gb"]
        self.reset_peak()
        input_ids = mx.random.randint(0, model.config.vocab_size, shape=(batch_size, seq_len), dtype=mx.uint32)
        model(input_ids)
        mx.synchronize()
        active = self.active_memory_gb()
        peak = self.peak_memory_gb()
        return {
            "active_memory_gb": active,
            "peak_memory_gb": peak,
            "activation_memory_gb": peak - weight_memory_gb,
        }
