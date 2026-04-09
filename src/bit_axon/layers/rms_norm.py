from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """RMS normalization using MLX's fast kernel.

    Unlike LayerNorm, does not center the input (no mean subtraction).
    Normalizes by the RMS of the input and applies a learned scale.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)
