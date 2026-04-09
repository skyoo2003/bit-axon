from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from bit_axon.training._adapter_base import _BaseAdapterLinear


class DoRALinear(_BaseAdapterLinear):
    """Weight-Decomposed Low-Rank Adaptation (DoRA) wrapper.

    Like LoRA but re-normalizes the adapted output to match the magnitude of the
    original weight matrix. Stores the per-output-dim norm of the base weight in
    ``m`` and divides by the norm of the adapted weight during forward.

    Args:
        input_dims: Input dimension of the linear layer.
        output_dims: Output dimension of the linear layer.
        r: LoRA rank.
        dropout: Dropout probability applied before the low-rank path.
        scale: Scaling factor for the LoRA output.
        bias: Whether to include a bias term in the base linear layer.

    Attributes:
        linear: Base frozen linear layer (nn.Linear or nn.QuantizedLinear).
        lora_a: Low-rank matrix A of shape (input_dims, r).
        lora_b: Low-rank matrix B of shape (r, output_dims), initialized to zeros.
        m: Magnitude vector of shape (output_dims,), frozen norm of base weight rows.
        scale: Output scaling factor.
    """

    def __init__(self, input_dims, output_dims, r=8, dropout=0.0, scale=20.0, bias=False):
        super().__init__(input_dims, output_dims, r, dropout, scale, bias)
        self.m = mx.linalg.norm(self.linear.weight.astype(mx.float32), axis=1)
        self._dora_w_sq_norm = mx.stop_gradient(mx.sum(self.linear.weight.astype(mx.float32) * self.linear.weight.astype(mx.float32), axis=1))

    def _dequantized_weight(self) -> mx.array:
        if isinstance(self.linear, nn.QuantizedLinear):
            return mx.dequantize(
                self.linear.weight,
                self.linear.scales,
                self.linear.biases,
                self.linear.group_size,
                self.linear.bits,
            )
        return self.linear.weight

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        out = y + (self.scale * z).astype(x.dtype)

        # Compute W @ A without retaining the dequantized weight for backprop.
        # stop_gradient prevents MLX autodiff from holding the full float32
        # dequantized weight as an intermediate, saving ~8.3 GB across 114 layers.
        # lora_a still receives its gradient through the LoRA path (z above).
        WA = mx.stop_gradient(self._dequantized_weight() @ self.lora_a)

        cross = self.scale * (self.lora_b * WA.T).sum(axis=0)
        AtA = self.lora_a.T @ self.lora_a
        d_sq = (self.scale * self.scale) * mx.sum((self.lora_b.T @ AtA) * self.lora_b.T, axis=1)

        denom = mx.sqrt(self._dora_w_sq_norm + cross + d_sq)
        out = (self.m / denom).astype(x.dtype) * out
        return out

    @staticmethod
    def from_base(linear, r=8, dropout=0.0, scale=20.0):
        """Create a DoRALinear wrapping an existing Linear or QuantizedLinear.

        Args:
            linear: Base linear layer to wrap. Its weights are preserved.
            r: DoRA rank.
            dropout: Dropout probability.
            scale: Output scaling factor.

        Returns:
            DoRALinear with the base layer's weights and magnitude vector.
        """
        if isinstance(linear, nn.QuantizedLinear):
            output_dims = linear.weight.shape[0]
            input_dims = linear.weight.shape[1] * 32 // linear.bits
        else:
            output_dims, input_dims = linear.weight.shape
        dora = DoRALinear(
            input_dims,
            output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
            bias="bias" in linear,
        )
        dora.linear = linear
        w = dora._dequantized_weight()
        dora._dora_w_sq_norm = mx.stop_gradient(mx.sum(w.astype(mx.float32) * w.astype(mx.float32), axis=1))
        dora.m = mx.sqrt(dora._dora_w_sq_norm)
        return dora

    def fuse(self):
        """Fuse DoRA weights into the base layer, producing a plain nn.Linear.

        Adds the scaled low-rank delta to the base weight, then re-normalizes
        each row to match the original magnitude stored in ``m``.
        """
        weight = self._dequantized_weight()
        bias = self.linear.bias if "bias" in self.linear else None
        delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
        adapted = weight + delta
        norm_scale = self.m / mx.linalg.norm(adapted.astype(mx.float32), axis=1)
        fused_weight = norm_scale[:, None] * adapted
        fused = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        fused.weight = fused_weight.astype(weight.dtype)
        if bias is not None:
            fused.bias = bias
        return fused
