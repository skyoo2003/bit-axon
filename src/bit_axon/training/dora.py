import math

import mlx.core as mx
import mlx.nn as nn


class DoRALinear(nn.Module):
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
        linear: Base frozen linear layer.
        lora_a: Low-rank matrix A of shape (input_dims, r).
        lora_b: Low-rank matrix B of shape (r, output_dims), initialized to zeros.
        m: Magnitude vector of shape (output_dims,), frozen norm of base weight rows.
        scale: Output scaling factor.
    """

    def __init__(self, input_dims, output_dims, r=8, dropout=0.0, scale=20.0, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        init_scale = 1.0 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-init_scale, high=init_scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.m = mx.linalg.norm(self.linear.weight.astype(mx.float32), axis=1)

    def __call__(self, x):
        w = self.linear.weight
        y = x @ w.T
        if "bias" in self.linear:
            y = y + self.linear.bias
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        out = y + (self.scale * z).astype(x.dtype)
        adapted = w + (self.scale * self.lora_b.T) @ self.lora_a.T
        denom = mx.stop_gradient(mx.linalg.norm(adapted.astype(mx.float32), axis=1))
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
        dora.m = mx.linalg.norm(linear.weight.astype(mx.float32), axis=1)
        return dora

    def fuse(self):
        """Fuse DoRA weights into the base layer, producing a plain nn.Linear.

        Adds the scaled low-rank delta to the base weight, then re-normalizes
        each row to match the original magnitude stored in ``m``.

        Returns:
            nn.Linear with fused and magnitude-normalized weights.
        """
        weight = self.linear.weight
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
