import math

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self, input_dims, output_dims, r=8, dropout=0.0, scale=20.0, bias=False
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        init_scale = 1.0 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale, high=init_scale, shape=(input_dims, r)
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

    @staticmethod
    def from_base(linear, r=8, dropout=0.0, scale=20.0):
        output_dims, input_dims = linear.weight.shape
        lora = LoRALinear(
            input_dims,
            output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
            bias="bias" in linear,
        )
        lora.linear = linear
        return lora

    def fuse(self, dequantize=False):
        weight = self.linear.weight
        bias = self.linear.bias if "bias" in self.linear else None
        delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
        fused = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        fused.weight = weight + delta
        if bias is not None:
            fused.bias = bias
        return fused
