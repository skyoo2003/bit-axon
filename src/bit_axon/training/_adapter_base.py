from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class _BaseAdapterLinear(nn.Module):
    def __init__(self, input_dims, output_dims, r=8, dropout=0.0, scale=20.0, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        init_scale = 1.0 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-init_scale, high=init_scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))

    @staticmethod
    def from_base(linear, r=8, dropout=0.0, scale=20.0):
        if isinstance(linear, nn.QuantizedLinear):
            output_dims = linear.weight.shape[0]
            input_dims = linear.weight.shape[1] * 32 // linear.bits
        else:
            output_dims, input_dims = linear.weight.shape
        adapter = _BaseAdapterLinear(
            input_dims,
            output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
            bias="bias" in linear,
        )
        adapter.linear = linear
        return adapter
