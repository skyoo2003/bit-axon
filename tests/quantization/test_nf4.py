import mlx.core as mx
import mlx.nn as nn
import pytest

from bit_axon.quantization.nf4 import (
    quantize_nf4,
    dequantize_nf4,
    replace_linear_with_quantized,
)


class TestNF4:
    def test_quantize_dequantize_roundtrip(self):
        mx.random.seed(42)
        w = mx.random.normal(shape=(128, 64))
        packed, scales, biases = quantize_nf4(w, group_size=64)
        recovered = dequantize_nf4(packed, scales, biases, group_size=64, bits=4)
        diff = mx.abs(w - recovered).mean()
        assert float(diff) < 0.15

    def test_quantized_linear_from_linear(self):
        linear = nn.Linear(64, 32)
        qlinear = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
        assert isinstance(qlinear, nn.QuantizedLinear)

    def test_replace_linear(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        replace_linear_with_quantized(model, group_size=64, bits=4)
        assert isinstance(model.layers[0], nn.QuantizedLinear)
        assert isinstance(model.layers[2], nn.QuantizedLinear)
