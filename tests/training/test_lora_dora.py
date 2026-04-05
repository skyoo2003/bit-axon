import mlx.core as mx
import mlx.nn as nn
import pytest

from bit_axon.training.lora import LoRALinear
from bit_axon.training.dora import DoRALinear


class TestLoRALinear:
    def test_lora_output_shape(self):
        input_dims, output_dims, batch, seq = 64, 32, 2, 8
        lora = LoRALinear(input_dims, output_dims, r=4)
        x = mx.random.normal(shape=(batch, seq, input_dims))
        out = lora(x)
        assert out.shape == (batch, seq, output_dims)

    def test_lora_zero_init(self):
        input_dims, output_dims = 64, 32
        lora = LoRALinear(input_dims, output_dims, r=4, dropout=0.0)
        x = mx.random.normal(shape=(1, 4, input_dims))
        lora_out = lora(x)
        base_out = lora.linear(x)
        diff = mx.abs(lora_out - base_out).max()
        assert float(diff) < 1e-5

    def test_lora_from_base(self):
        linear = nn.Linear(64, 32)
        lora = LoRALinear.from_base(linear, r=8, scale=10.0)
        assert isinstance(lora, LoRALinear)
        assert lora.lora_a.shape == (64, 8)
        assert lora.lora_b.shape == (8, 32)
        assert lora.linear is linear

    def test_lora_fuse(self):
        input_dims, output_dims, r = 64, 32, 4
        lora = LoRALinear(input_dims, output_dims, r=r, scale=5.0, dropout=0.0)
        x = mx.random.normal(shape=(2, 8, input_dims))
        lora_out = lora(x)
        fused = lora.fuse()
        fused_out = fused(x)
        diff = mx.abs(lora_out - fused_out).max()
        assert float(diff) < 1e-4


class TestDoRALinear:
    def test_dora_m_shape(self):
        input_dims, output_dims = 64, 32
        dora = DoRALinear(input_dims, output_dims, r=4)
        assert dora.m.shape == (output_dims,)

    def test_dora_m_values(self):
        input_dims, output_dims = 64, 32
        dora = DoRALinear(input_dims, output_dims, r=4)
        expected = mx.linalg.norm(dora.linear.weight.astype(mx.float32), axis=1)
        diff = mx.abs(dora.m - expected).max()
        assert float(diff) < 1e-5

    def test_dora_from_base(self):
        linear = nn.Linear(64, 32)
        dora = DoRALinear.from_base(linear, r=8, scale=10.0)
        assert isinstance(dora, DoRALinear)
        assert dora.lora_a.shape == (64, 8)
        assert dora.lora_b.shape == (8, 32)
        assert dora.linear is linear
        expected_m = mx.linalg.norm(linear.weight.astype(mx.float32), axis=1)
        diff = mx.abs(dora.m - expected_m).max()
        assert float(diff) < 1e-5

    def test_dora_fuse(self):
        input_dims, output_dims, r = 64, 32, 4
        dora = DoRALinear(input_dims, output_dims, r=r, scale=5.0, dropout=0.0)
        x = mx.random.normal(shape=(2, 8, input_dims))
        dora_out = dora(x)
        fused = dora.fuse()
        fused_out = fused(x)
        diff = mx.abs(dora_out - fused_out).max()
        assert float(diff) < 1e-3

    def test_dora_gradient_flow(self):
        input_dims, output_dims, r = 16, 8, 2
        dora = DoRALinear(input_dims, output_dims, r=r, dropout=0.0, scale=5.0)

        def loss_fn(a, b, m):
            dora.lora_a = a
            dora.lora_b = b
            dora.m = m
            x = mx.random.normal(shape=(2, 4, input_dims))
            out = dora(x)
            return out.sum()

        a_grad, b_grad, m_grad = mx.grad(loss_fn, argnums=[0, 1, 2])(
            dora.lora_a, dora.lora_b, dora.m
        )
        assert a_grad is not None and b_grad is not None and m_grad is not None
