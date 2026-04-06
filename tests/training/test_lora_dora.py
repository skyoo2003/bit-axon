import mlx.core as mx
import mlx.nn as nn

from bit_axon.model import BitAxonModel
from bit_axon.training.dora import DoRALinear
from bit_axon.training.lora import LoRALinear, apply_lora_to_model


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

        a_grad, b_grad, m_grad = mx.grad(loss_fn, argnums=[0, 1, 2])(dora.lora_a, dora.lora_b, dora.m)
        assert a_grad is not None and b_grad is not None and m_grad is not None


class TestQLoRASupport:
    def test_lora_from_base_quantized_linear(self):
        """LoRALinear.from_base should handle nn.QuantizedLinear."""
        linear = nn.Linear(64, 32)
        qlinear = nn.QuantizedLinear.from_linear(linear, bits=4, group_size=64)
        lora = LoRALinear.from_base(qlinear, r=4)
        assert lora.lora_a.shape == (64, 4)
        assert lora.lora_b.shape == (4, 32)

    def test_dora_from_base_quantized_linear(self):
        """DoRALinear.from_base should handle nn.QuantizedLinear."""
        linear = nn.Linear(64, 32)
        qlinear = nn.QuantizedLinear.from_linear(linear, bits=4, group_size=64)
        dora = DoRALinear.from_base(qlinear, r=4)
        assert dora.lora_a.shape == (64, 4)
        assert dora.lora_b.shape == (4, 32)

    def test_lora_quantized_forward_shape(self):
        """LoRA on QuantizedLinear should produce correct output shape."""
        linear = nn.Linear(64, 32)
        qlinear = nn.QuantizedLinear.from_linear(linear, bits=4, group_size=64)
        lora = LoRALinear.from_base(qlinear, r=4, dropout=0.0)
        x = mx.random.normal(shape=(2, 8, 64))
        out = lora(x)
        assert out.shape == (2, 8, 32)


class TestApplyLoraToModel:
    def test_apply_lora_to_small_model(self, small_config):
        """apply_lora_to_model should wrap target layers in small model."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        wrapped = apply_lora_to_model(model, rank=4)
        assert len(wrapped) > 0
        for path in wrapped:
            parts = path.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            assert isinstance(obj, (LoRALinear, DoRALinear))

    def test_apply_lora_excludes_lm_head(self, small_config):
        """lm_head should NOT be wrapped (weight-tied with embed_tokens)."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        wrapped = apply_lora_to_model(model, rank=4)
        assert not any("lm_head" in p for p in wrapped)
        assert isinstance(model.lm_head, nn.Linear)

    def test_apply_lora_excludes_router_gates(self, small_config):
        """Router gates (gate, shared_expert_gate) should NOT be wrapped."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        wrapped = apply_lora_to_model(model, rank=4)
        assert not any(p.endswith(".gate") for p in wrapped)
        assert not any("shared_expert_gate" in p for p in wrapped)

    def test_apply_lora_excludes_small_projections(self, small_config):
        """x_proj and dt_proj should NOT be wrapped (too small)."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        wrapped = apply_lora_to_model(model, rank=4)
        assert not any(p.endswith("x_proj") for p in wrapped)
        assert not any(p.endswith("dt_proj") for p in wrapped)

    def test_apply_dora_to_model(self, small_config):
        """apply_lora_to_model with use_dora=True should wrap with DoRALinear."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        wrapped = apply_lora_to_model(model, rank=4, use_dora=True)
        assert len(wrapped) > 0
        for path in wrapped:
            parts = path.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            assert isinstance(obj, DoRALinear)

    def test_apply_lora_preserves_forward(self, small_config):
        """After LoRA with zero-init B, forward output should match base model."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
        logits_before, _ = model(input_ids)
        mx.eval(logits_before)

        apply_lora_to_model(model, rank=4, dropout=0.0)
        logits_after, _ = model(input_ids)
        mx.eval(logits_after)

        diff = mx.abs(logits_before - logits_after).max()
        assert float(diff) < 1e-3

    def test_apply_lora_returns_path_list(self, small_config):
        """apply_lora_to_model should return list of wrapped layer paths."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        wrapped = apply_lora_to_model(model, rank=4)
        assert isinstance(wrapped, list)
        assert all(isinstance(p, str) for p in wrapped)
