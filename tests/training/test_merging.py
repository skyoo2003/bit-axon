"""Tests for adapter merging and model export utilities."""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel
from bit_axon.training.dora import DoRALinear
from bit_axon.training.lora import LoRALinear, apply_lora_to_model
from bit_axon.training.merging import (
    dequantize_model,
    load_and_merge,
    merge_adapters,
    quantize_model,
    save_merged_model,
)


class SimpleModel(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.relu(self.linear1(x)))


def _small_config():
    return BitAxonConfig(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        d_source_model=128,
        vocab_size=1024,
        ssm_d_state=4,
        ssm_d_conv=2,
        ssm_expand=2,
        swa_window_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_dim=512,
    )


class TestMergeAdapters:
    def test_lora_fusion(self):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        model.linear1 = LoRALinear.from_base(model.linear1, r=4, dropout=0.0)
        mx.eval(model.parameters())
        assert isinstance(model.linear1, LoRALinear)
        merge_adapters(model)
        assert not isinstance(model.linear1, LoRALinear)
        assert isinstance(model.linear1, nn.Linear)

    def test_dora_fusion(self):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        model.linear1 = DoRALinear.from_base(model.linear1, r=4, dropout=0.0)
        mx.eval(model.parameters())
        assert isinstance(model.linear1, DoRALinear)
        merge_adapters(model)
        assert not isinstance(model.linear1, DoRALinear)
        assert isinstance(model.linear1, nn.Linear)

    def test_no_adapters(self):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        w_before = mx.array(model.linear1.weight)
        merge_adapters(model)
        diff = mx.abs(model.linear1.weight - w_before).max()
        assert float(diff) < 1e-6

    def test_output_approximation(self):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        model.linear1 = LoRALinear.from_base(model.linear1, r=4, dropout=0.0, scale=5.0)
        mx.eval(model.parameters())
        x = mx.random.normal(shape=(2, 4, 16))
        out_before = model(x)
        mx.eval(out_before)
        merge_adapters(model)
        out_after = model(x)
        mx.eval(out_after)
        diff = mx.abs(out_before - out_after).max()
        assert float(diff) < 1e-2

    def test_small_config(self, small_config):
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        wrapped = apply_lora_to_model(model, rank=4, dropout=0.0)
        assert len(wrapped) > 0
        for path in wrapped:
            parts = path.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            assert isinstance(obj, (LoRALinear, DoRALinear))
        merge_adapters(model)
        for path in wrapped:
            parts = path.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            assert isinstance(obj, nn.Linear)


class TestDequantizeModel:
    def test_quantized_to_linear(self):
        linear = nn.Linear(64, 32)
        ql = nn.QuantizedLinear.from_linear(linear, bits=4, group_size=64)
        model = SimpleModel(dim=64)
        model.linear1 = ql
        mx.eval(model.parameters())
        dequantize_model(model)
        assert isinstance(model.linear1, nn.Linear)
        assert not isinstance(model.linear1, nn.QuantizedLinear)
        assert model.linear1.weight.shape == (32, 64)

    def test_no_quantized(self):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        w_before = mx.array(model.linear1.weight)
        dequantize_model(model)
        diff = mx.abs(model.linear1.weight - w_before).max()
        assert float(diff) < 1e-6

    def test_preserves_bias(self):
        linear = nn.Linear(64, 32, bias=True)
        ql = nn.QuantizedLinear.from_linear(linear, bits=4, group_size=64)
        container = nn.Module()
        container.ql = ql
        mx.eval(container.parameters())
        dequantize_model(container)
        assert isinstance(container.ql, nn.Linear)
        assert "bias" in container.ql
        assert container.ql.bias is not None


class TestQuantizeModel:
    def test_linear_to_quantized(self):
        model = SimpleModel(dim=64)
        mx.eval(model.parameters())
        quantize_model(model, bits=4, group_size=64)
        assert isinstance(model.linear1, nn.QuantizedLinear)
        assert isinstance(model.linear2, nn.QuantizedLinear)

    def test_already_quantized(self):
        linear = nn.Linear(64, 32)
        ql = nn.QuantizedLinear.from_linear(linear, bits=4, group_size=64)
        model = SimpleModel(dim=64)
        model.linear1 = ql
        mx.eval(model.parameters())
        quantize_model(model, bits=4, group_size=64)
        assert isinstance(model.linear1, nn.QuantizedLinear)

    def test_small_linear_skipped(self):
        model = SimpleModel(dim=8)
        mx.eval(model.parameters())
        quantize_model(model, bits=4, group_size=64)
        assert isinstance(model.linear1, nn.Linear)
        assert not isinstance(model.linear1, nn.QuantizedLinear)


class TestSaveMergedModel:
    def test_saves_safetensors(self, tmp_path):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        out = save_merged_model(model, tmp_path / "merged")
        assert (tmp_path / "merged" / "weights.safetensors").exists()
        assert out == tmp_path / "merged"

    def test_loadable(self, tmp_path):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        save_merged_model(model, tmp_path / "merged")
        loaded = mx.load(str(tmp_path / "merged" / "weights.safetensors"))
        assert len(loaded) > 0

    def test_saves_config(self, tmp_path):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        config = BitAxonConfig(hidden_dim=256, num_layers=4)
        save_merged_model(model, tmp_path / "merged", config=config)
        config_path = tmp_path / "merged" / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            saved_config = json.load(f)
        assert saved_config["hidden_dim"] == 256
        assert saved_config["num_layers"] == 4

    def test_creates_output_dir(self, tmp_path):
        model = SimpleModel(dim=16)
        mx.eval(model.parameters())
        deep_path = tmp_path / "a" / "b" / "c"
        save_merged_model(model, deep_path)
        assert (deep_path / "weights.safetensors").exists()


class TestLoadAndMerge:
    def test_end_to_end(self, tmp_path):
        config = _small_config()
        model = BitAxonModel(config)
        mx.eval(model.parameters())

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        mx.save_safetensors(str(base_dir / "weights.safetensors"), dict(tree_flatten(model.parameters())))

        apply_lora_to_model(model, rank=4, dropout=0.0)
        mx.eval(model.parameters())
        adapter_path = tmp_path / "adapters.safetensors"
        mx.save_safetensors(str(adapter_path), dict(tree_flatten(model.parameters())))

        output_dir = load_and_merge(base_dir, adapter_path, tmp_path / "output", config=config, quantize_after_merge=False)
        assert (output_dir / "weights.safetensors").exists()
        loaded = mx.load(str(output_dir / "weights.safetensors"))
        assert len(loaded) > 0

    def test_end_to_end_with_quantize(self, tmp_path):
        config = _small_config()
        model = BitAxonModel(config)
        mx.eval(model.parameters())

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        mx.save_safetensors(str(base_dir / "weights.safetensors"), dict(tree_flatten(model.parameters())))

        apply_lora_to_model(model, rank=4, dropout=0.0)
        mx.eval(model.parameters())
        adapter_path = tmp_path / "adapters.safetensors"
        mx.save_safetensors(str(adapter_path), dict(tree_flatten(model.parameters())))

        output_dir = load_and_merge(base_dir, adapter_path, tmp_path / "output", config=config, quantize_after_merge=True, bits=4, group_size=64)
        assert (output_dir / "weights.safetensors").exists()

    def test_returns_path(self, tmp_path):
        config = _small_config()
        model = BitAxonModel(config)
        mx.eval(model.parameters())

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        mx.save_safetensors(str(base_dir / "weights.safetensors"), dict(tree_flatten(model.parameters())))

        apply_lora_to_model(model, rank=4, dropout=0.0)
        mx.eval(model.parameters())
        adapter_path = tmp_path / "adapters.safetensors"
        mx.save_safetensors(str(adapter_path), dict(tree_flatten(model.parameters())))

        result = load_and_merge(base_dir, adapter_path, tmp_path / "output", config=config)
        assert result == tmp_path / "output"
        assert result.is_dir()
