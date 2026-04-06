"""Tests for the inference model loader."""

from __future__ import annotations

import json
from dataclasses import asdict

import mlx.core as mx
from mlx.utils import tree_flatten

from bit_axon.config import BitAxonConfig
from bit_axon.inference.loader import load_model
from bit_axon.model import BitAxonModel

SMALL_CONFIG = BitAxonConfig(
    hidden_dim=256,
    num_layers=4,
    num_heads=4,
    d_source_model=128,
    vocab_size=1024,
)


def _save_weights(tmp_path, model, config=None):
    params = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "weights.safetensors"), params)
    if config is not None:
        with open(tmp_path / "config.json", "w") as f:
            json.dump(asdict(config), f)
    return tmp_path


class TestLoadModel:
    def test_load_model_small_config(self):
        model = BitAxonModel(SMALL_CONFIG)
        mx.eval(model.parameters())
        assert isinstance(model, BitAxonModel)
        assert len(tree_flatten(model.parameters())) > 0

    def test_load_model_quantize(self):
        from bit_axon.quantization.nf4 import replace_linear_with_quantized

        model = BitAxonModel(SMALL_CONFIG)
        mx.eval(model.parameters())
        replace_linear_with_quantized(model, group_size=64, bits=4)
        assert isinstance(model, BitAxonModel)

    def test_load_model_from_safetensors(self, tmp_path):
        original = BitAxonModel(SMALL_CONFIG)
        mx.eval(original.parameters())
        _save_weights(tmp_path, original, config=SMALL_CONFIG)

        loaded = load_model(tmp_path, config=SMALL_CONFIG)

        orig_flat = dict(tree_flatten(original.parameters()))
        loaded_flat = dict(tree_flatten(loaded.parameters()))

        for key in orig_flat:
            assert key in loaded_flat, f"Missing key: {key}"
            assert orig_flat[key].shape == loaded_flat[key].shape, f"Shape mismatch for {key}: {orig_flat[key].shape} vs {loaded_flat[key].shape}"

    def test_load_model_returns_correct_type(self, tmp_path):
        original = BitAxonModel(SMALL_CONFIG)
        mx.eval(original.parameters())
        _save_weights(tmp_path, original)

        model = load_model(tmp_path, config=SMALL_CONFIG)
        assert isinstance(model, BitAxonModel)

    def test_load_model_weights_applied(self, tmp_path):
        original = BitAxonModel(SMALL_CONFIG)
        mx.eval(original.parameters())
        _save_weights(tmp_path, original, config=SMALL_CONFIG)

        loaded = load_model(tmp_path, config=SMALL_CONFIG)
        mx.eval(loaded.parameters())

        params = tree_flatten(loaded.parameters())
        has_nonzero = any(mx.any(mx.abs(val) > 0) for _, val in params)
        assert has_nonzero, "All model parameters are zero after loading"

    def test_load_model_with_config_json(self, tmp_path):
        original = BitAxonModel(SMALL_CONFIG)
        mx.eval(original.parameters())
        _save_weights(tmp_path, original, config=SMALL_CONFIG)

        model = load_model(tmp_path, config=None)
        assert isinstance(model, BitAxonModel)

    def test_load_model_no_safetensors_files(self, tmp_path):
        model = load_model(tmp_path, config=SMALL_CONFIG)
        assert isinstance(model, BitAxonModel)

    def test_load_model_quantize_flag(self, tmp_path):
        original = BitAxonModel(SMALL_CONFIG)
        mx.eval(original.parameters())
        _save_weights(tmp_path, original, config=SMALL_CONFIG)

        model = load_model(tmp_path, config=SMALL_CONFIG, quantize=True)
        assert isinstance(model, BitAxonModel)

        flat = tree_flatten(model.parameters())
        param_names = [name for name, _ in flat]
        has_quantized = any("scales" in n for n in param_names)
        assert has_quantized, "Quantization was not applied"
