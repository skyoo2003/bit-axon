"""Tests for the Qwen → Bit-Axon initialization pipeline."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel
from bit_axon.porting.pipeline import initialize_from_qwen_weights


def make_mock_qwen_weights(config: BitAxonConfig) -> dict[str, mx.array]:
    """Create mock Qwen weights with correct shapes for testing."""
    weights: dict[str, mx.array] = {}
    source_vocab = config.vocab_size * 2
    weights["model.embed_tokens.weight"] = mx.random.normal((source_vocab, config.d_source_model))
    for i in range(config.num_layers):
        weights[f"model.layers.{i}.input_layernorm.weight"] = mx.random.normal((config.d_source_model,))
    third = config.num_layers // 3
    source_inter = 4 * config.d_source_model
    for i in range(third, config.num_layers):
        weights[f"model.layers.{i}.post_attention_layernorm.weight"] = mx.random.normal((config.d_source_model,))
        weights[f"model.layers.{i}.mlp.gate_proj.weight"] = mx.random.normal((source_inter, config.d_source_model))
        weights[f"model.layers.{i}.mlp.up_proj.weight"] = mx.random.normal((source_inter, config.d_source_model))
        weights[f"model.layers.{i}.mlp.down_proj.weight"] = mx.random.normal((config.d_source_model, source_inter))
    return weights


@pytest.fixture()
def mock_qwen(small_config: BitAxonConfig) -> dict[str, mx.array]:
    return make_mock_qwen_weights(small_config)


def test_pipeline_returns_model(mock_qwen: dict[str, mx.array], small_config: BitAxonConfig) -> None:
    model, mapping = initialize_from_qwen_weights(mock_qwen, config=small_config)
    assert isinstance(model, BitAxonModel)
    assert isinstance(mapping, dict)
    assert len(mapping) == small_config.vocab_size


def test_pipeline_forward_pass(mock_qwen: dict[str, mx.array], small_config: BitAxonConfig) -> None:
    model, _ = initialize_from_qwen_weights(mock_qwen, config=small_config)
    input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
    logits, caches = model(input_ids)
    assert logits.shape == (1, 8, small_config.vocab_size)


def test_pipeline_no_nan(mock_qwen: dict[str, mx.array], small_config: BitAxonConfig) -> None:
    model, _ = initialize_from_qwen_weights(mock_qwen, config=small_config)
    params = dict(tree_flatten(model.parameters()))
    for key, value in params.items():
        assert mx.all(mx.isfinite(value)).item(), f"Non-finite value in {key}"


def test_pipeline_weight_tying(mock_qwen: dict[str, mx.array], small_config: BitAxonConfig) -> None:
    model, _ = initialize_from_qwen_weights(mock_qwen, config=small_config)
    assert mx.array_equal(model.embed_tokens.weight, model.lm_head.weight)


def test_pipeline_key_coverage(mock_qwen: dict[str, mx.array], small_config: BitAxonConfig) -> None:
    model, _ = initialize_from_qwen_weights(mock_qwen, config=small_config)
    params = dict(tree_flatten(model.parameters()))
    important_keys = ["embed_tokens.weight", "input_proj.weight", "output_proj.weight", "lm_head.weight"]
    for key in important_keys:
        assert key in params, f"Missing key: {key}"
        assert not mx.all(params[key] == 0).item(), f"All zeros in {key}"


def test_pipeline_with_mock_qwen(small_config: BitAxonConfig) -> None:
    weights = make_mock_qwen_weights(small_config)
    model, mapping = initialize_from_qwen_weights(weights, config=small_config)
    assert isinstance(model, BitAxonModel)
    assert len(mapping) == small_config.vocab_size


def test_pipeline_vocab_mapping_identity(mock_qwen: dict[str, mx.array], small_config: BitAxonConfig) -> None:
    identity = {i: i for i in range(small_config.vocab_size)}
    model, returned = initialize_from_qwen_weights(mock_qwen, vocab_mapping=identity, config=small_config)
    assert returned == identity
    assert isinstance(model, BitAxonModel)
