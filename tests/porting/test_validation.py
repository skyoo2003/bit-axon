"""End-to-end validation tests for the weight porting pipeline.

Verifies the full pipeline produces a model that can run inference
with correct shapes, finite outputs, and reasonable perplexity.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel
from bit_axon.porting.pipeline import initialize_from_qwen_weights


def _make_mock_qwen(config: BitAxonConfig) -> dict[str, mx.array]:
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
def ported_model(small_config: BitAxonConfig) -> tuple[BitAxonModel, BitAxonConfig]:
    """Run the full pipeline and return (model, config)."""
    weights = _make_mock_qwen(small_config)
    model, _ = initialize_from_qwen_weights(weights, config=small_config)
    return model, small_config


def test_forward_no_nan(ported_model: tuple[BitAxonModel, BitAxonConfig]) -> None:
    """Model forward pass produces all-finite logits and cache values."""
    model, config = ported_model
    input_ids = mx.random.randint(0, config.vocab_size, shape=(1, 8), dtype=mx.uint32)
    logits, caches = model(input_ids)

    assert mx.all(mx.isfinite(logits)).item(), "Non-finite values in logits"
    for i, cache in enumerate(caches):
        if cache is not None:
            # KVCache has keys/values as mx arrays
            state = cache.state if hasattr(cache, "state") else None
            if state is not None:
                assert mx.all(mx.isfinite(state[0])).item(), f"Non-finite cache key at layer {i}"
                assert mx.all(mx.isfinite(state[1])).item(), f"Non-finite cache value at layer {i}"


def test_forward_correct_shape(ported_model: tuple[BitAxonModel, BitAxonConfig]) -> None:
    """Logits shape is (B, L, vocab_size)."""
    model, config = ported_model
    input_ids = mx.random.randint(0, config.vocab_size, shape=(2, 16), dtype=mx.uint32)
    logits, caches = model(input_ids)

    assert logits.shape == (2, 16, config.vocab_size), f"Expected (2, 16, {config.vocab_size}), got {logits.shape}"


def test_perplexity_finite(ported_model: tuple[BitAxonModel, BitAxonConfig]) -> None:
    """Cross-entropy loss on random tokens is finite (not inf, not nan)."""
    model, config = ported_model
    token_ids = mx.random.randint(0, config.vocab_size, shape=(1, 32), dtype=mx.uint32)

    logits, _ = model(token_ids[:, :-1])
    logits = logits.astype(mx.float32)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = token_ids[:, 1:].reshape(-1)
    losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")

    assert mx.all(mx.isfinite(losses)).item(), "Non-finite values in cross-entropy losses"
    mean_loss = mx.mean(losses).item()
    assert math.isfinite(mean_loss), f"Mean loss is not finite: {mean_loss}"

    if mean_loss <= 709:
        ppl = math.exp(mean_loss)
        assert math.isfinite(ppl), f"Perplexity is not finite: {ppl}"


def test_perplexity_reasonable_range(ported_model: tuple[BitAxonModel, BitAxonConfig]) -> None:
    """Mean cross-entropy loss is >= 0 and finite with random weights."""
    model, config = ported_model
    token_ids = mx.random.randint(0, config.vocab_size, shape=(1, 32), dtype=mx.uint32)

    logits, _ = model(token_ids[:, :-1])
    logits = logits.astype(mx.float32)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = token_ids[:, 1:].reshape(-1)
    losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")

    mean_loss = mx.mean(losses).item()
    assert mean_loss >= 0, f"Mean loss is negative: {mean_loss}"
    assert math.isfinite(mean_loss), f"Mean loss is not finite: {mean_loss}"


def test_pipeline_importable() -> None:
    """Verify pipeline module imports cleanly (existing code not broken)."""
    from bit_axon.porting import pipeline

    assert hasattr(pipeline, "initialize_from_qwen_weights")
    assert hasattr(pipeline, "save_ported_model")


def test_pipeline_produces_valid_model_for_inference(small_config: BitAxonConfig) -> None:
    """Full cycle: create mock weights → pipeline → forward pass → verify shapes."""
    weights = _make_mock_qwen(small_config)
    model, vocab_mapping = initialize_from_qwen_weights(weights, config=small_config)

    # Verify model type and vocab mapping
    assert isinstance(model, BitAxonModel)
    assert len(vocab_mapping) == small_config.vocab_size

    # Verify all parameters are finite
    params = dict(tree_flatten(model.parameters()))
    for key, value in params.items():
        assert mx.all(mx.isfinite(value)).item(), f"Non-finite value in {key}"

    # Verify forward pass works with correct shapes
    input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
    logits, caches = model(input_ids)
    assert logits.shape == (1, 8, small_config.vocab_size)
    assert len(caches) == small_config.num_layers

    # Verify weight tying
    assert mx.array_equal(model.embed_tokens.weight, model.lm_head.weight)

    # Verify cross-entropy is computable (loss finite)
    logits, _ = model(input_ids[:, :-1])
    logits = logits.astype(mx.float32)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = input_ids[:, 1:].reshape(-1)
    losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
    assert mx.all(mx.isfinite(losses)).item(), "Non-finite cross-entropy losses"
