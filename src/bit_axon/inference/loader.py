"""Model loading utilities for Bit-Axon inference."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel


def load_model(
    weights_path: str | Path,
    config: BitAxonConfig | None = None,
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
) -> BitAxonModel:
    """Load a BitAxonModel from disk with optional NF4 quantization.

    Loads weights from safetensors files in weights_path. If no config is
    provided, attempts to read config.json from the same directory.

    Args:
        weights_path: Directory containing safetensors weight files.
        config: Model configuration. Falls back to config.json or defaults.
        quantize: If True, replace Linear layers with QuantizedLinear.
        bits: Quantization bit width (default 4 for NF4).
        group_size: Quantization group size.

    Returns:
        Loaded BitAxonModel with weights applied.
    """
    weights_path = Path(weights_path)

    if config is None:
        config_dir = weights_path.parent if weights_path.is_file() else weights_path
        config_path = config_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = BitAxonConfig(**config_dict)
        else:
            config = BitAxonConfig()

    model = BitAxonModel(config)

    weights: dict[str, mx.array] = {}
    if weights_path.is_file():
        weights.update(mx.load(str(weights_path)))
    else:
        for sf_file in sorted(weights_path.glob("*.safetensors")):
            weights.update(mx.load(str(sf_file)))

    if not weights:
        msg = f"No safetensors weight files found in '{weights_path}'. Model has random (untrained) weights."
        warnings.warn(msg, stacklevel=2)
    else:
        # Auto-detect quantized weights (NF4: packed .weight + .scales + .biases)
        has_quantized_keys = any(".scales" in k or ".biases" in k for k in weights)
        if has_quantized_keys and not quantize:
            from bit_axon.quantization.nf4 import replace_linear_with_quantized, replace_switch_linear_with_quantized

            replace_linear_with_quantized(model, group_size=group_size, bits=bits)
            replace_switch_linear_with_quantized(model, group_size=group_size, bits=bits)
            warnings.warn(
                "Detected quantized weights — automatically enabled NF4 quantization.",
                stacklevel=2,
            )
        elif quantize and not has_quantized_keys:
            from bit_axon.quantization.nf4 import replace_linear_with_quantized, replace_switch_linear_with_quantized

            replace_linear_with_quantized(model, group_size=group_size, bits=bits)
            replace_switch_linear_with_quantized(model, group_size=group_size, bits=bits)

        model_params = dict(tree_flatten(model.parameters()))
        model_keys = set(model_params)
        matched = {k: v for k, v in weights.items() if k in model_keys and v.shape == model_params[k].shape}
        shape_mismatch = sorted(k for k in weights if k in model_keys and k not in matched)
        skipped = sorted(set(weights) - model_keys)
        if shape_mismatch:
            examples = shape_mismatch[:3]
            details = ", ".join(f"{k} (weight={weights[k].shape}, model={model_params[k].shape})" for k in examples)
            warnings.warn(
                f"Skipped {len(shape_mismatch)} weight keys with shape mismatches: {details}. "
                "This can happen when quantized weights don't fully match the model architecture.",
                stacklevel=2,
            )
        if skipped:
            warnings.warn(
                f"Skipped {len(skipped)} weight keys not found in model (e.g. {skipped[:3]}). Check that the weights match the model architecture.",
                stacklevel=2,
            )
        if matched:
            model.update(tree_unflatten(list(matched.items())))
        else:
            warnings.warn(
                f"None of the {len(weights)} weight keys matched the model. Model has random (untrained) weights.",
                stacklevel=2,
            )

    mx.eval(model.parameters())

    return model


def resize_model_vocab(model: BitAxonModel, target_vocab_size: int) -> None:
    """Resize model embedding and lm_head to match a tokenizer's vocab size.

    If target_vocab_size > current vocab_size, extends the embedding table with
    zero-initialized rows. If target_vocab_size <= current, does nothing.

    When weight_tying is enabled (default), lm_head shares the same weight tensor
    as embed_tokens, so resizing embed_tokens automatically handles lm_head.

    Args:
        model: BitAxonModel instance.
        target_vocab_size: Desired vocabulary size (typically from tokenizer).
    """
    current = model.config.vocab_size
    if target_vocab_size <= current:
        return

    old_weight = model.embed_tokens.weight
    d_source = old_weight.shape[1]
    delta = target_vocab_size - current
    new_rows = mx.zeros((delta, d_source), dtype=old_weight.dtype)
    new_weight = mx.concatenate([old_weight, new_rows], axis=0)
    model.embed_tokens.weight = new_weight

    # If weight_tying is enabled, lm_head already shares the same tensor ref,
    # so it's automatically updated. Handle the non-tied case too.
    if not model.config.weight_tying:
        old_lm = model.lm_head.weight
        lm_new_rows = mx.zeros((delta, d_source), dtype=old_lm.dtype)
        model.lm_head.weight = mx.concatenate([old_lm, lm_new_rows], axis=0)

    model.config.vocab_size = target_vocab_size
    mx.eval(model.parameters())

    warnings.warn(
        f"Resized model vocab from {current} to {target_vocab_size} ({delta} new zero-initialized rows). This is expected when using a Qwen2.5 tokenizer.",
        stacklevel=2,
    )
