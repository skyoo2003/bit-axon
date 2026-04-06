#!/usr/bin/env python3
"""Evaluate perplexity on WikiText-103.

Usage:
    python scripts/evaluate_ppl.py --model-path ./bit-axon-ported
    python scripts/evaluate_ppl.py --model-path ./bit-axon-ported --config-small
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings

import mlx.core as mx
import mlx.nn as nn

warnings.warn(
    "This script is deprecated. Use the `bit-axon` CLI instead. Run `pip install -e .` and then `bit-axon --help` for available commands.",
    DeprecationWarning,
    stacklevel=2,
)


def _compute_loss(model, token_ids):
    logits, _ = model(token_ids[:, :-1])
    logits = logits.astype(mx.float32)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = token_ids[:, 1:].reshape(-1)
    losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
    mean_loss = mx.mean(losses).item()
    return mean_loss


def _make_mock_qwen(config) -> dict[str, mx.array]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--model-path", required=True, help="Path to ported model weights")
    parser.add_argument("--config-small", action="store_true", help="Use small config with mock model (no download)")
    args = parser.parse_args()

    if args.config_small:
        from bit_axon.config import BitAxonConfig
        from bit_axon.porting.pipeline import initialize_from_qwen_weights

        config = BitAxonConfig(
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
        print(f"Using small config: hidden_dim={config.hidden_dim}, num_layers={config.num_layers}")
        print("Creating mock model from random weights")

        weights = _make_mock_qwen(config)
        model, _ = initialize_from_qwen_weights(weights, config=config)

        token_ids = mx.random.randint(0, config.vocab_size, shape=(2, 64), dtype=mx.uint32)
        print(f"Computing perplexity on {token_ids.shape} random tokens...")
        mean_loss = _compute_loss(model, token_ids)

        if mean_loss > 709:
            print(f"Mean cross-entropy loss: {mean_loss:.4f}")
            print("Perplexity: inf (loss too large for exp — expected with random/untrained weights)")
        else:
            ppl = math.exp(mean_loss)
            print(f"Perplexity: {ppl:.2f} (mean loss: {mean_loss:.4f})")

        if not math.isfinite(mean_loss):
            print(f"WARNING: Mean loss is not finite ({mean_loss})", file=sys.stderr)
            sys.exit(1)
    else:
        print("Full evaluation: loading model and WikiText-103...")
        print("Requires: pip install -e '.[porting]' and datasets")
        print("TODO: implement full WikiText-103 perplexity evaluation")
        sys.exit(1)


if __name__ == "__main__":
    main()
