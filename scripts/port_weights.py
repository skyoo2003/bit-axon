#!/usr/bin/env python3
"""Download Qwen2.5-3B and initialize Bit-Axon model.

Usage:
    python scripts/port_weights.py --output ./bit-axon-ported
    python scripts/port_weights.py --output ./bit-axon-ported --config-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx


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
    parser = argparse.ArgumentParser(description="Initialize Bit-Axon from Qwen2.5-3B")
    parser.add_argument("--output", required=True, help="Output directory for ported weights")
    parser.add_argument("--config-small", action="store_true", help="Use small config with mock weights (no download)")
    args = parser.parse_args()

    if args.config_small:
        from bit_axon.config import BitAxonConfig
        from bit_axon.porting.pipeline import initialize_from_qwen_weights, save_ported_model

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
        print("Creating mock Qwen weights (no download needed)")

        weights = _make_mock_qwen(config)
        model, vocab_mapping = initialize_from_qwen_weights(weights, config=config)

        output_path = args.output
        if not output_path.endswith(".safetensors"):
            Path(output_path).mkdir(parents=True, exist_ok=True)
            output_path = f"{output_path}/model.safetensors"

        save_ported_model(model, output_path, vocab_mapping)
        print(f"Model saved to {output_path}")
        print(f"Vocab mapping size: {len(vocab_mapping)}")
    else:
        print("Full pipeline: downloading Qwen2.5-3B...")
        print("Requires: pip install -e '.[porting]'")
        print("TODO: implement full Qwen2.5-3B download and weight porting")
        sys.exit(1)


if __name__ == "__main__":
    main()
