"""Port Qwen2.5-3B weights to Bit-Axon format."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import typer
from rich.console import Console

from bit_axon.cli._console import print_error, print_info, print_success

console = Console()


def port_weights_cmd(output: str, config_small: bool, config_medium: bool = False) -> None:
    """Port Qwen2.5-3B weights to Bit-Axon model format."""
    from bit_axon.config import BitAxonConfig
    from bit_axon.porting.pipeline import initialize_from_qwen_weights, save_ported_model

    if config_small:
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
        print_info(f"Small config: hidden_dim={config.hidden_dim}, layers={config.num_layers}")

        with console.status("[bold green]Creating mock Qwen weights..."):
            weights = _make_mock_qwen(config)

        with console.status("[bold green]Initializing model from mock weights..."):
            model, vocab_mapping = initialize_from_qwen_weights(weights, config=config)

        print_success(f"Model initialized (vocab mapping: {len(vocab_mapping)} tokens)")
    else:
        config = BitAxonConfig()
        print_info(f"Full config: hidden_dim={config.hidden_dim}, layers={config.num_layers}, vocab_size={config.vocab_size}")

        with console.status("[bold green]Downloading Qwen2.5-3B from HuggingFace Hub..."):
            try:
                from huggingface_hub import snapshot_download

                qwen_dir = snapshot_download("Qwen/Qwen2.5-3B")
                print_success(f"Downloaded to: {qwen_dir}")
            except Exception as e:
                print_error(f"Download failed: {e}")
                print_info("Make sure you have: pip install -e '.[porting]'")
                raise typer.Exit(1) from None

        with console.status("[bold green]Loading Qwen2.5-3B weights..."):
            import glob

            weight_files = sorted(glob.glob(f"{qwen_dir}/*.safetensors"))
            if not weight_files:
                print_error(f"No safetensors files found in {qwen_dir}")
                raise typer.Exit(1)
            weights = {}
            for f in weight_files:
                w = mx.load(f)
                weights.update(w)
            print_success(f"Loaded {len(weight_files)} weight files")

        with console.status("[bold green]Building vocab mapping (152K -> 32K)..."):
            from bit_axon.porting.vocab_map import build_vocab_mapping

            vocab_mapping = build_vocab_mapping(target_size=config.vocab_size)
        print_success(f"Vocab mapping: {len(vocab_mapping)} tokens")

        with console.status("[bold green]Initializing Bit-Axon model from Qwen weights..."):
            model, vocab_mapping = initialize_from_qwen_weights(weights, vocab_mapping=vocab_mapping, config=config)
        print_success("Model initialized")

    output_path = output
    if not output_path.endswith(".safetensors"):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_path}/model.safetensors"

    save_ported_model(model, output_path, vocab_mapping)
    print_success(f"Model saved to: {output_path}")


def _make_mock_qwen(config) -> dict[str, mx.array]:
    """Create mock Qwen2.5-3B weights for testing."""
    import mlx.core as mx

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
