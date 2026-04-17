"""Adapter merging, dequantization, and model export for Bit-Axon."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from bit_axon.training.dora import DoRALinear
from bit_axon.training.lora import LoRALinear, apply_lora_to_model

_HF_TOKENIZER_META_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]


def _copy_tokenizer_meta(tokenizer: object, tokenizer_dir: Path) -> None:
    """Copy tokenizer metadata files from HuggingFace Hub when available."""
    source_path = getattr(tokenizer, "_path_or_name", None)
    if source_path is None:
        return
    try:
        import shutil

        from huggingface_hub import hf_hub_download

        source_str = str(source_path)
        if "/" not in source_str:
            return
        for fname in _HF_TOKENIZER_META_FILES:
            try:
                local = hf_hub_download(repo_id=source_str, filename=fname)
                shutil.copy2(local, tokenizer_dir / fname)
            except Exception:
                pass
    except ImportError:
        pass


def merge_adapters(model: nn.Module) -> nn.Module:
    """Walk the model tree and fuse all LoRA/DoRA adapters into base weights.

    Finds every LoRALinear and DoRALinear module, calls their fuse() method,
    and replaces them with the resulting plain nn.Linear.

    Args:
        model: Model containing LoRA/DoRA adapter modules.

    Returns:
        The modified model (mutation is in-place).
    """

    def _replace(mod: nn.Module) -> None:
        for name, child in mod.children().items():
            if isinstance(child, LoRALinear):
                fused = child.fuse(dequantize=False)
                setattr(mod, name, fused)
            elif isinstance(child, DoRALinear):
                fused = child.fuse()
                setattr(mod, name, fused)
            elif isinstance(child, nn.Module):
                _replace(child)

    _replace(model)
    return model


def dequantize_model(model: nn.Module) -> nn.Module:
    """Replace all QuantizedLinear layers with dequantized nn.Linear.

    For each nn.QuantizedLinear found in the model tree, dequantizes the
    weight and creates a plain nn.Linear with the recovered full-precision
    weights.

    Args:
        model: Model potentially containing QuantizedLinear layers.

    Returns:
        The modified model (mutation is in-place).
    """

    def _replace(mod: nn.Module) -> None:
        for name, child in mod.children().items():
            if isinstance(child, nn.QuantizedLinear):
                weight = mx.dequantize(
                    child.weight,
                    child.scales,
                    child.biases,
                    child.group_size,
                    child.bits,
                )
                has_bias = "bias" in child and child.bias is not None
                fused = nn.Linear(
                    weight.shape[1],
                    weight.shape[0],
                    bias=has_bias,
                )
                fused.weight = weight
                if has_bias:
                    fused.bias = child.bias
                setattr(mod, name, fused)
            elif isinstance(child, nn.Module):
                _replace(child)

    _replace(model)
    return model


def quantize_model(model: nn.Module, bits: int = 4, group_size: int = 64) -> nn.Module:
    """Re-quantize all nn.Linear layers to QuantizedLinear.

    Walks the model tree and replaces every nn.Linear (that has input_dim >= group_size)
    with an nn.QuantizedLinear at the specified bit-width.

    Args:
        model: Model with plain nn.Linear layers.
        bits: Quantization bit-width (default 4 for NF4).
        group_size: Quantization group size.

    Returns:
        The modified model (mutation is in-place).
    """

    def _replace(mod: nn.Module) -> None:
        for name, child in mod.children().items():
            if isinstance(child, nn.Linear) and not isinstance(child, nn.QuantizedLinear):
                if child.weight.shape[-1] >= group_size:
                    quantized = nn.QuantizedLinear.from_linear(child, group_size=group_size, bits=bits)
                    setattr(mod, name, quantized)
            elif isinstance(child, nn.Module):
                _replace(child)

    _replace(model)
    return model


def save_merged_model(
    model: nn.Module,
    output_dir: str | Path,
    config: object | None = None,
    tokenizer: object | None = None,
) -> Path:
    """Save merged model as safetensors with optional config and tokenizer.

    Args:
        model: Merged model to save.
        output_dir: Directory to write outputs into.
        config: Optional model configuration (saved as config.json).
        tokenizer: Optional tokenizer (saved via save_pretrained if available).

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(output_dir / "weights.safetensors"), weights, metadata={"format": "mlx"})

    if config is not None:
        if is_dataclass(config) and not isinstance(config, type):
            config_dict = asdict(config)
        elif hasattr(config, "__dict__"):
            config_dict = config.__dict__
        else:
            config_dict = config
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    if tokenizer is not None:
        tokenizer_dir = output_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(tokenizer_dir))
        elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "save"):
            tokenizer.tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
            _copy_tokenizer_meta(tokenizer, tokenizer_dir)

    return output_dir


def load_and_merge(
    base_model_path: str | Path,
    adapter_path: str | Path,
    output_dir: str | Path,
    config: object | None = None,
    quantize_after_merge: bool = True,
    bits: int = 4,
    group_size: int = 64,
    lora_rank: int = 8,
) -> Path:
    """End-to-end pipeline: load base + adapters, merge, optional re-quantize, save.

    Args:
        base_model_path: Directory containing base model safetensors files.
        adapter_path: Path to adapter weights (.safetensors file).
        output_dir: Directory to save the final merged model.
        config: Optional BitAxonConfig. Defaults to BitAxonConfig() if not provided.
        quantize_after_merge: Whether to re-quantize after merging (default True).
        bits: Quantization bit-width for re-quantization.
        group_size: Quantization group size.
        lora_rank: LoRA rank used for the adapter weights.

    Returns:
        Path to the output directory containing the merged model.
    """
    from bit_axon.config import BitAxonConfig
    from bit_axon.model import BitAxonModel

    base_model_path = Path(base_model_path)
    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)

    base_weights: dict[str, mx.array] = {}
    for sf_file in sorted(base_model_path.glob("*.safetensors")):
        loaded = mx.load(str(sf_file))
        base_weights.update(loaded)

    if config is None:
        config = BitAxonConfig()
    model = BitAxonModel(config)
    model.update(model.parameters())
    model.update(tree_unflatten(list(base_weights.items())))

    apply_lora_to_model(model, rank=lora_rank, dropout=0.0)
    mx.eval(model.parameters())

    adapter_weights = mx.load(str(adapter_path))
    if isinstance(adapter_weights, dict):
        model.update(tree_unflatten(list(adapter_weights.items())), strict=False)

    merge_adapters(model)

    if quantize_after_merge:
        quantize_model(model, bits=bits, group_size=group_size)

    return save_merged_model(model, output_dir)
