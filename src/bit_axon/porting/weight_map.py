"""Source-to-target parameter key enumeration for Qwen2.5-3B → Bit-Axon weight porting."""

from dataclasses import dataclass

from mlx.utils import tree_flatten

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel


@dataclass
class KeyMapping:
    """A single source→target parameter mapping for weight porting."""

    target_key: str
    source_key: str | None  # None = no Qwen equivalent (default init)
    transform: str  # "default", "vocab_extract", "pad_2048_2560", "moe_project", "copy_perturb"


def enumerate_target_keys(config: BitAxonConfig) -> list[str]:
    """Return all parameter keys from a BitAxonModel instance, sorted."""
    model = BitAxonModel(config)
    flat = tree_flatten(model.parameters())
    return sorted(k for k, _ in flat)


def build_key_mappings(config: BitAxonConfig) -> list[KeyMapping]:
    """Return list of all source→target mappings for weight porting."""
    target_keys = enumerate_target_keys(config)
    third = config.num_layers // 3
    mappings: list[KeyMapping] = []
    for key in target_keys:
        source, transform = _classify_key(key, third)
        mappings.append(KeyMapping(target_key=key, source_key=source, transform=transform))
    return mappings


def _classify_key(key: str, third: int) -> tuple[str | None, str]:
    """Determine the source key and transform for a single target key."""
    # Top-level keys
    if key == "embed_tokens.weight":
        return "model.embed_tokens.weight", "vocab_extract"
    if key == "lm_head.weight":
        return "model.embed_tokens.weight", "vocab_extract"
    if key in ("input_proj.weight", "output_proj.weight"):
        return None, "default"

    # Layer keys: parse "layer_{i}.rest"
    if not key.startswith("layer_"):
        return None, "default"
    rest = key[len("layer_") :]
    dot_pos = rest.index(".")
    layer_idx = int(rest[:dot_pos])
    suffix = rest[dot_pos + 1 :]

    # Determine layer type
    if layer_idx < third:
        layer_type = "ssm"
    elif layer_idx < 2 * third:
        layer_type = "swa_moe"
    else:
        layer_type = "ssm_moe"

    return _classify_layer_param(layer_idx, suffix, layer_type)


def _classify_layer_param(layer_idx: int, suffix: str, layer_type: str) -> tuple[str | None, str]:
    """Classify a parameter within a specific layer."""
    # input_norm → Qwen input_layernorm (all layers)
    if suffix == "input_norm.weight":
        return f"model.layers.{layer_idx}.input_layernorm.weight", "pad_2048_2560"

    # post_attention_norm (SWA+MoE layers)
    if suffix == "post_attention_norm.weight":
        return (
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            "pad_2048_2560",
        )

    # post_ssm_norm (SSM+MoE layers)
    if suffix == "post_ssm_norm.weight":
        return (
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            "pad_2048_2560",
        )

    # SSM parameters (SSM and SSM+MoE layers) — no Qwen source
    if suffix.startswith("ssm."):
        return None, "default"

    # Attention parameters (SWA+MoE layers) — no Qwen source
    if suffix.startswith("attention."):
        return None, "default"

    # MoE parameters
    if suffix.startswith("moe."):
        return _classify_moe_param(layer_idx, suffix)

    return None, "default"


def _classify_moe_param(layer_idx: int, suffix: str) -> tuple[str | None, str]:
    """Classify a MoE sub-parameter."""
    # shared_expert gate/up/down → Qwen MLP projections
    if suffix == "moe.shared_expert.gate_proj.weight":
        return f"model.layers.{layer_idx}.mlp.gate_proj.weight", "moe_project"
    if suffix == "moe.shared_expert.up_proj.weight":
        return f"model.layers.{layer_idx}.mlp.up_proj.weight", "moe_project"
    if suffix == "moe.shared_expert.down_proj.weight":
        return f"model.layers.{layer_idx}.mlp.down_proj.weight", "moe_project"

    # switch_mlp (routed experts) → copied from shared_expert with perturbation
    if suffix.startswith("moe.switch_mlp."):
        return None, "copy_perturb"

    # gate and shared_expert_gate → no Qwen source
    return None, "default"
