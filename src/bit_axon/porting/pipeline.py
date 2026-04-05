"""Full Qwen → Bit-Axon initialization pipeline.

Composes all mapping functions into a single pipeline that initializes a BitAxonModel
from Qwen2.5-3B weights: embedding extraction, RMSNorm padding, MLP→shared_expert
projection, and routed expert initialization.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel
from bit_axon.porting.mapper import (
    extract_embeddings,
    init_routed_experts,
    pad_rms_norm,
    project_mlp_to_shared_expert,
)
from bit_axon.porting.weight_map import build_key_mappings


def initialize_from_qwen_weights(
    qwen_weights: dict[str, mx.array],
    vocab_mapping: dict[int, int] | None = None,
    config: BitAxonConfig | None = None,
) -> tuple[BitAxonModel, dict[int, int]]:
    """Initialize BitAxonModel with structured projections from Qwen weights.

    Creates a fresh BitAxonModel (all default inits), then overlays ported weights:
    - Embedding: extract rows from Qwen's vocabulary via vocab_mapping
    - RMSNorm: pad from d_source_model → hidden_dim with 1.0s
    - MoE shared_expert: project Qwen MLP via structured truncation + zero-padding
    - MoE routed experts: copy from shared_expert + perturbation

    Args:
        qwen_weights: Dict of Qwen2.5-3B weights (from mx.load). Must contain:
            - model.embed_tokens.weight (source_vocab, d_source_model)
            - model.layers.{i}.input_layernorm.weight (d_source_model,) for all layers
            - model.layers.{i}.post_attention_layernorm.weight for MoE layers
            - model.layers.{i}.mlp.{gate,up,down}_proj.weight for MoE layers
        vocab_mapping: {old_qwen_id: new_bitaxon_id}. If None, uses identity mapping
            for first vocab_size tokens.
        config: BitAxonConfig. If None, uses default config.

    Returns:
        (model, vocab_mapping) — initialized model with ported weights.
    """
    # 1. Config defaults
    if config is None:
        config = BitAxonConfig()

    # 2. Create fresh model with default inits
    model = BitAxonModel(config)

    # 3. Vocab mapping defaults to identity for first vocab_size tokens
    if vocab_mapping is None:
        vocab_mapping = {i: i for i in range(config.vocab_size)}

    # 4. Get all key mappings
    mappings = build_key_mappings(config)

    # 5. Get model's current params as mutable flat dict
    params = dict(tree_flatten(model.parameters()))

    # 6a. Collect MoE layer indices for holistic processing
    moe_layers: set[int] = set()
    for m in mappings:
        if m.transform in ("moe_project", "copy_perturb"):
            key = m.target_key
            if key.startswith("layer_"):
                idx = int(key[len("layer_") : key.index(".", len("layer_"))])
                moe_layers.add(idx)

    # 6b. Process each MoE layer: project shared_expert, then init routed experts
    for layer_idx in sorted(moe_layers):
        qwen_gate = qwen_weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        qwen_up = qwen_weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        qwen_down = qwen_weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"]

        shared_gate, shared_up, shared_down = project_mlp_to_shared_expert(
            qwen_gate,
            qwen_up,
            qwen_down,
            target_intermediate=config.moe_intermediate_dim,
            target_hidden=config.hidden_dim,
            source_hidden=config.d_source_model,
        )

        # Assign shared_expert weights
        shared_prefix = f"layer_{layer_idx}.moe.shared_expert"
        params[f"{shared_prefix}.gate_proj.weight"] = shared_gate
        params[f"{shared_prefix}.up_proj.weight"] = shared_up
        params[f"{shared_prefix}.down_proj.weight"] = shared_down

        # Init routed experts from shared_expert + perturbation
        routed_gate, routed_up, routed_down = init_routed_experts(
            shared_gate,
            shared_up,
            shared_down,
            num_experts=config.moe_num_experts,
        )

        # Assign switch_mlp weights
        switch_prefix = f"layer_{layer_idx}.moe.switch_mlp"
        params[f"{switch_prefix}.gate_proj.weight"] = routed_gate
        params[f"{switch_prefix}.up_proj.weight"] = routed_up
        params[f"{switch_prefix}.down_proj.weight"] = routed_down

    # 6c. Process remaining transforms (vocab_extract, pad_2048_2560)
    for m in mappings:
        if m.transform == "default":
            continue
        if m.transform in ("moe_project", "copy_perturb"):
            continue

        if m.transform == "vocab_extract":
            emb = extract_embeddings(qwen_weights, vocab_mapping, config.vocab_size, config.d_source_model)
            params[m.target_key] = emb

        elif m.transform == "pad_2048_2560":
            params[m.target_key] = pad_rms_norm(qwen_weights[m.source_key], config.hidden_dim)

    # 7. Update model with ported weights
    model.update(tree_unflatten(list(params.items())))

    # 8. Return initialized model and vocab mapping
    return model, vocab_mapping


def save_ported_model(
    model: BitAxonModel,
    output_path: str,
    vocab_mapping: dict[int, int],
) -> None:
    """Save ported model weights + vocab mapping as safetensors.

    Args:
        model: Initialized BitAxonModel with ported weights.
        output_path: Path to write the safetensors file.
        vocab_mapping: Vocabulary mapping to store in metadata.
    """
    params = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(output_path, params, metadata={"vocab_mapping": str(vocab_mapping)})
