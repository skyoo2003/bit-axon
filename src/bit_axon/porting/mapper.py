"""Weight mapping functions for embedding extraction and RMSNorm padding."""

from __future__ import annotations

import mlx.core as mx


def extract_embeddings(
    qwen_weights: dict[str, mx.array],
    vocab_mapping: dict[int, int],
    target_vocab_size: int = 32000,
    source_hidden_dim: int = 2048,
) -> mx.array:
    """Extract and reorder embedding rows from Qwen's vocabulary.

    Args:
        qwen_weights: Dict from mx.load("qwen*.safetensors"), must contain "model.embed_tokens.weight".
        vocab_mapping: {old_qwen_id: new_bitaxon_id} mapping (from vocab_map.build_vocab_mapping).
        target_vocab_size: 32000.
        source_hidden_dim: 2048 (Qwen's hidden_size).

    Returns:
        mx.array of shape (target_vocab_size, source_hidden_dim), dtype float32.
    """
    source = qwen_weights["model.embed_tokens.weight"]
    output = mx.zeros((target_vocab_size, source_hidden_dim), dtype=mx.float32)

    rows = []
    indices = []
    for old_id, new_id in vocab_mapping.items():
        rows.append(source[old_id])
        indices.append(new_id)

    if rows:
        gathered = mx.stack(rows, axis=0)
        output[indices] = gathered

    return output


def pad_rms_norm(
    qwen_norm_weight: mx.array,
    target_dim: int = 2560,
) -> mx.array:
    """Pad RMSNorm weight with 1.0s. RMSNorm init is ones, so padding is near-lossless.

    Args:
        qwen_norm_weight: RMSNorm weight from Qwen, shape (2048,).
        target_dim: Bit-Axon hidden_dim, default 2560.

    Returns:
        mx.array of shape (target_dim,), dtype same as input.
    """
    source_dim = qwen_norm_weight.shape[0]
    output = mx.ones((target_dim,), dtype=qwen_norm_weight.dtype)
    output[:source_dim] = qwen_norm_weight
    return output


def project_mlp_to_shared_expert(
    qwen_gate: mx.array,  # (source_intermediate, source_hidden) e.g. (11008, 2048)
    qwen_up: mx.array,  # (source_intermediate, source_hidden) e.g. (11008, 2048)
    qwen_down: mx.array,  # (source_hidden, source_intermediate) e.g. (2048, 11008)
    target_intermediate: int = 4096,
    target_hidden: int = 2560,
    source_hidden: int = 2048,
) -> tuple[mx.array, mx.array, mx.array]:
    """Project Qwen dense MLP → Bit-Axon MoE shared_expert via structured truncation + zero-padding.

    Args:
        qwen_gate: Gate projection weight from Qwen, shape (source_intermediate, source_hidden).
        qwen_up: Up projection weight from Qwen, shape (source_intermediate, source_hidden).
        qwen_down: Down projection weight from Qwen, shape (source_hidden, source_intermediate).
        target_intermediate: Bit-Axon MoE intermediate dim (default 4096).
        target_hidden: Bit-Axon hidden dim (default 2560).
        source_hidden: Qwen hidden dim (default 2048).

    Returns:
        Tuple of (gate_proj, up_proj, down_proj) reshaped for Bit-Axon MoE shared expert.
        gate_proj: (target_intermediate, target_hidden)
        up_proj: (target_intermediate, target_hidden)
        down_proj: (target_hidden, target_intermediate)
    """
    # gate_proj (source_intermediate, source_hidden) → (target_intermediate, target_hidden)
    gate_proj = mx.zeros((target_intermediate, target_hidden), dtype=qwen_gate.dtype)
    gate_proj[:target_intermediate, :source_hidden] = qwen_gate[:target_intermediate, :]

    # up_proj (source_intermediate, source_hidden) → (target_intermediate, target_hidden)
    up_proj = mx.zeros((target_intermediate, target_hidden), dtype=qwen_up.dtype)
    up_proj[:target_intermediate, :source_hidden] = qwen_up[:target_intermediate, :]

    # down_proj (source_hidden, source_intermediate) → (target_hidden, target_intermediate)
    down_proj = mx.zeros((target_hidden, target_intermediate), dtype=qwen_down.dtype)
    down_proj[:source_hidden, :target_intermediate] = qwen_down[:, :target_intermediate]

    return gate_proj, up_proj, down_proj


def init_routed_experts(
    shared_gate: mx.array,  # (target_intermediate, target_hidden)
    shared_up: mx.array,  # same
    shared_down: mx.array,  # (target_hidden, target_intermediate)
    num_experts: int = 8,
    perturbation_std: float = 0.02,
) -> tuple[mx.array, mx.array, mx.array]:
    """Init SwitchLinear routed experts: expert 0 = exact copy, experts 1+ = copy + N(0, std) noise.

    Args:
        shared_gate: Shared expert gate projection, shape (target_intermediate, target_hidden).
        shared_up: Shared expert up projection, same shape as shared_gate.
        shared_down: Shared expert down projection, shape (target_hidden, target_intermediate).
        num_experts: Number of routed experts (default 8).
        perturbation_std: Standard deviation of Gaussian perturbation for experts 1+ (default 0.02).

    Returns:
        Tuple of (routed_gate, routed_up, routed_down) each with leading expert dimension.
        routed_gate: (num_experts, target_intermediate, target_hidden)
        routed_up: (num_experts, target_intermediate, target_hidden)
        routed_down: (num_experts, target_hidden, target_intermediate)
    """

    def _init_one(shared: mx.array) -> mx.array:
        if num_experts > 1:
            noise = mx.random.normal((num_experts - 1, *shared.shape), dtype=shared.dtype) * perturbation_std
            experts_1_plus = shared[None] + noise
            return mx.concatenate([shared[None], experts_1_plus], axis=0)
        return shared[None]

    return _init_one(shared_gate), _init_one(shared_up), _init_one(shared_down)
