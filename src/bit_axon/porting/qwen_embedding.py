"""Port Qwen2.5-3B's token-embedding table into a Bit-Axon model.

Why just the embedding (and via weight-tying, the LM head)?
-----------------------------------------------------------
The Mamba-3 block layout differs structurally from Qwen's Transformer, so a
full weight port is lossy at best. But the **input/output vocabulary layers**
are architecture-agnostic — they are just a lookup table plus its transpose.
Both Qwen2.5-3B and our Bit-Axon models use the Qwen tokenizer, so the
token → vector mapping is directly reusable. Initializing the random
Bit-Axon model's embedding with Qwen's pre-trained embedding collapses the
"every out-of-domain token is predicted uniformly" regime that causes
perplexity ≈ exp(log(vocab_size)) in random-init runs.

Shape handling
--------------
Qwen2.5-3B's `model.embed_tokens.weight` has shape ``(151936, 2048)``.
Our BitAxonConfig exposes `d_source_model`:

- `large()` uses 2048 → exact reuse, no loss.
- `medium()` uses 1536 → first 1536 dims (approximate; Qwen's embedding is
  isotropic enough in early training that truncation retains most
  information).
- `small()` uses 128 → first 128 dims (aggressive; still vastly better than
  random for smoke-test purposes).

A proper down-projection (PCA / random Gaussian) could reduce the
information loss, but prefix-truncation is simple, deterministic, and
good enough for the pipeline-validation use case.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download

_DEFAULT_TOKENIZER = "Qwen/Qwen2.5-3B"


def load_qwen_embedding(
    tokenizer_id: str = _DEFAULT_TOKENIZER,
    d_source: int | None = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Download Qwen's embedding table and return the first ``d_source`` dims.

    Args:
        tokenizer_id: HF repo id whose ``model.embed_tokens.weight`` we reuse.
            Defaults to Qwen/Qwen2.5-3B.
        d_source: Requested embedding width. ``None`` returns the native width.
        dtype: Output dtype (weights are stored in bfloat16 by Qwen; we cast).

    Returns:
        Array of shape ``(vocab_size, d_source)`` dtype ``dtype``.
    """
    index_path = hf_hub_download(tokenizer_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shard_name = weight_map.get("model.embed_tokens.weight")
    if shard_name is None:
        msg = f"{tokenizer_id} does not publish 'model.embed_tokens.weight' in its safetensors index. Qwen embedding port requires that key."
        raise RuntimeError(msg)
    shard_path = hf_hub_download(tokenizer_id, shard_name)
    shard = mx.load(shard_path)
    emb = shard["model.embed_tokens.weight"]
    if d_source is None:
        return emb.astype(dtype)
    V, D_native = emb.shape
    if d_source > D_native:
        msg = f"Requested d_source={d_source} exceeds Qwen's native dim {D_native}. Up-projection is not supported; use d_source ≤ Qwen hidden dim."
        raise ValueError(msg)
    return emb[:, :d_source].astype(dtype)


def apply_qwen_embedding_init(
    model: nn.Module,
    tokenizer_id: str = _DEFAULT_TOKENIZER,
    verbose: bool = False,
) -> None:
    """Overwrite the model's ``embed_tokens.weight`` with Qwen's.

    Assumes the model has ``model.embed_tokens`` (an ``nn.Embedding``) whose
    weight has shape ``(vocab_size, d_source)``. When the model uses weight
    tying (``lm_head.weight = embed_tokens.weight`` at init), we retie after
    the replacement so the assignment propagates to the LM head.
    """
    if not hasattr(model, "embed_tokens"):
        msg = "model has no 'embed_tokens' attribute"
        raise AttributeError(msg)
    current = model.embed_tokens.weight
    V_target, D_target = current.shape
    ported = load_qwen_embedding(tokenizer_id=tokenizer_id, d_source=D_target, dtype=current.dtype)
    V_ported, D_ported = ported.shape
    if V_ported != V_target:
        if V_ported > V_target:
            ported = ported[:V_target, :]
        else:
            pad = mx.zeros((V_target - V_ported, D_ported), dtype=current.dtype)
            ported = mx.concatenate([ported, pad], axis=0)
    # Rescale the ported embedding so its overall std matches what the
    # model's downstream layers were initialized against. Without this,
    # pre-trained Qwen magnitudes (especially after aggressive dim truncation
    # to small configs' d_source=128) cause immediate gradient blow-up and
    # NaN losses. Direction is preserved; only magnitude is adjusted.
    cur_std = float(current.astype(mx.float32).std())
    port_std = float(ported.astype(mx.float32).std())
    if port_std > 1e-6 and cur_std > 0.0:
        ported = ported * (cur_std / port_std)
    model.embed_tokens.weight = ported
    # Retie the LM head if config declares weight_tying (assigning a fresh
    # MLX array to embed_tokens.weight breaks the shared-object link).
    if hasattr(model, "lm_head") and hasattr(model, "config") and getattr(model.config, "weight_tying", False):
        model.lm_head.weight = model.embed_tokens.weight
    if verbose:
        src_file = Path(tokenizer_id).name
        print(f"[porting] ported embed_tokens from {src_file}: shape={ported.shape}")
