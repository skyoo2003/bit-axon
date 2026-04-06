"""Logit sampling strategies for autoregressive generation."""

from __future__ import annotations

import mlx.core as mx


def sample_logits(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
) -> mx.array:
    """Sample token IDs from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: Shape (batch, vocab_size) or (vocab_size,).
        temperature: Sampling temperature. 0.0 = greedy (argmax).
        top_k: Keep only top-k logits. 0 = disabled.
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        seed: Optional random seed for reproducibility.

    Returns:
        Token IDs with shape (batch,) or scalar.
    """
    if seed is not None:
        mx.random.seed(seed)

    if logits.ndim == 1:
        logits = logits[None, :]
        squeeze = True
    else:
        squeeze = False

    if temperature == 0.0:
        result = mx.argmax(logits, axis=-1)
        return result.squeeze(0) if squeeze else result

    logits = logits / temperature

    if top_k > 0:
        top_k_val = min(top_k, logits.shape[-1])
        kth = mx.sort(logits, axis=-1)[..., -top_k_val]
        indices_to_remove = logits < kth[..., None]
        logits = mx.where(indices_to_remove, mx.array(float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits = -mx.sort(-logits, axis=-1)
        cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove = mx.roll(sorted_remove, 1, axis=-1)
        sorted_remove[:, 0] = False
        sorted_kept = mx.where(sorted_remove, mx.array(float("inf")), sorted_logits)
        threshold = mx.min(sorted_kept, axis=-1, keepdims=True)
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    result = mx.random.categorical(logits, axis=-1)

    return result.squeeze(0) if squeeze else result
