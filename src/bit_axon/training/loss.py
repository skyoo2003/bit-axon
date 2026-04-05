from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def cross_entropy_loss(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = -100,
) -> tuple[mx.array, mx.array]:
    """Compute cross-entropy loss with support for ignored positions.

    Args:
        logits: Shape (B, T, V) or (T, V).
        labels: Shape (B, T) or (T,). Positions equal to ignore_index are excluded.
        ignore_index: Token ID value to ignore (default: -100).

    Returns:
        (scalar_loss, num_valid_tokens) where loss is mean CE over valid positions.
    """
    V = logits.shape[-1]
    mask = labels != ignore_index
    safe_labels = mx.where(mask, labels, mx.zeros_like(labels))
    ce = nn.losses.cross_entropy(logits.reshape(-1, V), safe_labels.reshape(-1), reduction="none")
    ce = ce.reshape(labels.shape)
    mask_f = mask.astype(mx.float32)
    ce = ce * mask_f
    num_valid = mask.sum()
    loss = ce.sum() / mx.maximum(num_valid, 1)
    return loss, num_valid
