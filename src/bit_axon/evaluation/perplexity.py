"""Perplexity evaluation for Bit-Axon models."""

import math

import mlx.core as mx
import mlx.nn as nn


def compute_perplexity(
    model: nn.Module,
    token_ids: mx.array,
) -> tuple[float, float]:
    """Compute perplexity from token IDs.

    Args:
        model: BitAxonModel instance with __call__(input_ids) -> (logits, caches).
        token_ids: (batch, seq_len) token IDs.

    Returns:
        (perplexity, standard_error).
        PPL = exp(mean(cross_entropy_loss))
        SE ≈ PPL * std(losses) / sqrt(N_tokens)
    """
    logits, _ = model(token_ids[:, :-1])
    logits = logits.astype(mx.float32)

    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = token_ids[:, 1:].reshape(-1)

    losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")

    mean_loss = mx.mean(losses).item()
    ppl = math.exp(mean_loss)

    n_tokens = losses.size
    se = ppl * mx.sqrt(mx.var(losses, ddof=1)).item() / math.sqrt(n_tokens)

    return ppl, se


def evaluate_ppl(
    model: nn.Module,
    token_ids: mx.array,
    batch_size: int = 1,
    seq_length: int = 2048,
) -> tuple[float, float]:
    """Evaluate perplexity on a token array with batching.

    Args:
        model: BitAxonModel instance.
        token_ids: (total_tokens,) flat array or (batch, seq_len) 2D array.
        batch_size: Number of sequences per forward pass.
        seq_length: Sequence length for chunking.

    Returns:
        (perplexity, standard_error).
    """
    if token_ids.ndim == 1:
        n_tokens = token_ids.size
        n_chunks = n_tokens // seq_length
        if n_chunks == 0:
            msg = f"Not enough tokens ({n_tokens}) for seq_length={seq_length}"
            raise ValueError(msg)
        token_ids = token_ids[: n_chunks * seq_length].reshape(n_chunks, seq_length)

    n_seqs = token_ids.shape[0]
    all_losses = []

    for start in range(0, n_seqs, batch_size):
        end = min(start + batch_size, n_seqs)
        batch = token_ids[start:end]

        logits, _ = model(batch[:, :-1])
        logits = logits.astype(mx.float32)

        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = batch[:, 1:].reshape(-1)

        losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
        all_losses.append(losses)

    all_losses = mx.concatenate(all_losses)
    mean_loss = mx.mean(all_losses).item()
    ppl = math.exp(mean_loss)

    n_tokens = all_losses.size
    se = ppl * mx.sqrt(mx.var(all_losses, ddof=1)).item() / math.sqrt(n_tokens)

    return ppl, se
