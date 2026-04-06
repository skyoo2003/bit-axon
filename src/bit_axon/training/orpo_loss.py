"""ORPO (Odds Ratio Preference Optimization) loss for preference alignment."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from bit_axon.training.loss import cross_entropy_loss

_LN2 = 0.6931471805599453


def log1mexp(x: mx.array) -> mx.array:
    """Compute log(1 - exp(x)) in a numerically stable way.

    Uses different branches depending on the magnitude of x to avoid
    catastrophic cancellation around the boundary at -ln(2).

    Args:
        x: Input array, typically containing log-probabilities (negative values).

    Returns:
        log(1 - exp(x)) computed with appropriate numerical stabilization.
    """
    threshold = mx.array(-_LN2)
    use_branch1 = x < threshold  # x < -ln(2) → use mx.log(-mx.expm1(x))
    safe_branch1 = mx.where(use_branch1, x, threshold)
    branch1 = mx.log(-mx.expm1(safe_branch1))
    safe_branch2 = mx.where(use_branch1, threshold, x)
    branch2 = mx.log1p(-mx.exp(safe_branch2))
    return mx.where(use_branch1, branch1, branch2)


def get_logps(logits: mx.array, labels: mx.array, mask: mx.array | None = None) -> mx.array:
    """Compute averaged log-probabilities per sequence (response-only).

    Applies a teacher-forcing shift so that position t in the logits predicts
    token t+1 in the labels, then averages the log-probabilities over valid
    (masked) positions for each sequence in the batch.

    Args:
        logits: Raw model output, shape (B, T, V).
        labels: Token IDs, shape (B, T).
        mask: Optional binary mask, shape (B, T-1). When ``None`` the mask is
            derived from ``labels != -100`` on the shifted labels.

    Returns:
        Per-sequence averaged log-probability, shape (B,).
    """
    # Shift: logits[t] predicts labels[t+1]
    logits_shifted = logits[:, :-1, :]
    labels_shifted = labels[:, 1:]

    # Per-token log-probabilities (negate CE because CE = -log p)
    log_probs = -nn.losses.cross_entropy(logits_shifted, labels_shifted, reduction="none")
    log_probs = mx.clip(log_probs, -1000.0, 0.0)

    # Build mask from labels if not provided
    if mask is None:
        mask = mx.array(labels_shifted != -100, dtype=mx.float32)

    # Masked mean over sequence dimension
    sum_logps = (log_probs * mask).sum(axis=-1)
    n_valid = mask.sum(axis=-1)
    avg_logps = sum_logps / mx.maximum(n_valid, 1.0)
    return avg_logps


def orpo_loss(chosen_logps: mx.array, rejected_logps: mx.array, beta: float = 0.1) -> mx.array:
    """Compute ORPO odds-ratio preference loss.

    The odds-ratio log compares the odds of the chosen response against the
    rejected response and applies a log-sigmoid with a temperature parameter
    ``beta`` to produce a differentiable loss.

    Args:
        chosen_logps: Averaged log-probs for chosen responses, shape (B,).
        rejected_logps: Averaged log-probs for rejected responses, shape (B,).
        beta: Temperature scaling for the odds-ratio penalty (default 0.1).

    Returns:
        Scalar ORPO penalty loss.
    """
    log_odds = (chosen_logps - rejected_logps) - (log1mexp(chosen_logps) - log1mexp(rejected_logps))
    loss = -mx.mean(nn.log_sigmoid(beta * log_odds))
    return loss


def compute_orpo_loss(
    model: nn.Module,
    chosen_ids: mx.array,
    chosen_labels: mx.array,
    rejected_ids: mx.array,
    rejected_labels: mx.array,
    beta: float = 0.1,
) -> tuple[mx.array, dict]:
    """Full ORPO loss pipeline: SFT NLL + odds-ratio preference penalty.

    Runs forward passes for both chosen and rejected sequences, computes the
    supervised NLL loss on chosen, extracts averaged log-probs for both, and
    combines everything into the final ORPO training objective.

    Args:
        model: Language model with a ``__call__`` method returning logits.
        chosen_ids: Token IDs for chosen responses, shape (B, T).
        chosen_labels: Labels for chosen responses, shape (B, T).
        rejected_ids: Token IDs for rejected responses, shape (B, T).
        rejected_labels: Labels for rejected responses, shape (B, T).
        beta: Temperature for the odds-ratio penalty (default 0.1).

    Returns:
        Tuple of (total_loss, metrics_dict) where total_loss is a scalar and
        metrics_dict contains per-component scalars for logging.
    """
    # Forward passes
    logits_chosen = model(chosen_ids)
    logits_rejected = model(rejected_ids)

    # SFT / NLL component on chosen sequences
    nll_loss, _ = cross_entropy_loss(logits_chosen, chosen_labels)

    # Averaged log-probabilities for preference comparison
    chosen_logps = get_logps(logits_chosen, chosen_labels)
    rejected_logps = get_logps(logits_rejected, rejected_labels)

    # ORPO odds-ratio penalty
    orpo_penalty = orpo_loss(chosen_logps, rejected_logps, beta)

    # Combined loss
    total_loss = nll_loss + orpo_penalty

    metrics = {
        "nll_loss": nll_loss,
        "orpo_loss": orpo_penalty,
        "chosen_logps": mx.mean(chosen_logps),
        "rejected_logps": mx.mean(rejected_logps),
        "reward_margin": mx.mean(chosen_logps - rejected_logps),
    }

    return total_loss, metrics
