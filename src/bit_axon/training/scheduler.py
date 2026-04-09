"""Learning rate schedules for training."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx.optimizers import cosine_decay, join_schedules, linear_schedule


def build_lr_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> Callable[[int], mx.array]:
    """Build cosine decay schedule with linear warmup.

    Args:
        learning_rate: Peak learning rate after warmup.
        warmup_steps: Number of linear warmup steps (0 to skip warmup).
        total_steps: Total training steps.
        min_lr: Minimum learning rate at end of cosine decay.

    Returns:
        Schedule function: step_index -> learning_rate (mx.array).
    """
    if warmup_steps > 0:
        warmup = linear_schedule(0.0, learning_rate, warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        cosine = cosine_decay(learning_rate, decay_steps, end=min_lr)
        return join_schedules([warmup, cosine], [warmup_steps + 1])
    else:
        decay_steps = max(total_steps, 1)
        return cosine_decay(learning_rate, decay_steps, end=min_lr)
