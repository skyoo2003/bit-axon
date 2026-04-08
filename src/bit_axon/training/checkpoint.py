"""Checkpoint save/load utilities for training."""

import json
import shutil
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from bit_axon.training.trainer import get_trainable_params


def save_checkpoint(
    model: nn.Module,
    optimizer,
    step: int,
    loss: float,
    output_dir: str | Path,
    max_checkpoints: int = 3,
) -> Path:
    """Save training checkpoint (adapter weights + optimizer state + metadata).

    Args:
        model: Model with LoRA/DoRA adapters.
        optimizer: MLX optimizer with state.
        step: Current training step.
        loss: Current loss value.
        output_dir: Root checkpoint directory.
        max_checkpoints: Maximum number of checkpoints to keep.

    Returns:
        Path to the saved checkpoint directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    adapter_weights = dict(tree_flatten(get_trainable_params(model)))
    mx.save_safetensors(str(ckpt_dir / "adapters.safetensors"), adapter_weights)

    flat_opt_state = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(str(ckpt_dir / "optimizer_state.safetensors"), flat_opt_state)

    metadata = {
        "step": step,
        "loss": loss,
    }
    with open(ckpt_dir / "training_state.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _rotate_checkpoints(output_dir, max_checkpoints)

    return ckpt_dir


def load_checkpoint(
    model: nn.Module,
    optimizer,
    checkpoint_dir: str | Path,
) -> tuple[int, float]:
    """Load training checkpoint.

    Args:
        model: Model to load adapter weights into.
        optimizer: Optimizer to restore state for.
        checkpoint_dir: Path to checkpoint directory.

    Returns:
        (step, loss) tuple from the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)

    adapter_weights = cast(dict[str, mx.array], mx.load(str(checkpoint_dir / "adapters.safetensors")))
    flat_opt_state = cast(dict[str, mx.array], mx.load(str(checkpoint_dir / "optimizer_state.safetensors")))

    with open(checkpoint_dir / "training_state.json") as f:
        metadata = json.load(f)

    nested_weights = tree_unflatten(list(adapter_weights.items()))
    model.update(nested_weights)

    optimizer.state = tree_unflatten(list(flat_opt_state.items()))

    return metadata["step"], metadata["loss"]


def get_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Find the most recent checkpoint by step number.

    Args:
        output_dir: Root checkpoint directory.

    Returns:
        Path to latest checkpoint directory, or None if no checkpoints exist.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    checkpoints = sorted(output_dir.glob("step_*"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def save_adapter_only(model: nn.Module, output_path: str | Path) -> None:
    """Save only trainable (adapter) weights for inference deployment.

    Args:
        model: Model with LoRA/DoRA adapters.
        output_path: Output file path (.safetensors).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_weights = dict(tree_flatten(get_trainable_params(model)))
    mx.save_safetensors(str(output_path), adapter_weights)


def _rotate_checkpoints(output_dir: Path, max_checkpoints: int) -> None:
    """Delete oldest checkpoints if count exceeds max_checkpoints."""
    checkpoints = sorted(output_dir.glob("step_*"))
    while len(checkpoints) > max_checkpoints:
        oldest = checkpoints.pop(0)
        shutil.rmtree(oldest)
