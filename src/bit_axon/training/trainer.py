"""Core training loop components for Bit-Axon."""

from __future__ import annotations

import gc
from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from mlx.optimizers import clip_grad_norm as mlx_clip_grad_norm
from mlx.utils import tree_flatten, tree_map

from bit_axon.training.loss import cross_entropy_loss


def make_loss_fn(model: nn.Module) -> Callable:
    """Create a loss function for the given model.

    Args:
        model: BitAxonModel instance.

    Returns:
        Function: (model, input_ids, labels) -> (loss_scalar, num_valid_tokens)
    """

    def loss_fn(model, input_ids, labels):
        logits, _ = model(input_ids)
        logits = logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        loss, n_valid = cross_entropy_loss(logits, labels_shifted)
        return loss, n_valid

    return loss_fn


def create_loss_and_grad(model: nn.Module) -> Callable:
    """Create a loss-and-gradient function using nn.value_and_grad.

    Args:
        model: BitAxonModel instance with LoRA adapters applied.

    Returns:
        Function: (model, input_ids, labels) -> ((loss, n_valid), grads)
    """
    loss_fn = make_loss_fn(model)
    return nn.value_and_grad(model, loss_fn)


def clip_grad_norm_(grads: dict, max_norm: float) -> tuple[dict, mx.array]:
    """Clip gradient norm in-place.

    Args:
        grads: Gradient dictionary from nn.value_and_grad.
        max_norm: Maximum gradient norm.

    Returns:
        (clipped_grads, total_norm) tuple.
    """
    return mlx_clip_grad_norm(grads, max_norm)


def accumulate_gradients(grads: dict | None, new_grads: dict) -> dict:
    """Accumulate gradients for gradient accumulation.

    Args:
        grads: Previous accumulated gradients (None for first step).
        new_grads: New gradients from current step.

    Returns:
        Accumulated gradients dict.
    """
    if grads is None:
        return new_grads
    return tree_map(lambda g, ng: g + ng, grads, new_grads)


def get_trainable_params(model: nn.Module) -> dict:
    """Extract only trainable (adapter) parameters from model.

    Filters parameters to only include LoRA/DoRA adapter weights.
    Base model parameters are frozen and should not be included.

    Args:
        model: BitAxonModel with LoRA adapters applied.

    Returns:
        Filtered parameter dictionary with only adapter params.
    """
    all_params = tree_flatten(model.parameters())
    adapter_params = {}
    for key, value in all_params:
        if any(k in key for k in ("lora_a", "lora_b")) or key.endswith(".m"):
            adapter_params[key] = value
    return adapter_params


class Trainer:
    """Thermal-aware QLoRA SFT trainer for Bit-Axon.

    Orchestrates the full training loop: data iteration, gradient accumulation,
    gradient clipping, checkpointing, evaluation, and thermal gating.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        dataset,
        val_dataset=None,
        cooling_scheduler=None,
        on_step=None,
    ):
        from bit_axon.training.config import TrainingConfig

        self.model = model
        self.config: TrainingConfig = config
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.cooling = cooling_scheduler
        self.on_step = on_step
        self.optimizer = None
        self.step_count = 0
        _loss_and_grad = None

    def setup(self) -> None:
        """Initialize optimizer, scheduler, and loss function. Resume from checkpoint if available."""
        from bit_axon.training.checkpoint import get_latest_checkpoint, load_checkpoint
        from bit_axon.training.scheduler import build_lr_schedule

        mx.random.seed(self.config.seed)

        lr_schedule = build_lr_schedule(
            self.config.learning_rate,
            self.config.warmup_steps,
            self.config.max_steps,
            initial_step=self.step_count,
        )
        self.optimizer = AdamW(learning_rate=lr_schedule, weight_decay=self.config.weight_decay)

        ckpt = get_latest_checkpoint(self.config.output_dir)
        if ckpt is not None:
            step, _ = load_checkpoint(self.model, self.optimizer, ckpt)
            self.step_count = step

    def _compute_loss_and_grads(self, input_ids: mx.array, labels: mx.array) -> tuple[mx.array, mx.array, dict]:
        """Compute loss and gradients for a single batch.

        Uses the model's normal forward path with nn.value_and_grad.
        Gradient checkpointing is applied at the model level to limit
        activation memory during backward.
        """

        def loss_fn(model, input_ids, labels):
            logits, _ = model(input_ids)
            loss, ntoks = cross_entropy_loss(logits, labels)
            return loss, ntoks

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        (loss, ntoks), grads = loss_and_grad(self.model, input_ids, labels)
        mx.eval(loss, ntoks)
        mx.clear_cache()
        return loss, ntoks, grads

    def _compute_loss_and_grads_v2(self, input_ids: mx.array, labels: mx.array) -> tuple[mx.array, mx.array, dict]:
        """Memory-efficient loss/grad computation placeholder.

        NOTE: MLX's mx.grad does not support closures that reference model state or
        external arrays. A true memory-efficient implementation would require either:
        (a) MLX native support for checkpointing with nn.value_and_grad, or
        (        b) Manual activation checkpointing at the layer level.
        Falls back to the standard _compute_loss_and_grads for now.
        """
        return self._compute_loss_and_grads(input_ids, labels)

    def train(self) -> dict:
        """Run the main training loop.

        Returns:
            Dict with final training stats: {"step": int, "loss": float, "grad_norm": float}
        """
        from bit_axon.training.checkpoint import save_checkpoint
        from bit_axon.training.collate import iterate_batches

        self.setup()

        mx.reset_peak_memory()

        batch_iter = iterate_batches(
            self.dataset,
            batch_size=self.config.batch_size,
            max_seq_len=self.config.max_seq_len,
            shuffle=True,
            loop=True,
            seed=self.config.seed,
            eos_token_id=self.config.eos_token_id,
        )

        grad_accum = None
        last_loss = 0.0
        last_grad_norm = 0.0

        while self.step_count < self.config.max_steps:
            batch = next(batch_iter)

            if self.cooling is not None:
                self.cooling.check_before_step(self.step_count)

            loss, ntoks, grads = (
                self._compute_loss_and_grads_v2(batch[0], batch[1]) if self.config.low_memory else self._compute_loss_and_grads(batch[0], batch[1])
            )
            last_loss = float(loss)

            if float(ntoks) == 0:
                del grads, loss, batch
                mx.clear_cache()
                continue

            grad_accum = accumulate_gradients(grad_accum, grads)
            del grads

            do_update = (self.step_count + 1) % self.config.grad_accum_steps == 0
            if do_update and grad_accum is not None:
                grad_accum, last_grad_norm = clip_grad_norm_(grad_accum, self.config.max_grad_norm)
                self.optimizer.update(self.model, grad_accum)
                grad_accum = None

            mx.eval(self.model.state, self.optimizer.state, loss)
            del batch
            self.step_count += 1

            if self.on_step is not None:
                self.on_step(self.step_count, {"loss": last_loss, "grad_norm": float(last_grad_norm)})

            if do_update:
                mx.clear_cache()
                gc.collect()

            if self.step_count % self.config.save_every == 0:
                mx.reset_peak_memory()
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.step_count,
                    last_loss,
                    output_dir=self.config.output_dir,
                )

            if self.val_dataset is not None and self.step_count % self.config.eval_every == 0:
                eval_result = self.evaluate()
                if self.on_step is not None:
                    self.on_step(
                        self.step_count,
                        {
                            "loss": last_loss,
                            "grad_norm": float(last_grad_norm),
                            "eval_loss": eval_result["loss"],
                            "eval_ppl": eval_result["perplexity"],
                        },
                    )

        return {"step": self.step_count, "loss": last_loss, "grad_norm": float(last_grad_norm)}

    def evaluate(self) -> dict:
        """Run evaluation on val_dataset.

        Returns:
            {"loss": float, "perplexity": float}
        """
        from bit_axon.training.collate import iterate_batches

        if self.val_dataset is None:
            return {"loss": 0.0, "perplexity": 0.0}

        total_loss = 0.0
        total_tokens = 0.0

        for num_batches, batch in enumerate(
            iterate_batches(
                self.val_dataset,
                batch_size=self.config.batch_size,
                max_seq_len=self.config.max_seq_len,
                shuffle=False,
                loop=False,
                eos_token_id=self.config.eos_token_id,
            ),
        ):
            logits, _ = self.model(batch[0])
            loss, n_valid = cross_entropy_loss(logits, batch[1])
            mx.eval(loss, n_valid)
            total_loss += float(loss) * float(n_valid)
            total_tokens += float(n_valid)
            if num_batches >= self.config.eval_batches - 1:
                break

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = mx.exp(mx.array(avg_loss))
        mx.eval(perplexity)
        return {"loss": avg_loss, "perplexity": float(perplexity)}
