"""ORPO preference alignment trainer for Bit-Axon."""

from __future__ import annotations

import gc

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW

from bit_axon.training.orpo_loss import compute_orpo_loss
from bit_axon.training.trainer import accumulate_gradients, clip_grad_norm_


class ORPOTrainer:
    """Thermal-aware ORPO preference trainer for Bit-Axon.

    Performs simultaneous SFT and preference alignment using the ORPO
    objective (no reference model required). Uses two forward passes
    per batch (chosen + rejected) with gradient accumulation.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        dataset,
        val_dataset=None,
        cooling_scheduler=None,
    ):
        from bit_axon.training.config import TrainingConfig

        self.model = model
        self.config: TrainingConfig = config
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.cooling = cooling_scheduler
        self.optimizer = None
        self.step_count = 0
        self._loss_and_grad = None

    def setup(self) -> None:
        """Initialize optimizer, scheduler, and loss function. Resume from checkpoint if available."""
        from bit_axon.training.checkpoint import get_latest_checkpoint, load_checkpoint
        from bit_axon.training.scheduler import build_lr_schedule

        mx.random.seed(self.config.seed)

        lr_schedule = build_lr_schedule(
            self.config.learning_rate,
            self.config.warmup_steps,
            self.config.max_steps,
        )
        self.optimizer = AdamW(learning_rate=lr_schedule, weight_decay=self.config.weight_decay)

        def loss_fn(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels):
            loss, metrics = compute_orpo_loss(
                model,
                chosen_ids,
                chosen_labels,
                rejected_ids,
                rejected_labels,
                beta=self.config.beta,
            )
            return loss, metrics

        self._loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        ckpt = get_latest_checkpoint(self.config.output_dir)
        if ckpt is not None:
            step, _ = load_checkpoint(self.model, self.optimizer, ckpt)
            self.step_count = step

    def train(self) -> dict:
        """Run the main ORPO training loop.

        Iterates over preference pairs, computing the ORPO loss (NLL + odds-ratio
        penalty), accumulating gradients, and updating the model. Tracks reward
        metrics (chosen/rejected log-probs and their margin) at each step.

        Returns:
            Dict with final training stats including reward metrics:
            {"step", "loss", "grad_norm", "chosen_reward", "rejected_reward",
             "reward_margin", "reward_accuracy"}.
        """
        from bit_axon.training.checkpoint import save_checkpoint
        from bit_axon.training.orpo_collate import iterate_orpo_batches

        self.setup()

        mx.reset_peak_memory()

        batch_iter = iterate_orpo_batches(
            self.dataset,
            batch_size=self.config.batch_size,
            max_seq_len=self.config.max_seq_len,
            shuffle=True,
            loop=True,
            seed=self.config.seed,
        )

        grad_accum = None
        last_loss = 0.0
        last_grad_norm = 0.0
        last_metrics: dict = {}

        while self.step_count < self.config.max_steps:
            batch = next(batch_iter)

            if self.cooling is not None:
                self.cooling.check_before_step(self.step_count)

            (loss, metrics), grads = self._loss_and_grad(self.model, *batch)
            mx.eval(loss)
            last_loss = float(loss)
            last_metrics = {k: float(v) for k, v in metrics.items()}

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
                self.evaluate()

        chosen_reward = self.config.beta * last_metrics.get("chosen_logps", 0.0)
        rejected_reward = self.config.beta * last_metrics.get("rejected_logps", 0.0)
        reward_margin = chosen_reward - rejected_reward
        reward_accuracy = 1.0 if last_metrics.get("chosen_logps", 0.0) > last_metrics.get("rejected_logps", 0.0) else 0.0

        return {
            "step": self.step_count,
            "loss": last_loss,
            "grad_norm": float(last_grad_norm),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "reward_margin": reward_margin,
            "reward_accuracy": reward_accuracy,
        }

    def evaluate(self) -> dict:
        """Run evaluation on val_dataset.

        Iterates over the validation preference pairs (up to 10 batches),
        computing ORPO loss and reward margin metrics without gradients.

        Returns:
            {"loss": float, "perplexity": float, "reward_margin": float}
        """
        from bit_axon.training.orpo_collate import iterate_orpo_batches

        if self.val_dataset is None:
            return {"loss": 0.0, "perplexity": 0.0, "reward_margin": 0.0}

        total_loss = 0.0
        total_reward_margin = 0.0
        num_batches = 0

        for num_batches, batch in enumerate(
            iterate_orpo_batches(
                self.val_dataset,
                batch_size=self.config.batch_size,
                max_seq_len=self.config.max_seq_len,
                shuffle=False,
                loop=False,
            ),
        ):
            loss, metrics = compute_orpo_loss(
                self.model,
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                beta=self.config.beta,
            )
            mx.eval(loss)
            total_loss += float(loss)
            total_reward_margin += float(metrics["reward_margin"])
            if num_batches >= 9:
                break

        n = num_batches + 1
        avg_loss = total_loss / n
        perplexity = mx.exp(mx.array(avg_loss))
        mx.eval(perplexity)

        return {
            "loss": avg_loss,
            "perplexity": float(perplexity),
            "reward_margin": total_reward_margin / n,
        }
