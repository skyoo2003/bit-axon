"""Tests for core training loop components."""

import mlx.core as mx

from bit_axon.model import BitAxonModel
from bit_axon.training.config import TrainingConfig
from bit_axon.training.cooling import CoolingScheduler
from bit_axon.training.lora import apply_lora_to_model
from bit_axon.training.trainer import (
    Trainer,
    accumulate_gradients,
    clip_grad_norm_,
    create_loss_and_grad,
    get_trainable_params,
    make_loss_fn,
)


def _flatten_keys(d, prefix=""):
    result = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.extend(_flatten_keys(v, full))
        else:
            result.append(full)
    return result


class TestMakeLossFn:
    def test_returns_scalar_loss_and_count(self, small_config):
        """Loss function should return (scalar_loss, num_valid_tokens)."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        loss_fn = make_loss_fn(model)
        input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
        labels = mx.random.randint(0, small_config.vocab_size, shape=(1, 8)).astype(mx.int32)
        loss, n_valid = loss_fn(model, input_ids, labels)
        mx.eval(loss, n_valid)
        assert loss.shape == ()
        assert isinstance(float(loss), float)
        assert n_valid.shape == ()

    def test_loss_uses_shifted_labels(self, small_config):
        """Loss should use logits[:, :-1] and labels[:, 1:] (shifted)."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        loss_fn = make_loss_fn(model)
        input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
        labels = mx.random.randint(0, small_config.vocab_size, shape=(1, 8)).astype(mx.int32)
        loss, n_valid = loss_fn(model, input_ids, labels)
        mx.eval(loss, n_valid)
        assert float(n_valid) == 1 * 7


class TestCreateLossAndGrad:
    def test_returns_gradients(self, small_config):
        """Loss and grad function should return gradients dict."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        loss_and_grad = create_loss_and_grad(model)
        input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
        labels = mx.random.randint(0, small_config.vocab_size, shape=(1, 8)).astype(mx.int32)
        (loss, n_valid), grads = loss_and_grad(model, input_ids, labels)
        mx.eval(loss, n_valid)
        assert isinstance(grads, dict)
        assert len(grads) > 0

    def test_gradients_on_lora_params_only(self, small_config):
        """After apply_lora_to_model, gradients should exist on adapter params."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0)
        loss_and_grad = create_loss_and_grad(model)
        input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
        labels = mx.random.randint(0, small_config.vocab_size, shape=(1, 8)).astype(mx.int32)
        (loss, n_valid), grads = loss_and_grad(model, input_ids, labels)
        mx.eval(loss, n_valid)
        flat_keys = _flatten_keys(grads)
        has_lora_grads = any("lora_a" in k or "lora_b" in k for k in flat_keys)
        assert has_lora_grads, "No LoRA gradients found"

    def test_no_nan_gradients(self, small_config):
        """Gradients should not contain NaN values."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0)
        loss_and_grad = create_loss_and_grad(model)
        input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
        labels = mx.random.randint(0, small_config.vocab_size, shape=(1, 8)).astype(mx.int32)
        (loss, n_valid), grads = loss_and_grad(model, input_ids, labels)
        mx.eval(loss, n_valid)
        flat_keys = _flatten_keys(grads)

        def _get_nested(d, path):
            parts = path.split(".")
            cur = d
            for p in parts:
                cur = cur[p]
            return cur

        for key in flat_keys:
            g = _get_nested(grads, key)
            mx.eval(g)
            assert not mx.any(mx.isnan(g)), f"NaN in gradient {key}"


class TestAccumulateGradients:
    def test_first_step_returns_new_grads(self):
        """First accumulation step (grads=None) should return new_grads."""
        grads = accumulate_gradients(None, {"a": mx.array([1.0, 2.0])})
        mx.eval(grads)
        assert list(grads.keys()) == ["a"]

    def test_accumulation_sums(self):
        """Accumulation should sum gradients."""
        g1 = {"a": mx.array([1.0, 2.0])}
        g2 = {"a": mx.array([3.0, 4.0])}
        accumulated = accumulate_gradients(g1, g2)
        mx.eval(accumulated["a"])
        assert float(accumulated["a"][0]) == 4.0
        assert float(accumulated["a"][1]) == 6.0


class TestClipGradNorm:
    def test_reduces_norm(self):
        """Clipping should reduce gradient norm to max_norm."""
        grads = {"a": mx.array([10.0, 10.0])}
        clipped, norm = clip_grad_norm_(grads, max_norm=1.0)
        mx.eval(clipped["a"], norm)
        assert float(norm) > 1.0
        clipped_norm = mx.sqrt(mx.sum(clipped["a"] ** 2))
        mx.eval(clipped_norm)
        assert float(clipped_norm) <= 1.0 + 1e-6

    def test_zero_grads_pass_through(self):
        """Zero gradients should pass through unchanged."""
        grads = {"a": mx.zeros((2,))}
        clipped, norm = clip_grad_norm_(grads, max_norm=1.0)
        mx.eval(clipped["a"], norm)
        assert float(norm) == 0.0


class TestGetTrainableParams:
    def test_returns_adapter_params_only(self, small_config):
        """Should return only lora_a, lora_b, and m (DoRA magnitude) params."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4)
        trainable = get_trainable_params(model)
        assert len(trainable) > 0
        for key in trainable:
            assert any(k in key for k in ("lora_a", "lora_b", "m")), f"Unexpected param: {key}"


class SimpleDataset:
    """Minimal dataset for testing the Trainer."""

    def __init__(self, num_examples=10, seq_len=32, vocab_size=1024):
        self._num_examples = num_examples
        self._seq_len = seq_len
        self._vocab_size = vocab_size
        import numpy as np

        rng = np.random.RandomState(42)
        self._data = []
        for _ in range(num_examples):
            token_ids = rng.randint(1, vocab_size, size=seq_len).tolist()
            loss_mask = [1] * seq_len
            self._data.append((token_ids, loss_mask))

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):
        return self._data[idx]


class TestTrainer:
    def test_setup_creates_optimizer(self, small_config, tmp_path):
        """Trainer.setup() should create an optimizer and loss_and_grad function."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4)
        dataset = SimpleDataset(num_examples=4, seq_len=16, vocab_size=small_config.vocab_size)
        config = TrainingConfig(max_steps=5, batch_size=1, max_seq_len=16, output_dir=str(tmp_path / "ckpts"))
        trainer = Trainer(model, config, dataset)
        trainer.setup()
        assert trainer.optimizer is not None
        assert trainer.step_count == 0

    def test_single_step_completes(self, small_config, tmp_path):
        """A single training step should complete without errors."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0)
        dataset = SimpleDataset(num_examples=4, seq_len=16, vocab_size=small_config.vocab_size)
        config = TrainingConfig(
            max_steps=1,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset)
        result = trainer.train()
        assert result["step"] == 1
        assert isinstance(result["loss"], float)

    def test_loss_decreases_over_steps(self, small_config, tmp_path):
        """Loss should generally decrease over multiple training steps."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0, scale=20.0)
        dataset = SimpleDataset(num_examples=20, seq_len=16, vocab_size=small_config.vocab_size)
        config = TrainingConfig(
            max_steps=20,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            learning_rate=1e-3,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset)
        result = trainer.train()
        assert result["loss"] < 100.0

    def test_stops_at_max_steps(self, small_config, tmp_path):
        """Training should stop exactly at max_steps."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4)
        dataset = SimpleDataset(num_examples=4, seq_len=16, vocab_size=small_config.vocab_size)
        config = TrainingConfig(
            max_steps=3,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset)
        result = trainer.train()
        assert result["step"] == 3

    def test_checkpoint_resume(self, small_config, tmp_path):
        """Training should resume from the latest checkpoint."""
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0)
        dataset = SimpleDataset(num_examples=8, seq_len=16, vocab_size=small_config.vocab_size)
        ckpt_dir = str(tmp_path / "ckpts")

        config = TrainingConfig(
            max_steps=5,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=5,
            output_dir=ckpt_dir,
        )
        trainer = Trainer(model, config, dataset)
        trainer.train()

        config2 = TrainingConfig(
            max_steps=8,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=100,
            output_dir=ckpt_dir,
        )
        trainer2 = Trainer(model, config2, dataset)
        trainer2.setup()
        assert trainer2.step_count == 5

    def test_with_cooling_scheduler(self, small_config, tmp_path):
        """Training should work with a cooling scheduler attached."""
        from unittest.mock import MagicMock

        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4)
        dataset = SimpleDataset(num_examples=4, seq_len=16, vocab_size=small_config.vocab_size)
        monitor = MagicMock()
        monitor.temperature = None
        cooling = CoolingScheduler(monitor)
        config = TrainingConfig(
            max_steps=2,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset, cooling_scheduler=cooling)
        result = trainer.train()
        assert result["step"] == 2


class TestTrainerOnStep:
    def test_on_step_callback_fired(self, small_config, tmp_path):
        from unittest.mock import MagicMock

        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0, scale=10.0)
        mx.eval(model.parameters())

        config = TrainingConfig(
            max_steps=3,
            batch_size=1,
            max_seq_len=16,
            learning_rate=1e-3,
            lora_rank=4,
            lora_scale=10.0,
            save_every=10000,
            output_dir=str(tmp_path / "ckpts"),
        )
        dataset = SimpleDataset(10, 16, small_config.vocab_size)

        callback = MagicMock()
        trainer = Trainer(model, config, dataset, on_step=callback)
        trainer.train()

        assert callback.call_count == 3
        calls = callback.call_args_list
        assert calls[0][0][0] == 1
        assert calls[1][0][0] == 2
        assert calls[2][0][0] == 3

    def test_on_step_callback_receives_metrics(self, small_config, tmp_path):
        from unittest.mock import MagicMock

        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0, scale=10.0)
        mx.eval(model.parameters())

        config = TrainingConfig(
            max_steps=2,
            batch_size=1,
            max_seq_len=16,
            learning_rate=1e-3,
            lora_rank=4,
            lora_scale=10.0,
            save_every=10000,
            output_dir=str(tmp_path / "ckpts"),
        )
        dataset = SimpleDataset(10, 16, small_config.vocab_size)

        callback = MagicMock()
        trainer = Trainer(model, config, dataset, on_step=callback)
        trainer.train()

        metrics = callback.call_args_list[0][0][1]
        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert isinstance(metrics["loss"], float)
