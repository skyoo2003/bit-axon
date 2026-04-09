"""Tests for ORPO preference alignment trainer."""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn

from bit_axon.training.config import TrainingConfig
from bit_axon.training.orpo_trainer import ORPOTrainer


class SimpleLM(nn.Module):
    """Tiny language model for testing."""

    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, input_ids):
        x = self.embed(input_ids)
        return self.linear(x)


class MockORPODataset:
    """Mock ORPO dataset yielding (chosen_ids, chosen_mask, rejected_ids, rejected_mask) tuples."""

    def __init__(self, n=20, seq_len=8, vocab_size=50):
        rng = random.Random(42)
        self.data = []
        for _ in range(n):
            prompt_len = 3
            response_len = seq_len - prompt_len
            prompt_ids = [rng.randint(1, vocab_size - 1) for _ in range(prompt_len)]
            chosen_response = [rng.randint(1, vocab_size - 1) for _ in range(response_len)]
            rejected_response = [rng.randint(1, vocab_size - 1) for _ in range(response_len)]
            chosen_ids = prompt_ids + chosen_response
            rejected_ids = prompt_ids + rejected_response
            chosen_mask = [0] * prompt_len + [1] * response_len
            rejected_mask = [0] * prompt_len + [1] * response_len
            self.data.append((chosen_ids, chosen_mask, rejected_ids, rejected_mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MockCoolingScheduler:
    def __init__(self):
        self.calls = 0

    def check_before_step(self, step):
        self.calls += 1


def _make_config(tmp_path, **overrides):
    params = {
        "max_steps": 5,
        "batch_size": 2,
        "max_seq_len": 8,
        "save_every": 100,
        "eval_every": 100,
        "grad_accum_steps": 1,
        "beta": 0.1,
        "training_mode": "orpo",
        "output_dir": str(tmp_path / "checkpoints"),
        "learning_rate": 1e-3,
        "warmup_steps": 0,
    }
    params.update(overrides)
    return TrainingConfig(**params)


class TestORPOTrainerSetup:
    def test_creates_optimizer(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path)
        trainer = ORPOTrainer(model, config, dataset)
        trainer.setup()
        assert trainer.optimizer is not None

    def test_loss_and_grad_exists(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path)
        trainer = ORPOTrainer(model, config, dataset)
        trainer.setup()
        assert callable(trainer._loss_and_grad)

    def test_step_count_init(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path)
        trainer = ORPOTrainer(model, config, dataset)
        assert trainer.step_count == 0


class TestORPOTrainerStep:
    def test_single_step(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path)
        trainer = ORPOTrainer(model, config, dataset)
        trainer.setup()

        from bit_axon.training.orpo_collate import iterate_orpo_batches

        batch_iter = iterate_orpo_batches(dataset, batch_size=2, max_seq_len=8, shuffle=True, loop=True, seed=42)
        batch = next(batch_iter)
        (loss, metrics), grads = trainer._loss_and_grad(model, *batch)
        mx.eval(loss)
        assert mx.isfinite(loss).all()

    def test_loss_components(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path)
        trainer = ORPOTrainer(model, config, dataset)
        trainer.setup()

        from bit_axon.training.orpo_collate import iterate_orpo_batches

        batch_iter = iterate_orpo_batches(dataset, batch_size=2, max_seq_len=8, shuffle=True, loop=True, seed=42)
        batch = next(batch_iter)
        (loss, metrics), grads = trainer._loss_and_grad(model, *batch)
        mx.eval(loss)
        assert "nll_loss" in metrics
        assert "orpo_loss" in metrics
        assert "chosen_logps" in metrics
        assert "rejected_logps" in metrics
        assert "reward_margin" in metrics


class TestORPOTrainerMetrics:
    def test_reward_margin_tracked(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path, max_steps=5)
        trainer = ORPOTrainer(model, config, dataset)
        result = trainer.train()
        assert "reward_margin" in result
        assert isinstance(result["reward_margin"], float)

    def test_return_dict_keys(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path, max_steps=3)
        trainer = ORPOTrainer(model, config, dataset)
        result = trainer.train()
        expected_keys = {"step", "loss", "grad_norm", "chosen_reward", "rejected_reward", "reward_margin", "reward_accuracy"}
        assert set(result.keys()) == expected_keys


class TestORPOTrainerIntegration:
    def test_full_train_loop(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path, max_steps=5)
        trainer = ORPOTrainer(model, config, dataset)
        result = trainer.train()
        assert result["step"] == 5
        assert isinstance(result["loss"], float)
        assert "reward_margin" in result

    def test_thermal_gating(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path, max_steps=3)
        cooling = MockCoolingScheduler()
        trainer = ORPOTrainer(model, config, dataset, cooling_scheduler=cooling)
        trainer.train()
        assert cooling.calls == 3


class TestORPOTrainerOnStep:
    def test_on_step_called_correct_number_of_times(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path, max_steps=5)
        callback = MagicMock()
        trainer = ORPOTrainer(model, config, dataset, on_step=callback)
        trainer.train()
        assert callback.call_count == 5

    def test_on_step_receives_step_and_metrics(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path, max_steps=3)
        callback = MagicMock()
        trainer = ORPOTrainer(model, config, dataset, on_step=callback)
        trainer.train()
        for call in callback.call_args_list:
            step, metrics = call.args
            assert isinstance(step, int)
            assert isinstance(metrics, dict)
            assert "loss" in metrics
            assert "grad_norm" in metrics
            assert "chosen_logps" in metrics
            assert "rejected_logps" in metrics
            assert "reward_margin" in metrics

    def test_on_step_step_values_increment(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path, max_steps=4)
        callback = MagicMock()
        trainer = ORPOTrainer(model, config, dataset, on_step=callback)
        trainer.train()
        steps = [call.args[0] for call in callback.call_args_list]
        assert steps == [1, 2, 3, 4]

    def test_no_callback_backward_compat(self, tmp_path):
        model = SimpleLM(vocab_size=50, hidden_dim=16)
        mx.eval(model.parameters())
        dataset = MockORPODataset(n=10, seq_len=8, vocab_size=50)
        config = _make_config(tmp_path, max_steps=3)
        trainer = ORPOTrainer(model, config, dataset)
        result = trainer.train()
        assert result["step"] == 3
        assert isinstance(result["loss"], float)
