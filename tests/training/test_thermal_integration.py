"""Integration tests for thermal-gated training."""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from bit_axon.model import BitAxonModel
from bit_axon.training.config import TrainingConfig
from bit_axon.training.cooling import CoolingScheduler, ThermalShutdownError
from bit_axon.training.lora import apply_lora_to_model
from bit_axon.training.trainer import Trainer


class SimpleDataset:
    """Minimal dataset for testing."""

    def __init__(self, num_examples=10, seq_len=16, vocab_size=1024):
        self._num_examples = num_examples
        import numpy as np

        rng = np.random.RandomState(42)
        self._data = []
        for _ in range(num_examples):
            token_ids = rng.randint(1, vocab_size, size=seq_len).tolist()
            self._data.append((token_ids, [1] * seq_len))

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):
        return self._data[idx]


class TestThermalTrainingIntegration:
    """Thermal-gated training with mock ThermalMonitor."""

    @staticmethod
    def _make_model(small_config):
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0)
        return model

    @staticmethod
    def _make_dataset(vocab_size=1024):
        return SimpleDataset(num_examples=10, seq_len=16, vocab_size=vocab_size)

    def test_training_with_cooling_none_temp(self, small_config, tmp_path):
        """Training with cooling scheduler but None temperature (CI mode) should complete."""
        model = self._make_model(small_config)
        dataset = self._make_dataset(small_config.vocab_size)

        monitor = MagicMock()
        monitor.temperature = None
        cooling = CoolingScheduler(monitor)

        config = TrainingConfig(
            max_steps=3,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset, cooling_scheduler=cooling)
        result = trainer.train()
        assert result["step"] == 3

    def test_training_with_cooling_normal_temp(self, small_config, tmp_path):
        """Training with normal temperature (below thresholds) should complete."""
        model = self._make_model(small_config)
        dataset = self._make_dataset(small_config.vocab_size)

        monitor = MagicMock()
        monitor.temperature = 50.0
        cooling = CoolingScheduler(monitor)

        config = TrainingConfig(
            max_steps=3,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset, cooling_scheduler=cooling)
        result = trainer.train()
        assert result["step"] == 3

    def test_training_shutdown_on_overheat(self, small_config, tmp_path):
        """Training should raise ThermalShutdownError when temp exceeds stop threshold."""
        model = self._make_model(small_config)
        dataset = self._make_dataset(small_config.vocab_size)

        monitor = MagicMock()
        monitor.temperature = 96.0
        cooling = CoolingScheduler(monitor)

        config = TrainingConfig(
            max_steps=100,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset, cooling_scheduler=cooling)
        with pytest.raises(ThermalShutdownError):
            trainer.train()

    def test_training_without_cooling_scheduler(self, small_config, tmp_path):
        """Training without cooling scheduler should work normally."""
        model = self._make_model(small_config)
        dataset = self._make_dataset(small_config.vocab_size)

        config = TrainingConfig(
            max_steps=3,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset, cooling_scheduler=None)
        result = trainer.train()
        assert result["step"] == 3
