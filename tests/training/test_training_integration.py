"""End-to-end integration tests for the full QLoRA SFT training pipeline."""

import mlx.core as mx

from bit_axon.model import BitAxonModel
from bit_axon.quantization.nf4 import replace_linear_with_quantized
from bit_axon.training.config import TrainingConfig
from bit_axon.training.lora import apply_lora_to_model
from bit_axon.training.trainer import Trainer, get_trainable_params


class SimpleDataset:
    """Minimal dataset that yields (token_ids, loss_mask) tuples without tokenizer."""

    def __init__(self, num_examples=20, seq_len=32, vocab_size=1024):
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


class TestQLoRATrainingPipeline:
    def test_qlora_pipeline_loss_bounded(self, small_config, tmp_path):
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        replace_linear_with_quantized(model, group_size=64, bits=4)
        apply_lora_to_model(model, rank=4, dropout=0.0, scale=20.0)

        dataset = SimpleDataset(num_examples=20, seq_len=16, vocab_size=small_config.vocab_size)
        config = TrainingConfig(
            max_steps=10,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            learning_rate=1e-3,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset)
        result = trainer.train()
        assert result["step"] == 10
        assert result["loss"] < 100.0

    def test_dora_pipeline_completes(self, small_config, tmp_path):
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, use_dora=True, dropout=0.0, scale=20.0)

        dataset = SimpleDataset(num_examples=10, seq_len=16, vocab_size=small_config.vocab_size)
        config = TrainingConfig(
            max_steps=5,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            use_dora=True,
            save_every=100,
            output_dir=str(tmp_path / "ckpts"),
        )
        trainer = Trainer(model, config, dataset)
        result = trainer.train()
        assert result["step"] == 5

    def test_checkpoint_save_and_resume(self, small_config, tmp_path):
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0)

        dataset = SimpleDataset(num_examples=20, seq_len=16, vocab_size=small_config.vocab_size)
        ckpt_dir = str(tmp_path / "ckpts")

        config1 = TrainingConfig(
            max_steps=5,
            grad_accum_steps=1,
            batch_size=1,
            max_seq_len=16,
            save_every=5,
            output_dir=ckpt_dir,
        )
        trainer1 = Trainer(model, config1, dataset)
        result1 = trainer1.train()
        assert result1["step"] == 5

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

    def test_adapter_weights_change_after_training(self, small_config, tmp_path):
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0, scale=20.0)

        initial_params = get_trainable_params(model)
        initial_lora_a = {}
        for key, val in initial_params.items():
            if "lora_a" in key:
                mx.eval(val)
                initial_lora_a[key] = val

        dataset = SimpleDataset(num_examples=10, seq_len=16, vocab_size=small_config.vocab_size)
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
        trainer.train()

        final_params = get_trainable_params(model)
        any_changed = False
        for key in initial_lora_a:
            if key in final_params:
                mx.eval(final_params[key])
                diff = mx.abs(final_params[key] - initial_lora_a[key]).max()
                mx.eval(diff)
                if float(diff) > 1e-8:
                    any_changed = True
                    break
        assert any_changed, "LoRA A weights should change after training"
