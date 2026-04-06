import json
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import SGD

from bit_axon.model import BitAxonModel
from bit_axon.training.checkpoint import (
    get_latest_checkpoint,
    load_checkpoint,
    save_adapter_only,
    save_checkpoint,
)
from bit_axon.training.lora import apply_lora_to_model


class TestSaveAndLoadCheckpoint:
    def test_roundtrip_preserves_weights(self, small_config, tmp_path):
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4, dropout=0.0)

        optimizer = SGD(learning_rate=1e-4)
        input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)
        labels = mx.random.randint(0, small_config.vocab_size, shape=(1, 8), dtype=mx.uint32)

        def compute_loss(model):
            logits, _ = model(input_ids)
            return mx.mean(nn.losses.cross_entropy(logits, labels))

        loss, grads = mx.value_and_grad(compute_loss)(model)
        mx.eval(loss)
        optimizer.update(model, grads)

        ckpt_dir = save_checkpoint(model, optimizer, step=100, loss=2.5, output_dir=tmp_path / "ckpts")
        assert ckpt_dir.exists()

        model2 = BitAxonModel(small_config)
        mx.eval(model2.parameters())
        apply_lora_to_model(model2, rank=4, dropout=0.0)
        optimizer2 = SGD(learning_rate=1e-4)
        step, loss_val = load_checkpoint(model2, optimizer2, ckpt_dir)

        assert step == 100
        assert loss_val == 2.5

    def test_training_state_json(self, tmp_path):
        model = nn.Linear(4, 2)
        mx.eval(model.parameters())
        optimizer = SGD(learning_rate=1e-4)

        ckpt_dir = save_checkpoint(model, optimizer, step=50, loss=3.14, output_dir=tmp_path / "ckpts")
        state_file = ckpt_dir / "training_state.json"
        assert state_file.exists()
        with open(state_file) as f:
            state = json.load(f)
        assert state["step"] == 50
        assert state["loss"] == 3.14


class TestGetLatestCheckpoint:
    def test_finds_latest(self, tmp_path):
        model = nn.Linear(4, 2)
        mx.eval(model.parameters())
        optimizer = SGD(learning_rate=1e-4)

        save_checkpoint(model, optimizer, step=100, loss=1.0, output_dir=tmp_path / "ckpts")
        save_checkpoint(model, optimizer, step=300, loss=0.5, output_dir=tmp_path / "ckpts")
        save_checkpoint(model, optimizer, step=200, loss=0.8, output_dir=tmp_path / "ckpts")

        latest = get_latest_checkpoint(tmp_path / "ckpts")
        assert latest is not None
        assert "step_00000300" in str(latest)

    def test_returns_none_when_empty(self, tmp_path):
        latest = get_latest_checkpoint(tmp_path / "nonexistent")
        assert latest is None


class TestCheckpointRotation:
    def test_deletes_old_checkpoints(self, tmp_path):
        model = nn.Linear(4, 2)
        mx.eval(model.parameters())
        optimizer = SGD(learning_rate=1e-4)

        for step in [100, 200, 300, 400]:
            save_checkpoint(model, optimizer, step=step, loss=1.0, output_dir=tmp_path / "ckpts", max_checkpoints=2)

        checkpoints = list((tmp_path / "ckpts").glob("step_*"))
        assert len(checkpoints) == 2
        steps = sorted([int(c.name.split("_")[1]) for c in checkpoints])
        assert steps == [300, 400]


class TestSaveAdapterOnly:
    def test_saves_only_trainable_params(self, small_config, tmp_path):
        model = BitAxonModel(small_config)
        mx.eval(model.parameters())
        apply_lora_to_model(model, rank=4)

        output_path = tmp_path / "adapter.safetensors"
        save_adapter_only(model, output_path)
        assert output_path.exists()

        loaded = cast(dict[str, mx.array], mx.load(str(output_path)))
        for key in loaded:
            assert any(k in key for k in ("lora_a", "lora_b", "m")), f"Unexpected key: {key}"
