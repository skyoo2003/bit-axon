from bit_axon.training.config import TrainingConfig


class TestTrainingConfig:
    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == 1e-4
        assert cfg.weight_decay == 0.01
        assert cfg.warmup_steps == 100
        assert cfg.max_steps == 10_000
        assert cfg.max_grad_norm == 1.0
        assert cfg.grad_accum_steps == 4
        assert cfg.lora_rank == 8
        assert cfg.lora_dropout == 0.0
        assert cfg.lora_scale == 20.0
        assert cfg.use_dora is True
        assert cfg.quantize_bits == 4
        assert cfg.quantize_group_size == 64
        assert cfg.batch_size == 1
        assert cfg.max_seq_len == 2048
        assert cfg.save_every == 500
        assert cfg.eval_every == 500
        assert cfg.output_dir == "checkpoints"
        assert cfg.temp_max_speed == 75.0
        assert cfg.temp_pause == 85.0
        assert cfg.temp_stop == 95.0
        assert cfg.temp_poll_interval == 1.0
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = TrainingConfig(learning_rate=5e-5, lora_rank=16, seed=123)
        assert cfg.learning_rate == 5e-5
        assert cfg.lora_rank == 16
        assert cfg.seed == 123

    def test_lora_targets_is_tuple(self):
        cfg = TrainingConfig()
        assert isinstance(cfg.lora_targets, tuple)
        assert all(isinstance(t, str) for t in cfg.lora_targets)
        assert len(cfg.lora_targets) == 11

    def test_thermal_thresholds_order(self):
        cfg = TrainingConfig()
        assert cfg.temp_max_speed < cfg.temp_pause < cfg.temp_stop

    def test_positive_learning_rate(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate > 0

    def test_max_seq_len_power_of_two(self):
        cfg = TrainingConfig()
        assert cfg.max_seq_len & (cfg.max_seq_len - 1) == 0

    def test_grad_accum_steps_positive(self):
        cfg = TrainingConfig()
        assert cfg.grad_accum_steps >= 1

    def test_quantize_bits_valid(self):
        cfg = TrainingConfig()
        assert cfg.quantize_bits in {2, 3, 4, 8}

    def test_lora_rank_positive(self):
        cfg = TrainingConfig()
        assert cfg.lora_rank >= 1

    def test_seed_reproducible(self):
        cfg1 = TrainingConfig(seed=42)
        cfg2 = TrainingConfig(seed=42)
        assert cfg1.seed == cfg2.seed
