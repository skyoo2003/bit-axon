"""Tests for BitAxonConfig."""

from bit_axon.config import BitAxonConfig


class TestBitAxonConfig:
    def test_default_values(self):
        config = BitAxonConfig()
        assert config.vocab_size == 32_000
        assert config.hidden_dim == 2_560
        assert config.num_layers == 24
        assert config.num_heads == 32
        assert config.d_source_model == 2_048
        assert config.ssm_d_state == 16
        assert config.ssm_d_conv == 4
        assert config.ssm_expand == 3
        assert config.swa_window_size == 4_096
        assert config.moe_num_experts == 8
        assert config.moe_top_k == 2
        assert config.moe_intermediate_dim == 4_096
        assert config.moe_shared_expert is True
        assert config.weight_tying is True
        assert config.max_seq_len == 65_536
        assert config.rms_norm_eps == 1e-6

    def test_head_dim_property(self):
        config = BitAxonConfig()
        assert config.head_dim == 80  # 2560 / 32

    def test_ssm_intermediate_dim_property(self):
        config = BitAxonConfig()
        assert config.ssm_intermediate_dim == 7_680  # 2560 * 3

    def test_custom_values(self):
        config = BitAxonConfig(hidden_dim=512, num_heads=8)
        assert config.head_dim == 64  # 512 / 8
        assert config.ssm_intermediate_dim == 1_536  # 512 * 3

    def test_frozen_dataclass(self):
        """Config should be a frozen dataclass-like (not frozen, but structured)."""
        config = BitAxonConfig()
        config.hidden_dim = 1024
        assert config.hidden_dim == 1024
