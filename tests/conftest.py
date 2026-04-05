import mlx.core as mx
import pytest

from bit_axon.config import BitAxonConfig


@pytest.fixture
def default_config() -> BitAxonConfig:
    """Default model configuration for testing."""
    return BitAxonConfig()


@pytest.fixture
def small_config() -> BitAxonConfig:
    """Small configuration for fast unit tests."""
    return BitAxonConfig(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        d_source_model=128,
        vocab_size=1024,
        ssm_d_state=4,
        ssm_d_conv=2,
        ssm_expand=2,
        swa_window_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_dim=512,
    )


@pytest.fixture
def batch_input(default_config: BitAxonConfig) -> mx.array:
    """Random batch of token IDs (batch=1, seq_len=128)."""
    return mx.random.randint(
        0, default_config.vocab_size, shape=(1, 128), dtype=mx.uint32
    )


@pytest.fixture
def hidden_states(default_config: BitAxonConfig) -> mx.array:
    """Random hidden states (batch=1, seq_len=128, hidden_dim=2560)."""
    return mx.random.normal(shape=(1, 128, default_config.hidden_dim))
