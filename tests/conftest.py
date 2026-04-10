import os

import mlx.core as mx
import pytest

from bit_axon.config import BitAxonConfig


def pytest_configure(config: pytest.Config) -> None:
    """Set environment variables before test collection to prevent Rich/tqdm threading crashes.

    Rich's ``console.status()`` spawns a background ``Live`` thread, and the ``datasets``
    library (used by some CLI commands) triggers ``tqdm``'s ``_monitor`` thread.  On
    macOS / CPython 3.10 these threads race on the redirected stdout inside Typer's
    ``CliRunner``, producing a SIGABRT (exit 134).  Setting these vars at *session*
    scope—before any production module is imported—avoids the crash both locally and
    in CI.
    """
    os.environ.setdefault("TERM", "dumb")
    os.environ.setdefault("NO_COLOR", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")


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
    return mx.random.randint(0, default_config.vocab_size, shape=(1, 128), dtype=mx.uint32)


@pytest.fixture
def hidden_states(default_config: BitAxonConfig) -> mx.array:
    """Random hidden states (batch=1, seq_len=128, hidden_dim=2560)."""
    return mx.random.normal(shape=(1, 128, default_config.hidden_dim))
