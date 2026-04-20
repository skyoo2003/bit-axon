"""Bit-Axon 3B model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BitAxonConfig:
    """Bit-Axon 3B model configuration.

    Hybrid SSM + MoE + Quantized architecture for Apple Silicon.
    Target: MacBook Air M4 (16GB unified memory, ~8GB available for model).
    """

    # Model dimensions
    vocab_size: int = 32_000
    hidden_dim: int = 2_560  # d_model
    num_layers: int = 24
    num_heads: int = 32  # SWA heads, head_dim=80 (2560/32)
    d_source_model: int = 2048  # Qwen2.5-3B bridge dimension

    # Axon-SSM (Mamba-style State Space Model)
    ssm_d_state: int = 16  # state vector dimension
    ssm_d_conv: int = 4  # 1D conv kernel size
    ssm_expand: int = 3  # expansion ratio
    ssm_scan_step: int = 64  # chunk size for parallel scan

    # Sliding Window Attention (Layer 9-16 only)
    swa_window_size: int = 4_096  # sliding window

    # Shared-Expert MoE
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_intermediate_dim: int = 4_096  # expert FFN dim
    moe_shared_expert: bool = True

    # General
    weight_tying: bool = True  # embedding = output head
    max_seq_len: int = 65_536  # max 64K context
    rms_norm_eps: float = 1e-6

    # Mamba-3 block parameters (arXiv:2603.15569).
    # Consumed by bit_axon.layers.mamba3.Mamba3 (replaces AxonSSM once
    # task #16 lands). Invariants: (mamba3_expand * hidden_dim) must be
    # divisible by mamba3_headdim; (mamba3_d_state * mamba3_rope_fraction)
    # must be even.
    mamba3_d_state: int = 128
    mamba3_headdim: int = 64
    mamba3_expand: int = 2
    mamba3_ngroups: int = 1
    mamba3_rope_fraction: float = 0.5
    mamba3_chunk_size: int = 64
    mamba3_d_conv: int = 4
    mamba3_dt_min: float = 0.001
    mamba3_dt_max: float = 0.1
    mamba3_dt_init_floor: float = 1e-4
    mamba3_a_floor: float = 1e-4
    mamba3_is_mimo: bool = False
    mamba3_mimo_rank: int = 4

    @property
    def head_dim(self) -> int:
        """SWA head dimension (hidden_dim / num_heads)."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")
        return self.hidden_dim // self.num_heads

    @property
    def ssm_intermediate_dim(self) -> int:
        """SSM expanded dimension (hidden_dim * ssm_expand)."""
        return self.hidden_dim * self.ssm_expand

    @classmethod
    def small(cls) -> BitAxonConfig:
        # d_inner = mamba3_expand * hidden_dim = 2 * 256 = 512; headdim=32 →
        # nheads=16. d_state*rope_fraction = 32*0.5 = 16 (even).
        return cls(
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
            mamba3_d_state=32,
            mamba3_headdim=32,
        )

    @classmethod
    def medium(cls) -> BitAxonConfig:
        return cls(
            hidden_dim=2048,
            num_layers=12,
            num_heads=16,
            d_source_model=1536,
            vocab_size=32000,
            ssm_d_state=12,
            ssm_d_conv=4,
            ssm_expand=3,
            swa_window_size=4096,
            moe_num_experts=6,
            moe_top_k=2,
            moe_intermediate_dim=3072,
        )

    @classmethod
    def large(cls) -> BitAxonConfig:
        """Full-size Bit-Axon 3.2B config (production default)."""
        return cls()
