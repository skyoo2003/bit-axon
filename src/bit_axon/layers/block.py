from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from bit_axon.config import BitAxonConfig
from bit_axon.layers.mamba3 import Mamba3
from bit_axon.layers.moe import SharedExpertMoE
from bit_axon.layers.rms_norm import RMSNorm
from bit_axon.layers.swa import SlidingWindowAttention


def _build_mamba3(config: BitAxonConfig) -> Mamba3:
    """Construct a Mamba-3 block from the BitAxonConfig's mamba3_* fields.

    Centralized so both AxonSSMBlock and AxonSSMMoEBlock share one
    parameter-mapping and we only need to touch this function if we add
    more config knobs.
    """
    return Mamba3(
        d_model=config.hidden_dim,
        d_state=config.mamba3_d_state,
        expand=config.mamba3_expand,
        headdim=config.mamba3_headdim,
        ngroups=config.mamba3_ngroups,
        rope_fraction=config.mamba3_rope_fraction,
        dt_min=config.mamba3_dt_min,
        dt_max=config.mamba3_dt_max,
        dt_init_floor=config.mamba3_dt_init_floor,
        a_floor=config.mamba3_a_floor,
        d_conv=config.mamba3_d_conv,
        chunk_size=config.mamba3_chunk_size,
        is_mimo=config.mamba3_is_mimo,
        mimo_rank=config.mamba3_mimo_rank,
    )


class AxonSSMBlock(nn.Module):
    """Pure SSM block. No MLP — SSM's internal expansion serves the FFN role."""

    def __init__(self, config: BitAxonConfig):
        """Initialize the SSM block.

        Args:
            config: BitAxonConfig with hidden_dim and rms_norm_eps settings.
        """
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ssm = _build_mamba3(config)

    def __call__(self, x: mx.array, cache: list | None = None) -> tuple[mx.array, list | None]:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            cache: Optional SSM cache from a previous step.

        Returns:
            Tuple of (output, cache). Output has shape (batch, seq_len, hidden_dim).
        """
        residual = x
        x = self.input_norm(x)
        ssm_out, new_cache = self.ssm(x, cache=cache)
        return residual + ssm_out, new_cache


class AxonSWAMoEBlock(nn.Module):
    """SWA + MoE block with sliding window attention and sparse experts."""

    def __init__(self, config: BitAxonConfig):
        """Initialize the SWA + MoE block.

        Args:
            config: BitAxonConfig with attention and MoE hyperparameters.
        """
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attention = SlidingWindowAttention(config.hidden_dim, config.num_heads, config.swa_window_size)
        self.post_attention_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.moe = SharedExpertMoE(
            config.hidden_dim,
            config.moe_intermediate_dim,
            config.moe_num_experts,
            config.moe_top_k,
        )

    def __call__(self, x: mx.array, cache: list | None = None) -> tuple[mx.array, list | None]:
        """Forward pass: attention with residual, then MoE with residual.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            cache: Optional KVCache for autoregressive decoding.

        Returns:
            Tuple of (output, cache). Output has shape (batch, seq_len, hidden_dim).
        """
        residual = x
        x = self.input_norm(x)
        attn_out, new_cache = self.attention(x, cache=cache)
        x = residual + attn_out
        residual = x
        x = self.post_attention_norm(x)
        x = residual + self.moe(x)
        return x, new_cache


class AxonSSMMoEBlock(nn.Module):
    """SSM + MoE block with linear recurrence and sparse experts."""

    def __init__(self, config: BitAxonConfig):
        """Initialize the SSM + MoE block.

        Args:
            config: BitAxonConfig with SSM and MoE hyperparameters.
        """
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ssm = _build_mamba3(config)
        self.post_ssm_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.moe = SharedExpertMoE(
            config.hidden_dim,
            config.moe_intermediate_dim,
            config.moe_num_experts,
            config.moe_top_k,
        )

    def __call__(self, x: mx.array, cache: list | None = None) -> tuple[mx.array, list | None]:
        """Forward pass: SSM with residual, then MoE with residual.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            cache: Optional SSM cache from a previous step.

        Returns:
            Tuple of (output, ssm_cache). Output has shape (batch, seq_len, hidden_dim).
        """
        residual = x
        x = self.input_norm(x)
        ssm_out, ssm_cache = self.ssm(x, cache=cache)
        x = residual + ssm_out
        residual = x
        x = self.post_ssm_norm(x)
        x = residual + self.moe(x)
        return x, ssm_cache
