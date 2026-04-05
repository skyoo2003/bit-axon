import mlx.nn as nn

from bit_axon.config import BitAxonConfig
from bit_axon.layers.axon_ssm import AxonSSM
from bit_axon.layers.moe import SharedExpertMoE
from bit_axon.layers.rms_norm import RMSNorm
from bit_axon.layers.swa import SlidingWindowAttention


class AxonSSMBlock(nn.Module):
    """Pure SSM block. No MLP — SSM's internal expansion serves the FFN role."""

    def __init__(self, config: BitAxonConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ssm = AxonSSM(config)

    def __call__(self, x, cache=None):
        residual = x
        x = self.input_norm(x)
        ssm_out, new_cache = self.ssm(x, cache=cache)
        return residual + ssm_out, new_cache


class AxonSWAMoEBlock(nn.Module):
    """SWA + MoE block with sliding window attention and sparse experts."""

    def __init__(self, config: BitAxonConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attention = SlidingWindowAttention(
            config.hidden_dim, config.num_heads, config.swa_window_size
        )
        self.post_attention_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.moe = SharedExpertMoE(
            config.hidden_dim,
            config.moe_intermediate_dim,
            config.moe_num_experts,
            config.moe_top_k,
        )

    def __call__(self, x, cache=None):
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
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ssm = AxonSSM(config)
        self.post_ssm_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.moe = SharedExpertMoE(
            config.hidden_dim,
            config.moe_intermediate_dim,
            config.moe_num_experts,
            config.moe_top_k,
        )

    def __call__(self, x, cache=None):
        residual = x
        x = self.input_norm(x)
        ssm_out, ssm_cache = self.ssm(x, cache=cache)
        x = residual + ssm_out
        residual = x
        x = self.post_ssm_norm(x)
        x = residual + self.moe(x)
        return x, ssm_cache
