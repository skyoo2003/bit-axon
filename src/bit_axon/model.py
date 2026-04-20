from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from bit_axon.config import BitAxonConfig
from bit_axon.layers.block import AxonSSMBlock, AxonSSMMoEBlock, AxonSWAMoEBlock
from bit_axon.utils.cache import KVCache


class BitAxonModel(nn.Module):
    """3.2B hybrid language model with SSM, SWA, and MoE layers.

    24-layer sandwich architecture:
        - Layers 0-7: Pure SSM (linear recurrence, no KV cache)
        - Layers 8-15: SWA + MoE (sliding window attention + sparse experts)
        - Layers 16-23: SSM + MoE (linear recurrence + sparse experts)

    Attributes:
        config: Model configuration.
        embed_tokens: Token embedding table.
        input_proj: Projects from source model dimension to hidden dim.
        output_proj: Projects from hidden dim back to source model dimension.
        lm_head: Output projection to vocabulary logits.
    """

    def __init__(self, config: BitAxonConfig):
        """Initialize the BitAxon model.

        Args:
            config: BitAxonConfig with architecture hyperparameters.
        """
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_source_model)
        self.input_proj = nn.Linear(config.d_source_model, config.hidden_dim, bias=False)

        for i in range(config.num_layers):
            layer_type = self._get_layer_type(i, config.num_layers)
            if layer_type == "ssm":
                layer = AxonSSMBlock(config)
            elif layer_type == "swa_moe":
                layer = AxonSWAMoEBlock(config)
            else:
                layer = AxonSSMMoEBlock(config)
            setattr(self, f"layer_{i}", layer)

        self.output_proj = nn.Linear(config.hidden_dim, config.d_source_model, bias=False)
        self.lm_head = nn.Linear(config.d_source_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _get_layer_type(layer_idx: int, total_layers: int) -> str:
        """Determine layer type based on position in the sandwich architecture.

        Args:
            layer_idx: Zero-based layer index.
            total_layers: Total number of layers.

        Returns:
            One of "ssm", "swa_moe", or "ssm_moe".
        """
        third = total_layers // 3
        if layer_idx < third:
            return "ssm"
        elif layer_idx < 2 * third:
            return "swa_moe"
        else:
            return "ssm_moe"

    def _create_caches(self) -> list[object]:
        """Create KV caches for SWA layers; None for SSM layers.

        Returns:
            List of KVCache objects for swa_moe layers and None for ssm/ssm_moe layers.
        """
        caches = []
        for i in range(self.config.num_layers):
            if self._get_layer_type(i, self.config.num_layers) == "swa_moe":
                caches.append(KVCache(window_size=self.config.swa_window_size))
            else:
                caches.append(None)
        return caches

    def __call__(self, input_ids: mx.array, cache: list | None = None) -> tuple[mx.array, list]:
        """Forward pass through all layers.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            cache: Optional list of per-layer caches from a previous call.

        Returns:
            Tuple of (logits, new_caches). Logits have shape (batch, seq_len, vocab_size).
            new_caches is a list of updated per-layer caches.
        """
        x = self.embed_tokens(input_ids)
        x = self.input_proj(x)

        # When no cache is supplied (prefill path) we still need KVCache
        # instances for the SWA layers so their K/V are persisted for
        # subsequent incremental-decode calls. Without this, prefill runs
        # SWA with cache=None, the layer holds no KVCache, and the very
        # next decode call starts from an empty context — prefill and
        # step-by-step decode diverge at the model level (~O(1) diff).
        if cache is None:
            cache = self._create_caches()

        new_caches = []
        for i in range(self.config.num_layers):
            layer = getattr(self, f"layer_{i}")
            layer_cache = cache[i]
            x, layer_cache_out = layer(x, cache=layer_cache)
            new_caches.append(layer_cache_out)

        x = self.output_proj(x)
        logits = self.lm_head(x)
        return logits, new_caches
