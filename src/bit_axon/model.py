import mlx.core as mx
import mlx.nn as nn

from bit_axon.config import BitAxonConfig
from bit_axon.layers.block import AxonSSMBlock, AxonSSMMoEBlock, AxonSWAMoEBlock
from bit_axon.utils.cache import KVCache


class BitAxonModel(nn.Module):
    def __init__(self, config: BitAxonConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_source_model)
        self.input_proj = nn.Linear(
            config.d_source_model, config.hidden_dim, bias=False
        )

        for i in range(config.num_layers):
            layer_type = self._get_layer_type(i, config.num_layers)
            if layer_type == "ssm":
                layer = AxonSSMBlock(config)
            elif layer_type == "swa_moe":
                layer = AxonSWAMoEBlock(config)
            else:
                layer = AxonSSMMoEBlock(config)
            setattr(self, f"layer_{i}", layer)

        self.output_proj = nn.Linear(
            config.hidden_dim, config.d_source_model, bias=False
        )
        self.lm_head = nn.Linear(config.d_source_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _get_layer_type(layer_idx: int, total_layers: int) -> str:
        third = total_layers // 3
        if layer_idx < third:
            return "ssm"
        elif layer_idx < 2 * third:
            return "swa_moe"
        else:
            return "ssm_moe"

    def _create_caches(self) -> list[object]:
        caches = []
        for i in range(self.config.num_layers):
            if self._get_layer_type(i, self.config.num_layers) == "swa_moe":
                caches.append(KVCache())
            else:
                caches.append(None)
        return caches

    def __call__(self, input_ids: mx.array, cache=None):
        x = self.embed_tokens(input_ids)
        x = self.input_proj(x)

        new_caches = []
        for i in range(self.config.num_layers):
            layer = getattr(self, f"layer_{i}")
            layer_cache = cache[i] if cache is not None else None
            x, layer_cache_out = layer(x, cache=layer_cache)
            new_caches.append(layer_cache_out)

        x = self.output_proj(x)
        logits = self.lm_head(x)
        return logits, new_caches
