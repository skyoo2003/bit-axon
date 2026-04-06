"""Model loading utilities for Bit-Axon inference."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel


def load_model(
    weights_path: str | Path,
    config: BitAxonConfig | None = None,
    quantize: bool = False,
    bits: int = 4,
    group_size: int = 64,
) -> BitAxonModel:
    weights_path = Path(weights_path)

    if config is None:
        config_path = weights_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = BitAxonConfig(**config_dict)
        else:
            config = BitAxonConfig()

    model = BitAxonModel(config)

    weights: dict[str, mx.array] = {}
    for sf_file in sorted(weights_path.glob("*.safetensors")):
        weights.update(mx.load(str(sf_file)))

    if weights:
        model.update(tree_unflatten(list(weights.items())))

    mx.eval(model.parameters())

    if quantize:
        from bit_axon.quantization.nf4 import replace_linear_with_quantized

        replace_linear_with_quantized(model, group_size=group_size, bits=bits)

    return model
