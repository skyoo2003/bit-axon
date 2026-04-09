"""NF4 (4-bit NormalFloat) quantization for MLX models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def quantize_nf4(weight: mx.array, group_size: int = 64) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize a weight matrix to NF4 format.

    Args:
        weight: Weight matrix to quantize.
        group_size: Number of elements per quantization group.

    Returns:
        Tuple of (packed_weights, scales, biases).
    """
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
    return packed, scales, biases


def dequantize_nf4(packed, scales, biases, group_size: int = 64, bits: int = 4) -> mx.array:
    """Dequantize NF4 packed weights back to full precision.

    Args:
        packed: Packed quantized weights.
        scales: Per-group scale factors.
        biases: Per-group bias values.
        group_size: Number of elements per quantization group.
        bits: Bit width used during quantization.

    Returns:
        Dequantized weight matrix.
    """
    return mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)


def _replace_submodules(
    mod: nn.Module,
    target_cls: type,
    can_quantize_fn: Callable[[Any], bool],
    replace_fn: Callable[[Any], nn.Module],
) -> None:
    for name, child in mod.children().items():
        if isinstance(child, list):
            new_list = []
            for item in child:
                if isinstance(item, target_cls) and can_quantize_fn(item):
                    new_list.append(replace_fn(item))
                else:
                    if isinstance(item, nn.Module):
                        _replace_submodules(item, target_cls, can_quantize_fn, replace_fn)
                    new_list.append(item)
            setattr(mod, name, new_list)
        elif isinstance(child, target_cls):
            if can_quantize_fn(child):
                setattr(mod, name, replace_fn(child))
        elif isinstance(child, nn.Module):
            _replace_submodules(child, target_cls, can_quantize_fn, replace_fn)


def replace_linear_with_quantized(module: nn.Module, group_size: int = 64, bits: int = 4) -> nn.Module:
    """Recursively replace nn.Linear layers with nn.QuantizedLinear.

    Skips layers whose input dimension is smaller than group_size.

    Args:
        module: Root module to traverse.
        group_size: Quantization group size.
        bits: Quantization bit width.

    Returns:
        The modified module (mutated in-place).
    """

    def _can_quantize(linear: nn.Linear) -> bool:
        if linear.weight is None:
            return False
        return linear.weight.shape[-1] >= group_size

    def _replace(linear: nn.Linear) -> nn.QuantizedLinear:
        return nn.QuantizedLinear.from_linear(linear, group_size=group_size, bits=bits)

    _replace_submodules(module, nn.Linear, _can_quantize, _replace)
    return module


def replace_switch_linear_with_quantized(module: nn.Module, group_size: int = 64, bits: int = 4) -> nn.Module:
    from bit_axon.layers.moe import QuantizedSwitchLinear, SwitchLinear

    def _can_quantize(sl: SwitchLinear) -> bool:
        return sl.weight.shape[-1] >= group_size

    def _replace(sl: SwitchLinear) -> QuantizedSwitchLinear:
        return QuantizedSwitchLinear.from_switch_linear(sl, group_size=group_size, bits=bits)

    _replace_submodules(module, SwitchLinear, _can_quantize, _replace)
    return module
