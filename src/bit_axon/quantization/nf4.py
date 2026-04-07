"""NF4 (4-bit NormalFloat) quantization for MLX models."""

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


def replace_linear_with_quantized(module: nn.Module, group_size: int = 64, bits: int = 4):
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
        input_dim = linear.weight.shape[-1]
        return input_dim >= group_size

    def _replace_submodules(mod):
        for name, child in mod.children().items():
            if isinstance(child, list):
                new_list = []
                for item in child:
                    if isinstance(item, nn.Linear) and _can_quantize(item):
                        new_list.append(nn.QuantizedLinear.from_linear(item, group_size=group_size, bits=bits))
                    else:
                        if isinstance(item, nn.Module):
                            _replace_submodules(item)
                        new_list.append(item)
                setattr(mod, name, new_list)
            elif isinstance(child, nn.Linear):
                if _can_quantize(child):
                    setattr(
                        mod,
                        name,
                        nn.QuantizedLinear.from_linear(child, group_size=group_size, bits=bits),
                    )
            elif isinstance(child, nn.Module):
                _replace_submodules(child)

    _replace_submodules(module)
    return module
