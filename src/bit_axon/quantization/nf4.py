import mlx.core as mx
import mlx.nn as nn


def quantize_nf4(weight: mx.array, group_size: int = 64) -> tuple[mx.array, mx.array, mx.array]:
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
    return packed, scales, biases


def dequantize_nf4(packed, scales, biases, group_size: int = 64, bits: int = 4) -> mx.array:
    return mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)


def replace_linear_with_quantized(module: nn.Module, group_size: int = 64, bits: int = 4):
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
