import math

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, input_dims, output_dims, r=8, dropout=0.0, scale=20.0, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        init_scale = 1.0 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-init_scale, high=init_scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

    @staticmethod
    def from_base(linear, r=8, dropout=0.0, scale=20.0):
        if isinstance(linear, nn.QuantizedLinear):
            output_dims = linear.weight.shape[0]
            input_dims = linear.weight.shape[1] * 32 // linear.bits
        else:
            output_dims, input_dims = linear.weight.shape
        lora = LoRALinear(
            input_dims,
            output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
            bias="bias" in linear,
        )
        lora.linear = linear
        return lora

    def fuse(self, dequantize=False):
        weight = self.linear.weight
        bias = self.linear.bias if "bias" in self.linear else None
        delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
        fused = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        fused.weight = weight + delta
        if bias is not None:
            fused.bias = bias
        return fused


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "input_proj",
    "output_proj",
)

LORA_EXCLUDED_PATHS = ("switch_mlp", "lm_head")
LORA_EXCLUDED_NAMES = ("x_proj", "dt_proj", "gate", "shared_expert_gate")


def _should_apply_lora(name: str, full_path: str, targets: tuple[str, ...]) -> bool:
    """Check if a linear layer should get LoRA based on name and path."""
    for excluded in LORA_EXCLUDED_PATHS:
        if excluded in full_path:
            return False
    if name in LORA_EXCLUDED_NAMES:
        return False
    return any(name.endswith(t) for t in targets)


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    dropout: float = 0.0,
    scale: float = 20.0,
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    use_dora: bool = False,
) -> list[str]:
    """Walk model tree and replace target nn.Linear/nn.QuantizedLinear with LoRA/DoRA wrappers.

    Returns list of wrapped layer paths for verification.
    """
    from bit_axon.training.dora import DoRALinear

    wrapped = []

    def _replace(mod, parent_path=""):
        for name, child in mod.children().items():
            child_path = f"{parent_path}.{name}" if parent_path else name

            if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
                if _should_apply_lora(name, child_path, targets):
                    wrapper_cls = DoRALinear if use_dora else LoRALinear
                    replacement = wrapper_cls.from_base(child, r=rank, dropout=dropout, scale=scale)
                    setattr(mod, name, replacement)
                    wrapped.append(child_path)
            elif isinstance(child, nn.Module):
                _replace(child, child_path)

    _replace(model)
    return wrapped
