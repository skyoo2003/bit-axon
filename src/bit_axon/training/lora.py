from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from bit_axon.training._adapter_base import _BaseAdapterLinear


class LoRALinear(_BaseAdapterLinear):
    """Low-Rank Adaptation wrapper around a base Linear layer.

    Adds a trainable low-rank decomposition (lora_a @ lora_b) scaled by
    ``scale`` and added to the base layer output. Base weights are frozen.

    Args:
        input_dims: Input dimension of the linear layer.
        output_dims: Output dimension of the linear layer.
        r: LoRA rank.
        dropout: Dropout probability applied before the low-rank path.
        scale: Scaling factor for the LoRA output.
        bias: Whether to include a bias term in the base linear layer.

    Attributes:
        linear: Base frozen linear layer.
        lora_a: Low-rank matrix A of shape (input_dims, r).
        lora_b: Low-rank matrix B of shape (r, output_dims), initialized to zeros.
        scale: Output scaling factor.
    """

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

    @staticmethod
    def from_base(linear, r=8, dropout=0.0, scale=20.0):
        """Create a LoRALinear wrapping an existing Linear or QuantizedLinear.

        Args:
            linear: Base linear layer to wrap. Its weights are preserved.
            r: LoRA rank.
            dropout: Dropout probability.
            scale: Output scaling factor.

        Returns:
            LoRALinear with the base layer's weights and new LoRA matrices.
        """
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
        """Fuse LoRA weights into the base layer, producing a plain nn.Linear.

        Adds the scaled low-rank delta to the base weight. If the base is
        QuantizedLinear and dequantize=True, dequantizes before fusing.

        Args:
            dequantize: If True, dequantize QuantizedLinear weights before fusing.

        Returns:
            nn.Linear with fused weights.
        """
        weight = self.linear.weight
        is_quantized = isinstance(self.linear, nn.QuantizedLinear)
        if dequantize and is_quantized:
            weight = mx.dequantize(
                weight,
                self.linear.scales,
                self.linear.biases,
                self.linear.group_size,
                self.linear.bits,
            )
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

    Layers matching names in LORA_EXCLUDED_NAMES or paths in LORA_EXCLUDED_PATHS
    are skipped regardless of target matching.

    Args:
        model: BitAxonModel to apply adapters to.
        rank: LoRA rank.
        dropout: Dropout probability for the adapter.
        scale: Output scaling factor.
        targets: Tuple of linear layer name suffixes to wrap.
        use_dora: If True, use DoRALinear instead of LoRALinear.

    Returns:
        List of dot-separated paths to wrapped layers.
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
