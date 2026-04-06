"""Export reference layer outputs for Swift numerical equivalence testing.

Usage:
    cd /Users/lukas/Workspace/bit-axon
    python BitAxonApp/Tests/EquivalenceTestSupport/export_reference.py

Outputs JSON files to BitAxonApp/Tests/EquivalenceTestSupport/reference/
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add project root to path
sys.path.insert(0, "/Users/lukas/Workspace/bit-axon/src")

from bit_axon.config import BitAxonConfig
from bit_axon.layers.axon_ssm import AxonSSM
from bit_axon.layers.moe import SharedExpertMoE
from bit_axon.layers.rms_norm import RMSNorm
from bit_axon.layers.swa import SlidingWindowAttention


def to_python(obj):
    """Recursively convert mlx arrays to nested Python lists."""
    if isinstance(obj, mx.array):
        return obj.astype(mx.float32).tolist()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj


def export_rmsnorm(output_dir: Path):
    """Export RMSNorm reference."""
    config = BitAxonConfig()
    layer = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
    # Override weight with fixed values
    layer.weight = mx.arange(1, config.hidden_dim + 1, dtype=mx.float32) / config.hidden_dim

    # Input: (2, 4, hidden_dim) — batch=2, seq_len=4
    np.random.seed(42)
    x = mx.array(np.random.randn(2, 4, config.hidden_dim).astype(np.float32))

    output = layer(x)

    data = {
        "layer": "rms_norm",
        "hidden_dim": config.hidden_dim,
        "eps": config.rms_norm_eps,
        "input_shape": list(x.shape),
        "weight": to_python(layer.weight),
        "input": to_python(x),
        "output": to_python(output),
    }

    path = output_dir / "rms_norm.json"
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  ✓ rms_norm -> {path}  (input {list(x.shape)} -> output {list(output.shape)})")


def export_ssm(output_dir: Path):
    """Export AxonSSM reference."""
    # Use a small config for manageable file size
    small_config = BitAxonConfig(
        hidden_dim=64,
        num_layers=1,
        num_heads=2,
        d_source_model=32,
        vocab_size=512,
        ssm_d_state=4,
        ssm_d_conv=3,
        ssm_expand=2,
    )

    layer = AxonSSM(small_config)

    # Input: (1, 8, 64) — batch=1, seq_len=8, hidden_dim=64
    np.random.seed(42)
    x = mx.array(np.random.randn(1, 8, small_config.hidden_dim).astype(np.float32))

    output, cache = layer(x)

    data = {
        "layer": "axon_ssm",
        "config": {
            "hidden_dim": small_config.hidden_dim,
            "ssm_intermediate_dim": small_config.ssm_intermediate_dim,
            "ssm_d_state": small_config.ssm_d_state,
            "ssm_d_conv": small_config.ssm_d_conv,
        },
        "input_shape": list(x.shape),
        "weights": {
            "in_proj_weight": to_python(layer.in_proj.weight),
            "conv1d_weight": to_python(layer.conv1d.weight),
            "conv1d_bias": to_python(layer.conv1d.bias),
            "x_proj_weight": to_python(layer.x_proj.weight),
            "dt_proj_weight": to_python(layer.dt_proj.weight),
            "dt_proj_bias": to_python(layer.dt_proj.bias),
            "out_proj_weight": to_python(layer.out_proj.weight),
            "A_log": to_python(layer.A_log),
            "D": to_python(layer.D),
        },
        "input": to_python(x),
        "output": to_python(output),
        "cache_shapes": {
            "conv_cache": list(cache[0].shape),
            "ssm_state": list(cache[1].shape),
        },
        "cache": {
            "conv_cache": to_python(cache[0]),
            "ssm_state": to_python(cache[1]),
        },
    }

    path = output_dir / "axon_ssm.json"
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  ✓ axon_ssm -> {path}  (input {list(x.shape)} -> output {list(output.shape)})")


def export_swa(output_dir: Path):
    """Export SlidingWindowAttention reference."""
    small_config = BitAxonConfig(
        hidden_dim=64,
        num_layers=1,
        num_heads=2,
        swa_window_size=4,
    )

    layer = SlidingWindowAttention(
        hidden_dim=small_config.hidden_dim,
        num_heads=small_config.num_heads,
        window_size=small_config.swa_window_size,
    )

    # Input: (1, 6, 64)
    np.random.seed(42)
    x = mx.array(np.random.randn(1, 6, small_config.hidden_dim).astype(np.float32))

    output, _ = layer(x)

    data = {
        "layer": "axon_swa",
        "config": {
            "hidden_dim": small_config.hidden_dim,
            "num_heads": small_config.num_heads,
            "head_dim": small_config.head_dim,
            "window_size": small_config.swa_window_size,
        },
        "input_shape": list(x.shape),
        "weights": {
            "q_proj_weight": to_python(layer.q_proj.weight),
            "k_proj_weight": to_python(layer.k_proj.weight),
            "v_proj_weight": to_python(layer.v_proj.weight),
            "o_proj_weight": to_python(layer.o_proj.weight),
        },
        "input": to_python(x),
        "output": to_python(output),
    }

    path = output_dir / "axon_swa.json"
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  ✓ axon_swa -> {path}  (input {list(x.shape)} -> output {list(output.shape)})")


def export_moe(output_dir: Path):
    """Export SharedExpertMoE reference."""
    small_config = BitAxonConfig(
        hidden_dim=64,
        moe_num_experts=4,  # Small for manageable test
        moe_top_k=2,
        moe_intermediate_dim=128,
    )

    layer = SharedExpertMoE(
        dim=small_config.hidden_dim,
        intermediate_dim=small_config.moe_intermediate_dim,
        num_experts=small_config.moe_num_experts,
        top_k=small_config.moe_top_k,
    )

    # Input: (1, 4, 64)
    np.random.seed(42)
    x = mx.array(np.random.randn(1, 4, small_config.hidden_dim).astype(np.float32))

    output = layer(x)

    # SwitchLinear weights are (num_experts, output_dims, input_dims)
    switch_gate_w = layer.switch_mlp.gate_proj.weight  # (4, 128, 64)
    switch_up_w = layer.switch_mlp.up_proj.weight  # (4, 128, 64)
    switch_down_w = layer.switch_mlp.down_proj.weight  # (4, 64, 128)

    data = {
        "layer": "axon_moe",
        "config": {
            "hidden_dim": small_config.hidden_dim,
            "intermediate_dim": small_config.moe_intermediate_dim,
            "num_experts": small_config.moe_num_experts,
            "top_k": small_config.moe_top_k,
        },
        "input_shape": list(x.shape),
        "weights": {
            "gate_weight": to_python(layer.gate.weight),
            "shared_expert_gate_weight": to_python(layer.shared_expert_gate.weight),
            # Shared expert (standard nn.Linear, no bias)
            "shared_expert_gate_proj_weight": to_python(layer.shared_expert.gate_proj.weight),
            "shared_expert_up_proj_weight": to_python(layer.shared_expert.up_proj.weight),
            "shared_expert_down_proj_weight": to_python(layer.shared_expert.down_proj.weight),
            # SwitchGLU experts (SwitchLinear, weight shape: (num_experts, out, in))
            "switch_gate_proj_weight": to_python(switch_gate_w),
            "switch_up_proj_weight": to_python(switch_up_w),
            "switch_down_proj_weight": to_python(switch_down_w),
        },
        "input": to_python(x),
        "output": to_python(output),
    }

    path = output_dir / "axon_moe.json"
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  ✓ axon_moe -> {path}  (input {list(x.shape)} -> output {list(output.shape)})")


def main():
    output_dir = Path(__file__).parent / "reference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting reference tensors for Swift equivalence testing...")
    print(f"Output directory: {output_dir}")
    print()

    export_rmsnorm(output_dir)
    export_ssm(output_dir)
    export_swa(output_dir)
    export_moe(output_dir)

    print()
    print("Done! Reference files written.")


if __name__ == "__main__":
    main()
