"""Bit-Axon SFT training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Bit-Axon SFT training configuration.

    Thermal-aware QLoRA fine-tuning for fanless MacBook Air M4.
    """

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10_000
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 4

    # LoRA / DoRA
    lora_rank: int = 8
    lora_dropout: float = 0.0
    lora_scale: float = 20.0
    lora_targets: tuple[str, ...] = (
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
    use_dora: bool = True

    # ORPO alignment
    beta: float = 0.1
    training_mode: str = "sft"  # "sft" or "orpo"

    # Quantization
    quantize_bits: int = 4
    quantize_group_size: int = 64

    # Data
    batch_size: int = 1
    max_seq_len: int = 512
    eos_token_id: int = 0

    # Checkpointing
    save_every: int = 500
    eval_every: int = 500
    eval_batches: int = 10
    output_dir: str = "checkpoints"

    # Thermal thresholds (°C)
    temp_max_speed: float = 75.0
    temp_pause: float = 85.0
    temp_stop: float = 95.0
    temp_poll_interval: float = 1.0

    # Misc
    seed: int = 42
    low_memory: bool = False

    @classmethod
    def low_memory_preset(cls) -> TrainingConfig:
        return cls(batch_size=1, max_seq_len=256, grad_accum_steps=8, lora_rank=4, low_memory=True)

    @classmethod
    def fast_dev(cls) -> TrainingConfig:
        return cls(max_steps=100, eval_every=50, save_every=10000)
