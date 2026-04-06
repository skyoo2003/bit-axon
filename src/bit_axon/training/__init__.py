"""Bit-Axon training module: SFT, ORPO alignment, LoRA/DoRA adapters, and model merging."""

from __future__ import annotations

from bit_axon.training.checkpoint import load_checkpoint, save_adapter_only, save_checkpoint
from bit_axon.training.config import TrainingConfig
from bit_axon.training.data import AlpacaDataset, ORPODataset, SFTDataset
from bit_axon.training.dora import DoRALinear
from bit_axon.training.lora import LoRALinear, apply_lora_to_model
from bit_axon.training.merging import (
    dequantize_model,
    load_and_merge,
    merge_adapters,
    quantize_model,
    save_merged_model,
)
from bit_axon.training.orpo_collate import collate_orpo_batch, iterate_orpo_batches
from bit_axon.training.orpo_loss import compute_orpo_loss, get_logps, orpo_loss
from bit_axon.training.orpo_trainer import ORPOTrainer
from bit_axon.training.trainer import Trainer

__all__ = [
    "AlpacaDataset",
    "DoRALinear",
    "LoRALinear",
    "ORPODataset",
    "ORPOTrainer",
    "SFTDataset",
    "Trainer",
    "TrainingConfig",
    "apply_lora_to_model",
    "collate_orpo_batch",
    "compute_orpo_loss",
    "dequantize_model",
    "get_logps",
    "iterate_orpo_batches",
    "load_and_merge",
    "load_checkpoint",
    "merge_adapters",
    "orpo_loss",
    "quantize_model",
    "save_adapter_only",
    "save_checkpoint",
    "save_merged_model",
]
