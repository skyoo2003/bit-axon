#!/usr/bin/env python3
"""Thermal-aware SFT training for Bit-Axon with QLoRA/Bit-DoRA.

Usage:
    python scripts/train.py --train-data data/train.jsonl --model-weights weights/
    python scripts/train.py --config-small --train-data data/train.jsonl --model-weights weights/ --max-steps 10
    python scripts/train.py --train-data data/train.jsonl --model-weights weights/ --no-thermal

Requires sudo for thermal monitoring (powermetrics). Use --no-thermal to disable.
"""

import argparse
import warnings

warnings.warn(
    "This script is deprecated. Use the `bit-axon` CLI instead. Run `pip install -e .` and then `bit-axon --help` for available commands.",
    DeprecationWarning,
    stacklevel=2,
)


def main():
    parser = argparse.ArgumentParser(description="Bit-Axon SFT Trainer (Thermal-Aware QLoRA)")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation JSONL file")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-3B", help="Tokenizer name or path")
    parser.add_argument("--model-weights", type=str, required=True, help="Path to model weights directory")
    parser.add_argument("--config-small", action="store_true", help="Use small config for testing")

    # LoRA args
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--no-dora", action="store_true", help="Use LoRA instead of DoRA")

    # Training args
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Thermal args
    parser.add_argument("--no-thermal", action="store_true", help="Disable thermal monitoring")
    parser.add_argument("--temp-pause", type=float, default=85.0)
    parser.add_argument("--temp-stop", type=float, default=95.0)

    # Output args
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    from bit_axon.config import BitAxonConfig
    from bit_axon.model import BitAxonModel
    from bit_axon.quantization.nf4 import replace_linear_with_quantized
    from bit_axon.tokenizer import QwenTokenizerWrapper
    from bit_axon.training.checkpoint import save_adapter_only
    from bit_axon.training.config import TrainingConfig
    from bit_axon.training.cooling import CoolingScheduler, ThermalPolicy
    from bit_axon.training.data import SFTDataset
    from bit_axon.training.lora import apply_lora_to_model
    from bit_axon.training.trainer import Trainer

    # 1. Build config
    if args.config_small:
        config = BitAxonConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            d_source_model=128,
            vocab_size=1024,
        )
    else:
        config = BitAxonConfig()

    # 2. Build training config
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        grad_accum_steps=args.grad_accum_steps,
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
        lora_scale=args.lora_scale,
        use_dora=not args.no_dora,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        save_every=args.save_every,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
        temp_pause=args.temp_pause,
        temp_stop=args.temp_stop,
        seed=args.seed,
    )

    # 3. Load model
    print(f"Loading model (hidden_dim={config.hidden_dim}, layers={config.num_layers})...")
    model = BitAxonModel(config)
    if not args.config_small:
        model.load_weights(args.model_weights)
    else:
        import mlx.core as mx

        mx.eval(model.parameters())
        print("Using random weights (config-small mode)")

    # 4. Quantize to Q4
    print("Quantizing to Q4...")
    replace_linear_with_quantized(model, group_size=training_config.quantize_group_size, bits=training_config.quantize_bits)

    # 5. Freeze all, apply LoRA/DoRA
    print(f"Applying {'DoRA' if training_config.use_dora else 'LoRA'} (rank={training_config.lora_rank})...")
    wrapped = apply_lora_to_model(
        model,
        rank=training_config.lora_rank,
        dropout=training_config.lora_dropout,
        scale=training_config.lora_scale,
        targets=training_config.lora_targets,
        use_dora=training_config.use_dora,
    )
    print(f"Wrapped {len(wrapped)} layers with adapters")

    # 6. Load datasets
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = QwenTokenizerWrapper(args.tokenizer)
    print(f"Loading training data: {args.train_data}")
    train_dataset = SFTDataset(args.train_data, tokenizer, max_seq_len=training_config.max_seq_len)
    val_dataset = None
    if args.val_data:
        print(f"Loading validation data: {args.val_data}")
        val_dataset = SFTDataset(args.val_data, tokenizer, max_seq_len=training_config.max_seq_len)

    # 7. Setup cooling scheduler
    cooling = None
    monitor = None
    if not args.no_thermal:
        from bit_axon.profiling.thermal import ThermalMonitor

        monitor = ThermalMonitor(poll_interval=training_config.temp_poll_interval)
        monitor.start()
        policy = ThermalPolicy(
            pause_temp=training_config.temp_pause,
            stop_temp=training_config.temp_stop,
        )
        cooling = CoolingScheduler(monitor, policy)
        print(f"Thermal monitoring enabled (pause={policy.pause_temp}C, stop={policy.stop_temp}C)")

    # 8. Train
    print(f"\nStarting training: {training_config.max_steps} steps, lr={training_config.learning_rate}")
    print(f"Batch size={training_config.batch_size}, grad_accum={training_config.grad_accum_steps}")
    print(f"Effective batch size={training_config.batch_size * training_config.grad_accum_steps}")
    print(f"Output dir: {training_config.output_dir}\n")

    trainer = Trainer(model, training_config, train_dataset, val_dataset=val_dataset, cooling_scheduler=cooling)
    result = trainer.train()

    print(f"\nTraining complete: step={result['step']}, loss={result['loss']:.4f}, grad_norm={result['grad_norm']:.4f}")

    # 9. Save final adapter
    adapter_path = f"{training_config.output_dir}/final_adapter.safetensors"
    save_adapter_only(model, adapter_path)
    print(f"Adapter saved to: {adapter_path}")

    # 10. Cleanup
    if monitor is not None and cooling is not None:
        monitor.stop()
        print(f"Total thermal pause time: {cooling.total_pause_time:.1f}s")


if __name__ == "__main__":
    main()
