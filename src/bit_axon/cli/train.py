"""CLI wrapper for Bit-Axon SFT training pipeline."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from bit_axon.cli._console import print_info, print_success

console = Console()


def train_cmd(
    *,
    data: str,
    val_data: str | None,
    tokenizer: str,
    model_weights: str,
    config_small: bool,
    lora_rank: int,
    lora_dropout: float,
    lora_scale: float,
    no_dora: bool,
    learning_rate: float,
    max_steps: int,
    batch_size: int,
    grad_accum_steps: int,
    max_seq_len: int,
    warmup_steps: int,
    max_grad_norm: float,
    seed: int,
    no_thermal: bool,
    temp_pause: float,
    temp_stop: float,
    output_dir: str,
    save_every: int,
    eval_every: int,
    resume: bool,
) -> None:
    """Run the 10-step SFT training pipeline."""
    from pathlib import Path

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
    with console.status("[bold green]Building model config..."):
        if config_small:
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
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
        grad_accum_steps=grad_accum_steps,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        lora_scale=lora_scale,
        use_dora=not no_dora,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        save_every=save_every,
        eval_every=eval_every,
        output_dir=output_dir,
        temp_pause=temp_pause,
        temp_stop=temp_stop,
        seed=seed,
    )

    # 3. Load model
    console.print(f"Loading model (hidden_dim={config.hidden_dim}, layers={config.num_layers})...")
    model = BitAxonModel(config)
    if not config_small:
        model.load_weights(model_weights)
    else:
        import mlx.core as mx

        mx.eval(model.parameters())
        print_info("Using random weights (config-small mode)")

    # 4. Quantize to Q4
    with console.status("[bold green]Quantizing to Q4..."):
        replace_linear_with_quantized(
            model,
            group_size=training_config.quantize_group_size,
            bits=training_config.quantize_bits,
        )
    print_success("Quantization complete")

    # 5. Freeze all, apply LoRA/DoRA
    adapter_type = "DoRA" if training_config.use_dora else "LoRA"
    with console.status(f"[bold green]Applying {adapter_type} (rank={training_config.lora_rank})..."):
        wrapped = apply_lora_to_model(
            model,
            rank=training_config.lora_rank,
            dropout=training_config.lora_dropout,
            scale=training_config.lora_scale,
            targets=training_config.lora_targets,
            use_dora=training_config.use_dora,
        )
    print_success(f"Wrapped {len(wrapped)} layers with {adapter_type} adapters")

    # 6. Load datasets
    console.print(f"Loading tokenizer: {tokenizer}")
    tok = QwenTokenizerWrapper(tokenizer)
    console.print(f"Loading training data: {data}")
    train_dataset = SFTDataset(data, tok, max_seq_len=training_config.max_seq_len)
    val_dataset = None
    if val_data:
        console.print(f"Loading validation data: {val_data}")
        val_dataset = SFTDataset(val_data, tok, max_seq_len=training_config.max_seq_len)

    # 7. Setup cooling scheduler
    cooling = None
    monitor = None
    if not no_thermal:
        from bit_axon.profiling.thermal import ThermalMonitor

        monitor = ThermalMonitor(poll_interval=training_config.temp_poll_interval)
        monitor.start()
        policy = ThermalPolicy(
            pause_temp=training_config.temp_pause,
            stop_temp=training_config.temp_stop,
        )
        cooling = CoolingScheduler(monitor, policy)
        print_info(f"Thermal monitoring enabled (pause={policy.pause_temp}°C, stop={policy.stop_temp}°C)")

    # 8. Train
    effective_batch = training_config.batch_size * training_config.grad_accum_steps
    console.print()
    console.print(f"[bold]Starting training:[/bold] {training_config.max_steps} steps, lr={training_config.learning_rate}")
    console.print(f"Batch size={training_config.batch_size}, grad_accum={training_config.grad_accum_steps}, effective={effective_batch}")
    console.print(f"Output dir: {training_config.output_dir}")
    console.print()

    trainer = Trainer(model, training_config, train_dataset, val_dataset=val_dataset, cooling_scheduler=cooling)
    if resume:
        trainer.step_count = 0  # setup() handles checkpoint restoration
    result = trainer.train()

    # 9. Save final adapter
    adapter_path = f"{training_config.output_dir}/final_adapter.safetensors"
    Path(training_config.output_dir).mkdir(parents=True, exist_ok=True)
    save_adapter_only(model, adapter_path)
    print_success(f"Adapter saved to: {adapter_path}")

    # 10. Print results table
    table = Table(title="Training Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Step", str(result["step"]))
    table.add_row("Loss", f"{result['loss']:.4f}")
    table.add_row("Grad Norm", f"{result['grad_norm']:.4f}")
    table.add_row("Adapter", adapter_path)
    console.print(table)

    # Cleanup
    if monitor is not None and cooling is not None:
        monitor.stop()
        print_info(f"Total thermal pause time: {cooling.total_pause_time:.1f}s")
