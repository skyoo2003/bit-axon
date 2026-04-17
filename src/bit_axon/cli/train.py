"""CLI wrapper for Bit-Axon SFT training pipeline."""

from __future__ import annotations

import gc

from rich.console import Console
from rich.table import Table

from bit_axon.cli._console import print_info, print_success

console = Console()


def _load_vocab_mapping(safetensors_path: str) -> dict[int, int] | None:
    """Load vocab mapping from safetensors file metadata."""
    try:
        from safetensors import safe_open

        f = safe_open(safetensors_path, framework="mlx")
        meta = f.metadata() or {}
        raw = meta.get("vocab_mapping")
        if raw is None:
            return None
        return eval(raw)
    except Exception:
        return None


def train_cmd(
    *,
    data: str,
    val_data: str | None,
    tokenizer: str,
    model_weights: str,
    config_small: bool,
    config_medium: bool = False,
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
    low_memory: bool = False,
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
            config = BitAxonConfig.small()
        elif config_medium:
            config = BitAxonConfig.medium()
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
        low_memory=low_memory,
    )

    # 3. Load model
    console.print(f"Loading model (hidden_dim={config.hidden_dim}, layers={config.num_layers})...")
    model = BitAxonModel(config)
    vocab_mapping: dict[int, int] | None = None
    if not config_small:
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        # Resolve HF repo ID to local path if needed
        weights_path = Path(model_weights)
        if not weights_path.exists():
            from huggingface_hub import snapshot_download

            console.print(f"Downloading weights from {model_weights}...")
            weights_path = Path(snapshot_download(model_weights))

        # Load safetensors weights from directory or file
        weights: dict[str, mx.array] = {}
        if weights_path.is_file():
            weights.update(mx.load(str(weights_path)))
            vocab_mapping = _load_vocab_mapping(str(weights_path))
        else:
            for sf_file in sorted(weights_path.glob("*.safetensors")):
                weights.update(mx.load(str(sf_file)))
            vocab_mapping = _load_vocab_mapping(str(sorted(weights_path.glob("*.safetensors"))[0]))
        if weights:
            model.update(tree_unflatten(list(weights.items())))
        mx.eval(model.parameters())
        print_info(f"Loaded weights from {weights_path}")
    else:
        import mlx.core as mx

        mx.eval(model.parameters())
        vocab_mapping = {i: i for i in range(config.vocab_size)}
        print_info("Using random weights (config-small mode)")

    # 4. Quantize to Q4 (regular Linear only — SwitchLinear/MoE kept in fp16)
    with console.status("[bold green]Quantizing to Q4..."):
        replace_linear_with_quantized(
            model,
            group_size=training_config.quantize_group_size,
            bits=training_config.quantize_bits,
        )
    mx.eval(model.parameters())
    mx.clear_cache()
    gc.collect()
    print_success("Quantization complete")

    # 4.5. Cast embedding to float16 to save ~125 MB
    import mlx.core as mx

    model.embed_tokens.weight = model.embed_tokens.weight.astype(mx.float16)
    mx.eval(model.parameters())
    print_info("Embedding cast to float16")

    # 5. Freeze all, apply LoRA (DoRA incompatible with NF4 quantized base weights)
    adapter_type = "LoRA"
    with console.status(f"[bold green]Applying {adapter_type} (rank={training_config.lora_rank})..."):
        wrapped = apply_lora_to_model(
            model,
            rank=training_config.lora_rank,
            dropout=training_config.lora_dropout,
            scale=training_config.lora_scale,
            targets=training_config.lora_targets,
            use_dora=False,
        )
    mx.eval(model.parameters())
    mx.clear_cache()
    gc.collect()
    print_success(f"Wrapped {len(wrapped)} layers with {adapter_type} adapters")

    # Freeze everything, then unfreeze only adapter params so nn.value_and_grad
    # doesn't retain activations for the entire 4B+ param model during backprop.
    model.freeze()
    model.apply_to_modules(lambda k, m: m.unfreeze(keys=["lora_a", "lora_b"]) if type(m).__name__ == "LoRALinear" else None)

    # 6. Load datasets
    console.print(f"Loading tokenizer: {tokenizer}")
    tok = QwenTokenizerWrapper(tokenizer)
    console.print(f"Loading training data: {data}")
    train_dataset = SFTDataset(data, tok, max_seq_len=training_config.max_seq_len, vocab_mapping=vocab_mapping)
    val_dataset = None
    if val_data:
        console.print(f"Loading validation data: {val_data}")
        val_dataset = SFTDataset(val_data, tok, max_seq_len=training_config.max_seq_len, vocab_mapping=vocab_mapping)

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
