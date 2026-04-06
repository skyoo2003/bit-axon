"""Full ML pipeline: SFT train, merge, quantize, evaluate, inference, ORPO."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from rich.console import Console
from rich.table import Table

from bit_axon.cli._console import print_info, print_success

console = Console()

_SMALL_CONFIG = {
    "hidden_dim": 256,
    "num_layers": 4,
    "num_heads": 4,
    "d_source_model": 128,
    "vocab_size": 1024,
    "ssm_d_state": 4,
    "ssm_d_conv": 2,
    "ssm_expand": 2,
    "swa_window_size": 64,
    "moe_num_experts": 4,
    "moe_top_k": 2,
    "moe_intermediate_dim": 512,
}


class SimpleDataset:
    """Minimal dataset yielding (token_ids, loss_mask) tuples."""

    def __init__(self, num_examples=50, seq_len=32, vocab_size=1024):
        rng = np.random.RandomState(42)
        self._data = [(rng.randint(1, vocab_size, size=seq_len).tolist(), [1] * seq_len) for _ in range(num_examples)]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class SimpleORPODataset:
    """Minimal dataset yielding (chosen_ids, chosen_mask, rejected_ids, rejected_mask)."""

    def __init__(self, num_examples=50, seq_len=32, vocab_size=1024):
        rng = np.random.RandomState(42)
        self._data = []
        for _ in range(num_examples):
            prompt_len = seq_len // 2
            response_len = seq_len - prompt_len
            prompt_ids = rng.randint(1, vocab_size, size=prompt_len).tolist()
            chosen_response = rng.randint(1, vocab_size, size=response_len).tolist()
            rejected_response = rng.randint(1, vocab_size, size=response_len).tolist()
            chosen_ids = list(prompt_ids) + list(chosen_response)
            rejected_ids = list(prompt_ids) + list(rejected_response)
            chosen_mask = [0] * prompt_len + [1] * response_len
            rejected_mask = [0] * prompt_len + [1] * response_len
            self._data.append((chosen_ids, chosen_mask, rejected_ids, rejected_mask))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class LogitsOnlyModel(nn.Module):
    """Wraps BitAxonModel to return only logits, discarding KV caches."""

    def __init__(self, model):
        super().__init__()
        self.inner = model

    def __call__(self, input_ids, cache=None):
        logits, _ = self.inner(input_ids, cache=cache)
        return logits


def pipeline_cmd(
    output_dir: str,
    max_steps: int,
    orpo_steps: int,
    max_seq_len: int,
    lora_rank: int,
    batch_size: int,
) -> None:
    """Run full ML pipeline end-to-end."""
    import time

    from mlx.utils import tree_flatten

    from bit_axon.config import BitAxonConfig
    from bit_axon.evaluation.perplexity import compute_perplexity
    from bit_axon.inference.loader import load_model
    from bit_axon.inference.sampling import sample_logits
    from bit_axon.model import BitAxonModel
    from bit_axon.quantization.nf4 import replace_linear_with_quantized
    from bit_axon.training.checkpoint import save_adapter_only
    from bit_axon.training.config import TrainingConfig
    from bit_axon.training.lora import apply_lora_to_model
    from bit_axon.training.merging import load_and_merge, save_merged_model
    from bit_axon.training.orpo_trainer import ORPOTrainer
    from bit_axon.training.trainer import Trainer

    base_dir = Path(output_dir) / "base"
    sft_dir = Path(output_dir) / "sft"
    merged_dir = Path(output_dir) / "merged"
    quant_dir = Path(output_dir) / "quantized"
    orpo_dir = Path(output_dir) / "orpo"
    final_dir = Path(output_dir) / "final"

    t_start = time.perf_counter()
    results: dict[str, float] = {}

    print_info("Stage 1: SFT Training")
    config = BitAxonConfig(**_SMALL_CONFIG)
    model = BitAxonModel(config)
    mx.eval(model.parameters())

    base_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(base_dir / "weights.safetensors"), dict(tree_flatten(model.parameters())))

    apply_lora_to_model(model, rank=lora_rank, dropout=0.0, scale=20.0)
    mx.eval(model.parameters())

    sft_config = TrainingConfig(
        max_steps=max_steps,
        grad_accum_steps=1,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        learning_rate=1e-3,
        lora_rank=lora_rank,
        lora_scale=20.0,
        use_dora=False,
        save_every=10000,
        output_dir=str(sft_dir),
    )
    trainer = Trainer(model, sft_config, SimpleDataset(50, max_seq_len, config.vocab_size))
    with console.status("[bold green]SFT training...", spinner="dots"):
        result = trainer.train()
    results["sft_loss"] = result["loss"]

    sft_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(sft_dir / "adapter.safetensors")
    save_adapter_only(model, adapter_path)
    print_success(f"SFT: step={result['step']}, loss={result['loss']:.4f}")

    print_info("Stage 2: Merge adapter into base")
    with console.status("[bold green]Merging adapter...", spinner="dots"):
        load_and_merge(
            base_dir,
            adapter_path,
            merged_dir,
            config=config,
            quantize_after_merge=False,
            lora_rank=lora_rank,
        )
    print_success("Merge complete")

    print_info("Stage 3: Quantize to NF4")
    merged_model = load_model(merged_dir, config=config)
    replace_linear_with_quantized(merged_model, group_size=64, bits=4)
    mx.eval(merged_model.parameters())
    with console.status("[bold green]Saving quantized model...", spinner="dots"):
        save_merged_model(merged_model, quant_dir, config=config)
    print_success("Quantization complete")

    print_info("Stage 4: Evaluate perplexity")
    eval_rng = np.random.RandomState(123)
    test_tokens = mx.array(eval_rng.randint(1, config.vocab_size, size=(4, max_seq_len + 1)), dtype=mx.uint32)
    with console.status("[bold green]Computing perplexity...", spinner="dots"):
        ppl, se = compute_perplexity(merged_model, test_tokens)
    results["ppl"] = ppl
    print_success(f"Perplexity: {ppl:.2f} +/- {se:.2f}")

    print_info("Stage 5: Autoregressive inference")
    prompt_ids = mx.array([[1, 42, 100, 200, 500]], dtype=mx.uint32)
    logits, caches = merged_model(prompt_ids)

    t_gen = time.perf_counter()
    for _ in range(20):
        next_token = sample_logits(logits[:, -1, :], temperature=0.8, top_k=50, top_p=0.95)
        logits, caches = merged_model(mx.array([[int(next_token.item())]], dtype=mx.uint32), cache=caches)
    mx.synchronize()
    tok_s = 20 / (time.perf_counter() - t_gen)
    results["tok_s"] = tok_s
    print_success(f"Inference: {tok_s:.1f} tok/s")

    print_info("Stage 6: ORPO preference alignment")
    orpo_model = BitAxonModel(config)
    mx.eval(orpo_model.parameters())
    apply_lora_to_model(orpo_model, rank=lora_rank, dropout=0.0, scale=20.0)
    mx.eval(orpo_model.parameters())

    orpo_wrapper = LogitsOnlyModel(orpo_model)
    mx.eval(orpo_wrapper.parameters())

    orpo_config = TrainingConfig(
        max_steps=orpo_steps,
        grad_accum_steps=1,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        learning_rate=1e-3,
        lora_rank=lora_rank,
        lora_scale=20.0,
        use_dora=False,
        beta=0.1,
        training_mode="orpo",
        save_every=10000,
        output_dir=str(orpo_dir),
    )
    orpo_trainer = ORPOTrainer(orpo_wrapper, orpo_config, SimpleORPODataset(50, max_seq_len, config.vocab_size))
    with console.status("[bold green]ORPO training...", spinner="dots"):
        orpo_result = orpo_trainer.train()
    results["orpo_loss"] = orpo_result["loss"]

    orpo_dir.mkdir(parents=True, exist_ok=True)
    orpo_adapter = str(orpo_dir / "orpo_adapter.safetensors")
    save_adapter_only(orpo_model, orpo_adapter)
    print_success(f"ORPO: step={orpo_result['step']}, loss={orpo_result['loss']:.4f}")

    print_info("Stage 7: Final merge + quantize")
    with console.status("[bold green]Final merge + quantize...", spinner="dots"):
        load_and_merge(
            base_dir,
            orpo_adapter,
            final_dir,
            config=config,
            quantize_after_merge=True,
            bits=4,
            group_size=64,
            lora_rank=lora_rank,
        )
    print_success("Final merge + quantize complete")

    elapsed = time.perf_counter() - t_start
    table = Table(title="Pipeline Results")
    table.add_column("Stage", style="cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("SFT Training", "Loss", f"{results['sft_loss']:.4f}")
    table.add_row("SFT Training", "Steps", str(result["step"]))
    table.add_row("Perplexity", "PPL", f"{results['ppl']:.2f}")
    table.add_row("Inference", "Speed", f"{results['tok_s']:.1f} tok/s")
    table.add_row("ORPO", "Loss", f"{results['orpo_loss']:.4f}")
    table.add_row("Total", "Time", f"{elapsed:.1f}s")
    console.print(table)

    log_path = Path(output_dir) / "pipeline_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Pipeline completed in {elapsed:.1f}s\n")
        f.write(f"SFT loss: {results['sft_loss']:.4f}\n")
        f.write(f"Perplexity: {results['ppl']:.2f}\n")
        f.write(f"Inference: {results['tok_s']:.1f} tok/s\n")
        f.write(f"ORPO loss: {results['orpo_loss']:.4f}\n")
    print_success(f"Log saved to {log_path}")
