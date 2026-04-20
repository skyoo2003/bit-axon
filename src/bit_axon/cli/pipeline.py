"""Full ML pipeline: SFT train, merge, quantize, evaluate, inference, ORPO."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from rich.console import Console
from rich.table import Table

from bit_axon.cli._console import print_error, print_info, print_success

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
    sft_data: str | None = None,
    sft_split: str = "train",
    sft_limit: int | None = None,
    orpo_data: str | None = None,
    orpo_split: str = "train",
    orpo_limit: int | None = None,
    tokenizer: str | None = None,
    config_small: bool = False,
    config_medium: bool = False,
    config_large: bool = False,
    repo_id: str | None = None,
    benchmark_limit: int | None = 100,
    benchmark_max_tokens: int = 256,
    benchmarks_enabled: bool = True,
) -> None:
    """Run full ML pipeline end-to-end."""
    import gc
    import time

    from mlx.utils import tree_flatten
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

    from bit_axon.cli._datasets import resolve_orpo_data, resolve_sft_data
    from bit_axon.config import BitAxonConfig
    from bit_axon.evaluation.perplexity import compute_perplexity
    from bit_axon.inference.sampling import sample_logits
    from bit_axon.model import BitAxonModel
    from bit_axon.quantization.nf4 import replace_linear_with_quantized
    from bit_axon.tokenizer import QwenTokenizerWrapper
    from bit_axon.training.checkpoint import save_adapter_only
    from bit_axon.training.config import TrainingConfig
    from bit_axon.training.data import ORPODataset, SFTDataset
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
    training_metrics: list[dict] = []
    errors: list[str] = []

    def _save_partial() -> None:
        partial = {
            "total_time_seconds": round(time.perf_counter() - t_start, 1),
            "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in results.items()},
            "errors": errors,
        }
        p = Path(output_dir) / "pipeline_results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(partial, f, indent=2)

    # Validate dataset/tokenizer arguments
    if (sft_data or orpo_data) and tokenizer is None:
        print_error("--tokenizer is required when using real datasets (--sft-data or --orpo-data)")
        raise SystemExit(1)

    tok = None
    if sft_data is not None:
        print_info("Stage 1: SFT Training (real dataset)")
        if tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        tok = QwenTokenizerWrapper(tokenizer)
        resolved_sft = resolve_sft_data(sft_data, sft_split, sft_limit)
        if resolved_sft is None:
            raise RuntimeError("Failed to resolve SFT dataset")
        if config_small:
            config = BitAxonConfig.small()
        elif config_medium:
            config = BitAxonConfig.medium()
        elif config_large:
            config = BitAxonConfig.large()
        else:
            config = BitAxonConfig(**{**_SMALL_CONFIG, "vocab_size": tok.vocab_size})
        # Preset vocab_size (small=1024, medium/large=32000) is smaller than
        # real tokenizers (Qwen ~151k). Without this override, token ids
        # beyond the preset vocab cause out-of-bounds embedding reads and
        # produce NaN on the very first forward pass.
        if config.vocab_size < tok.vocab_size:
            config.vocab_size = tok.vocab_size
        model = BitAxonModel(config)
        mx.eval(model.parameters())

        # Port Qwen2.5-3B's pre-trained embedding (and LM head via weight
        # tying) into the freshly-initialized model. This collapses the
        # "every token is predicted uniformly" regime that causes
        # perplexity ≈ exp(log(vocab_size)) on out-of-domain text.
        #
        # Skip the port when d_source_model is far below Qwen's 2048 — the
        # required truncation (e.g. 128 for the small preset keeps only
        # 6.25% of each embedding vector) destroys enough structure that
        # the resulting initialization is *less* stable than random. We've
        # observed SFT NaN even with std-rescaling at d_source=128.
        if config.d_source_model >= 512:
            try:
                from bit_axon.porting.qwen_embedding import apply_qwen_embedding_init

                apply_qwen_embedding_init(model, tokenizer_id=tokenizer, verbose=True)
                mx.eval(model.parameters())
            except Exception as e:
                print_info(f"Qwen embedding port skipped: {e}")
        else:
            print_info(f"Qwen embedding port skipped: d_source_model={config.d_source_model} < 512 (truncation too aggressive)")

        base_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(base_dir / "weights.safetensors"), dict(tree_flatten(model.parameters())))

        apply_lora_to_model(model, rank=lora_rank, dropout=0.0, scale=8.0)
        mx.eval(model.parameters())

        sft_config = TrainingConfig(
            max_steps=max_steps,
            grad_accum_steps=1,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            learning_rate=1e-4,
            lora_rank=lora_rank,
            lora_scale=8.0,
            use_dora=False,
            save_every=10000,
            output_dir=str(sft_dir),
        )
        sft_dataset = SFTDataset(resolved_sft, tok, max_seq_len=max_seq_len)
    else:
        print_info("Stage 1: SFT Training")
        if config_small:
            config = BitAxonConfig.small()
        elif config_medium:
            config = BitAxonConfig.medium()
        elif config_large:
            config = BitAxonConfig.large()
        else:
            config = BitAxonConfig(**_SMALL_CONFIG)
        model = BitAxonModel(config)
        mx.eval(model.parameters())

        base_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(base_dir / "weights.safetensors"), dict(tree_flatten(model.parameters())))

        apply_lora_to_model(model, rank=lora_rank, dropout=0.0, scale=8.0)
        mx.eval(model.parameters())

        sft_config = TrainingConfig(
            max_steps=max_steps,
            grad_accum_steps=1,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            learning_rate=1e-4,
            lora_rank=lora_rank,
            lora_scale=8.0,
            use_dora=False,
            save_every=10000,
            output_dir=str(sft_dir),
        )
        sft_dataset = SimpleDataset(50, max_seq_len, config.vocab_size)

    try:
        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("  loss: [yellow]{task.fields[loss]:.4f}[/]"),
            TextColumn("  grad: [cyan]{task.fields[grad_norm]:.4f}[/]"),
            console=console,
        ) as progress:
            task_id = progress.add_task("SFT Training", total=max_steps, loss=0.0, grad_norm=0.0)

            def _sft_on_step(step: int, metrics: dict) -> None:
                training_metrics.append({"step": step, **metrics})
                progress.update(task_id, completed=step, loss=metrics["loss"], grad_norm=metrics["grad_norm"])
                # Emit a newline status every 50 steps (and at the final
                # step) so background logs capture progress. Use the
                # builtin print with flush=True — rich's Console buffers
                # independently of PYTHONUNBUFFERED, so its output would
                # not reach the nohup-redirected file until the block
                # exits.
                if step % 50 == 0 or step == max_steps:
                    print(
                        f"  SFT step {step}/{max_steps}  loss={metrics['loss']:.4f}  grad={metrics['grad_norm']:.4f}",
                        flush=True,
                    )

            trainer = Trainer(
                model,
                sft_config,
                sft_dataset,
                on_step=_sft_on_step,
            )
            result = trainer.train()
        results["sft_loss"] = result["loss"]
    except Exception as e:
        errors.append(f"Stage 1 (SFT Training): {e}")
        print_error(f"SFT training failed: {e}")
        _save_partial()
        raise

    sft_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(sft_dir / "adapter.safetensors")
    save_adapter_only(model, adapter_path)
    print_success(f"SFT: step={result['step']}, loss={result['loss']:.4f}")

    # Stage 2: fuse LoRA into the live trained model and persist the FULL
    # parameter tree (embed_tokens, lm_head via weight_tying, norms, biases,
    # and Mamba-3 B_bias/C_bias). The previous implementation used
    # ``load_and_merge`` which reloaded base weights (random init) and only
    # overlaid LoRA ``lora_a/lora_b`` — silently discarding every non-LoRA
    # update from training. That produced PPL ≈ exp(log(vocab)) on out-of-
    # domain text because the merged embedding stayed random.
    try:
        print_info("Stage 2: Fuse adapter into trained model")
        from bit_axon.training.merging import merge_adapters as _merge_adapters

        with console.status("[bold green]Fusing LoRA into model...", spinner="dots"):
            _merge_adapters(model)
            save_merged_model(model, merged_dir, config=config)
        print_success("Merge complete (full trained parameters preserved)")
    except Exception as e:
        errors.append(f"Stage 2 (Merge): {e}")
        print_error(f"Merge failed: {e}")
        _save_partial()
        raise

    trained_model = model
    del trainer
    gc.collect()
    mx.clear_cache()

    # Stage 3: perplexity on the pre-quantize merged model. Measuring PPL on
    # the NF4-quantized model conflates "is the model learning" with "is NF4
    # a good compression"; we separate them by running PPL before quantize.
    print_info("Stage 3: Evaluate perplexity (pre-quantize)")
    if tok is not None:
        from bit_axon.evaluation.dataset import WikiTextDataset

        with console.status("[bold green]Loading WikiText-103 for perplexity...", spinner="dots"):
            wt_ds = WikiTextDataset(split="test", seq_length=max_seq_len + 1, max_tokens=max_seq_len * 8, tokenizer=tok)
            chunks = [wt_ds[i] for i in range(len(wt_ds))]
            test_tokens = mx.stack(chunks)
        print_info(f"Loaded {test_tokens.shape[0]} sequences ({test_tokens.shape[1]} tokens each) for perplexity evaluation")
    else:
        eval_rng = np.random.RandomState(123)
        test_tokens = mx.array(eval_rng.randint(1, config.vocab_size, size=(4, max_seq_len + 1)), dtype=mx.uint32)
    with console.status("[bold green]Computing perplexity...", spinner="dots"):
        ppl, se = compute_perplexity(trained_model, test_tokens)
    results["ppl"] = ppl
    print_success(f"Perplexity: {ppl:.2f} +/- {se:.2f}")

    bench_results = []
    if tok is not None and benchmarks_enabled:
        print_info("Stage 4: Benchmark evaluation (pre-quantize)")
        from bit_axon.evaluation.benchmark import evaluate_benchmarks
        from bit_axon.evaluation.tasks import BenchmarkConfig

        bench_config = BenchmarkConfig(limit=benchmark_limit, max_tokens=benchmark_max_tokens)
        bench_results = evaluate_benchmarks(trained_model, tok, config=bench_config, console=console)
        for br in bench_results:
            results[f"bench_{br.benchmark_name}"] = br.accuracy
        print_success(f"Benchmarks: {', '.join(f'{r.benchmark_name}={r.accuracy:.1%}' for r in bench_results)}")

    print_info("Stage 5: Autoregressive inference (pre-quantize)")
    prompt_ids = mx.array([[1, 42, 100, 200, 500]], dtype=mx.uint32)
    logits, caches = trained_model(prompt_ids)

    gen_tokens = 20
    with Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("  [cyan]{task.fields[speed]:.1f}[/] tok/s"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Generating tokens", total=gen_tokens, speed=0.0)
        t_gen = time.perf_counter()
        for i in range(gen_tokens):
            next_token = sample_logits(logits[:, -1, :], temperature=0.8, top_k=50, top_p=0.95)
            logits, caches = trained_model(mx.array([[int(next_token.item())]], dtype=mx.uint32), cache=caches)
            elapsed = time.perf_counter() - t_gen
            progress.update(task_id, completed=i + 1, speed=(i + 1) / max(elapsed, 1e-9))
        mx.synchronize()
        tok_s = gen_tokens / (time.perf_counter() - t_gen)
    results["tok_s"] = tok_s
    print_success(f"Inference: {tok_s:.1f} tok/s")

    # Quantize AFTER the evaluation stages so metrics reflect the trained
    # model, not the NF4-compressed version. The NF4 artifact is still
    # produced for downstream deployment; it just is not what we evaluate.
    try:
        print_info("Stage 5b: Quantize to NF4 (for deployment artifact)")
        replace_linear_with_quantized(trained_model, group_size=64, bits=4)
        mx.eval(trained_model.parameters())
        with console.status("[bold green]Saving quantized model...", spinner="dots"):
            save_merged_model(trained_model, quant_dir, config=config)
        print_success("Quantization complete")
    except Exception as e:
        errors.append(f"Stage 5b (Quantize): {e}")
        print_error(f"Quantization failed: {e}")
        _save_partial()
        raise

    del trained_model
    gc.collect()
    mx.clear_cache()

    print_info("Stage 6: ORPO preference alignment")
    orpo_model = BitAxonModel(config)
    mx.eval(orpo_model.parameters())
    apply_lora_to_model(orpo_model, rank=lora_rank, dropout=0.0, scale=8.0)
    mx.eval(orpo_model.parameters())

    orpo_wrapper = LogitsOnlyModel(orpo_model)
    mx.eval(orpo_wrapper.parameters())

    orpo_config = TrainingConfig(
        max_steps=orpo_steps,
        grad_accum_steps=1,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        learning_rate=1e-4,
        lora_rank=lora_rank,
        lora_scale=8.0,
        use_dora=False,
        beta=0.1,
        training_mode="orpo",
        save_every=10000,
        output_dir=str(orpo_dir),
    )
    if orpo_data is not None:
        if tok is None:
            if tokenizer is None:
                raise RuntimeError("Tokenizer not initialized")
            tok = QwenTokenizerWrapper(tokenizer)
        resolved_orpo = resolve_orpo_data(orpo_data, orpo_split, orpo_limit)
        if resolved_orpo is None:
            raise RuntimeError("Failed to resolve ORPO dataset")
        orpo_dataset = ORPODataset(resolved_orpo, tok, max_seq_len=max_seq_len)
    else:
        orpo_dataset = SimpleORPODataset(50, max_seq_len, config.vocab_size)
    with Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("  loss: [yellow]{task.fields[loss]:.4f}[/]"),
        TextColumn("  margin: [cyan]{task.fields[margin]:.4f}[/]"),
        console=console,
    ) as progress:
        task_id = progress.add_task("ORPO Training", total=orpo_steps, loss=0.0, margin=0.0)

        def _orpo_on_step(s: int, m: dict) -> None:
            progress.update(
                task_id,
                completed=s,
                loss=m["loss"],
                margin=m.get("reward_margin", 0.0),
            )
            if s % 50 == 0 or s == orpo_steps:
                print(
                    f"  ORPO step {s}/{orpo_steps}  loss={m['loss']:.4f}  margin={m.get('reward_margin', 0.0):.4f}",
                    flush=True,
                )

        orpo_trainer = ORPOTrainer(
            orpo_wrapper,
            orpo_config,
            orpo_dataset,
            on_step=_orpo_on_step,
        )
        orpo_result = orpo_trainer.train()
    results["orpo_loss"] = orpo_result["loss"]

    orpo_dir.mkdir(parents=True, exist_ok=True)
    orpo_adapter = str(orpo_dir / "orpo_adapter.safetensors")
    save_adapter_only(orpo_model, orpo_adapter)
    print_success(f"ORPO: step={orpo_result['step']}, loss={orpo_result['loss']:.4f}")

    del orpo_model, orpo_wrapper, orpo_trainer
    gc.collect()
    mx.clear_cache()

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

    # Save training metrics
    metrics_path = Path(output_dir) / "training_metrics.json"
    if training_metrics:
        with open(metrics_path, "w") as f:
            json.dump(training_metrics, f, indent=2)
        print_info(f"Training metrics saved to {metrics_path}")

    # Stage 8: Upload to HuggingFace (optional)
    if repo_id is not None and tok is not None:
        print_info(f"Stage 8: Uploading to HuggingFace ({repo_id})")
        from bit_axon.cli.upload import upload_cmd as _upload_cmd

        bench_str = ",".join(f"{k}={v:.4f}" for k, v in results.items() if k.startswith("bench_"))
        _upload_cmd(
            model_path=str(final_dir),
            repo_id=repo_id,
            tokenizer=tokenizer,
            benchmark_results=bench_str if bench_str else None,
        )

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
    for br in bench_results if tok is not None else []:
        table.add_row("Benchmark", br.benchmark_name, f"{br.accuracy:.1%}")
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
        for br in bench_results if tok is not None else []:
            f.write(f"Benchmark {br.benchmark_name}: {br.accuracy:.1%} ({br.correct}/{br.total})\n")
    print_success(f"Log saved to {log_path}")

    results_json_path = Path(output_dir) / "pipeline_results.json"
    pipeline_summary = {
        "total_time_seconds": round(elapsed, 1),
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in results.items()},
        "benchmarks": [{"name": br.benchmark_name, "accuracy": round(br.accuracy, 6), "correct": br.correct, "total": br.total} for br in bench_results],
        "artifacts": {
            "base_weights": str(base_dir / "weights.safetensors"),
            "sft_adapter": adapter_path,
            "merged_model": str(merged_dir),
            "quantized_model": str(quant_dir),
            "orpo_adapter": str(orpo_dir / "orpo_adapter.safetensors"),
            "final_model": str(final_dir),
        },
    }
    with open(results_json_path, "w") as f:
        json.dump(pipeline_summary, f, indent=2)
    print_success(f"Results saved to {results_json_path}")
