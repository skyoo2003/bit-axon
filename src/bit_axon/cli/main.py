"""Bit-Axon CLI entry point."""

from __future__ import annotations

from typing import Annotated

import typer

from bit_axon import __version__

app = typer.Typer(
    name="bit-axon",
    no_args_is_help=True,
    add_completion=True,
    context_settings={"max_content_width": 120, "help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"bit-axon {__version__}")
        raise typer.Exit()


@app.callback()
def _callback(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", callback=version_callback, is_eager=True, help="Show version"),
    ] = None,
) -> None:
    """Bit-Axon: Run, fine-tune, quantize, and benchmark LLMs on Apple Silicon."""


@app.command()
def run(
    prompt: Annotated[str | None, typer.Argument(help="Text prompt")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Model path or HF repo ID")] = "skyoo2003/bit-axon",
    tokenizer: Annotated[str | None, typer.Option("--tokenizer", "-t", help="Tokenizer path or HF repo ID")] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Max tokens to generate")] = 512,
    temperature: Annotated[float, typer.Option("--temperature", help="Sampling temperature")] = 0.6,
    top_k: Annotated[int, typer.Option("--top-k", help="Top-k filtering")] = 50,
    top_p: Annotated[float, typer.Option("--top-p", help="Nucleus sampling threshold")] = 0.95,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    config_small: Annotated[bool, typer.Option("--config-small", help="Use small model for testing")] = False,
    chat: Annotated[bool, typer.Option("--chat", "-c", help="Interactive chat mode")] = False,
    no_stream: Annotated[bool, typer.Option("--no-stream", help="Disable streaming output")] = False,
) -> None:
    """Run LLM inference on a prompt."""
    from bit_axon.cli.run import run_inference

    run_inference(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        config_small=config_small,
        chat=chat,
        no_stream=no_stream,
    )


@app.command(name="train")
def train(
    data: Annotated[str, typer.Argument(help="Path to training JSONL file")],
    model_weights: Annotated[str, typer.Option("--model-weights", "-w", help="Model weights directory or HuggingFace repo ID")] = "skyoo2003/bit-axon",
    val_data: Annotated[str | None, typer.Option("--val-data", help="Path to validation JSONL file")] = None,
    tokenizer: Annotated[str, typer.Option("--tokenizer", "-t", help="Tokenizer name or path")] = "Qwen/Qwen2.5-3B",
    config_small: Annotated[bool, typer.Option("--config-small", help="Use small config for testing")] = False,
    lora_rank: Annotated[int, typer.Option("--lora-rank", help="LoRA adapter rank")] = 8,
    lora_dropout: Annotated[float, typer.Option("--lora-dropout", help="LoRA dropout")] = 0.0,
    lora_scale: Annotated[float, typer.Option("--lora-scale", help="LoRA scale")] = 20.0,
    no_dora: Annotated[bool, typer.Option("--no-dora", help="Use LoRA instead of DoRA")] = False,
    learning_rate: Annotated[float, typer.Option("--learning-rate", "-lr", help="Learning rate")] = 1e-4,
    max_steps: Annotated[int, typer.Option("--max-steps", help="Maximum training steps")] = 10_000,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 1,
    grad_accum_steps: Annotated[int, typer.Option("--grad-accum-steps", help="Gradient accumulation steps")] = 4,
    max_seq_len: Annotated[int, typer.Option("--max-seq-len", help="Maximum sequence length")] = 512,
    warmup_steps: Annotated[int, typer.Option("--warmup-steps", help="Warmup steps")] = 100,
    max_grad_norm: Annotated[float, typer.Option("--max-grad-norm", help="Max gradient norm")] = 1.0,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    no_thermal: Annotated[bool, typer.Option("--no-thermal", help="Disable thermal monitoring")] = False,
    temp_pause: Annotated[float, typer.Option("--temp-pause", help="Temperature pause threshold (°C)")] = 85.0,
    temp_stop: Annotated[float, typer.Option("--temp-stop", help="Temperature stop threshold (°C)")] = 95.0,
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory")] = "checkpoints",
    save_every: Annotated[int, typer.Option("--save-every", help="Save checkpoint every N steps")] = 500,
    eval_every: Annotated[int, typer.Option("--eval-every", help="Evaluate every N steps")] = 500,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from latest checkpoint")] = False,
) -> None:
    """Fine-tune with SFT (thermal-aware QLoRA)."""
    from bit_axon.cli.train import train_cmd

    train_cmd(
        data=data,
        val_data=val_data,
        tokenizer=tokenizer,
        model_weights=model_weights,
        config_small=config_small,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        lora_scale=lora_scale,
        no_dora=no_dora,
        learning_rate=learning_rate,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        max_seq_len=max_seq_len,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        seed=seed,
        no_thermal=no_thermal,
        temp_pause=temp_pause,
        temp_stop=temp_stop,
        output_dir=output_dir,
        save_every=save_every,
        eval_every=eval_every,
        resume=resume,
    )


@app.command()
def quantize(
    model_path: Annotated[str, typer.Argument(help="Path to model directory")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "",
    bits: Annotated[int, typer.Option("--bits", "-b", help="Quantization bit-width")] = 4,
    group_size: Annotated[int, typer.Option("--group-size", "-g", help="Quantization group size")] = 64,
    config_small: Annotated[bool, typer.Option("--config-small", help="Use small model for testing")] = False,
) -> None:
    """Quantize model weights to lower bit-width."""
    from bit_axon.cli.quantize import quantize_cmd

    quantize_cmd(model_path, output, bits, group_size, config_small)


@app.command()
def merge(
    base_model: Annotated[str, typer.Argument(help="Base model directory")],
    adapter: Annotated[str, typer.Option("--adapter", "-a", help="Adapter weights path (.safetensors)")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "",
    no_re_quantize: Annotated[bool, typer.Option("--no-re-quantize", help="Skip re-quantization after merge")] = False,
    bits: Annotated[int, typer.Option("--bits", "-b", help="Quantization bit-width")] = 4,
    group_size: Annotated[int, typer.Option("--group-size", "-g", help="Quantization group size")] = 64,
    lora_rank: Annotated[int, typer.Option("--lora-rank", "-r", help="LoRA rank")] = 8,
) -> None:
    """Merge LoRA/DoRA adapter weights into a base model."""
    from bit_axon.cli.merge import merge_cmd

    merge_cmd(base_model, adapter, output, no_re_quantize, bits, group_size, lora_rank)


@app.command()
def benchmark(
    seq_lengths: Annotated[str, typer.Option("--seq-lengths", "-s", help="Comma-separated sequence lengths")] = "128,512,1024,2048",
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 1,
    warmup: Annotated[int, typer.Option("--warmup", "-w", help="Warmup iterations")] = 2,
    iterations: Annotated[int, typer.Option("--iterations", "-i", help="Timed iterations")] = 5,
    config_small: Annotated[bool, typer.Option("--config-small", help="Use small config for testing")] = False,
) -> None:
    """Benchmark model performance across sequence lengths."""
    from bit_axon.cli.benchmark import benchmark_cmd

    benchmark_cmd(seq_lengths, batch_size, warmup, iterations, config_small)


@app.command()
def download(
    repo_id: Annotated[str, typer.Argument(help="HuggingFace repository ID")] = "skyoo2003/bit-axon",
    local_dir: Annotated[str | None, typer.Option("--local-dir", "-d", help="Local directory to download to")] = None,
    include: Annotated[list[str] | None, typer.Option("--include", help="Glob patterns to include")] = None,
) -> None:
    """Download model weights from HuggingFace Hub."""
    from bit_axon.cli.download import download_cmd

    download_cmd(repo_id, local_dir, include)


@app.command()
def evaluate(
    model_path: Annotated[str, typer.Argument(help="Path to model weights")],
    config_small: Annotated[bool, typer.Option("--config-small", help="Use small model for testing")] = False,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Max tokens to evaluate")] = 100_000,
    seq_length: Annotated[int, typer.Option("--seq-length", help="Sequence length for evaluation")] = 2048,
    tokenizer: Annotated[str | None, typer.Option("--tokenizer", "-t", help="Tokenizer path or HF repo ID")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 4,
    benchmarks: Annotated[
        str | None,
        typer.Option("--benchmarks", "-b", help="Comma-separated benchmarks: mmlu,gsm8k,arc_challenge,arc_easy,hellaswag,winogrande"),
    ] = None,
    benchmark_limit: Annotated[
        int | None,
        typer.Option("--benchmark-limit", help="Max samples per benchmark"),
    ] = None,
) -> None:
    """Evaluate model perplexity on WikiText-103."""
    if benchmarks is not None and tokenizer is None:
        from bit_axon.cli._console import print_error

        print_error("--tokenizer is required when using --benchmarks")
        raise SystemExit(1)

    if benchmarks is not None:
        from bit_axon.cli.evaluate import evaluate_benchmarks_cmd

        if tokenizer is None:
            raise SystemExit(1)
        benchmark_names = [b.strip() for b in benchmarks.split(",")]
        evaluate_benchmarks_cmd(
            model_path=model_path,
            config_small=config_small,
            tokenizer=tokenizer,
            benchmarks=benchmark_names,
            benchmark_limit=benchmark_limit,
            max_tokens=max_tokens,
        )
    else:
        from bit_axon.cli.evaluate import evaluate_cmd

        evaluate_cmd(model_path, config_small, max_tokens, seq_length, tokenizer, batch_size)


@app.command(name="port-weights")
def port_weights(
    output: Annotated[str, typer.Argument(help="Output directory for ported weights")],
    config_small: Annotated[bool, typer.Option("--config-small", help="Use small config with mock weights")] = False,
) -> None:
    """Port Qwen2.5-3B weights to Bit-Axon model format."""
    from bit_axon.cli.port_weights import port_weights_cmd

    port_weights_cmd(output, config_small)


@app.command()
def pipeline(
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for pipeline artifacts")] = "pipeline_output",
    max_steps: Annotated[int, typer.Option("--max-steps", help="Maximum SFT training steps")] = 100,
    orpo_steps: Annotated[int, typer.Option("--orpo-steps", help="Maximum ORPO training steps")] = 50,
    max_seq_len: Annotated[int, typer.Option("--max-seq-len", help="Maximum sequence length")] = 32,
    lora_rank: Annotated[int, typer.Option("--lora-rank", help="LoRA adapter rank")] = 8,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 1,
    sft_data: Annotated[
        str | None, typer.Option("--sft-data", help="SFT dataset: preset name (ultrachat/alpaca/openorca), HuggingFace ID, or local JSONL path")
    ] = None,
    sft_split: Annotated[str, typer.Option("--sft-split", help="SFT dataset split")] = "train",
    sft_limit: Annotated[int | None, typer.Option("--sft-limit", help="Max SFT rows to load")] = None,
    orpo_data: Annotated[
        str | None, typer.Option("--orpo-data", help="ORPO dataset: preset name (ultrafeedback/hh-rlhf), HuggingFace ID, or local JSONL path")
    ] = None,
    orpo_split: Annotated[str, typer.Option("--orpo-split", help="ORPO dataset split")] = "train",
    orpo_limit: Annotated[int | None, typer.Option("--orpo-limit", help="Max ORPO rows to load")] = None,
    tokenizer: Annotated[str | None, typer.Option("--tokenizer", "-t", help="Tokenizer name or path (required when using real datasets)")] = None,
) -> None:
    """Run full ML pipeline: SFT, merge, quantize, evaluate, ORPO (supports real datasets)."""
    from bit_axon.cli.pipeline import pipeline_cmd

    pipeline_cmd(
        output_dir=output_dir,
        max_steps=max_steps,
        orpo_steps=orpo_steps,
        max_seq_len=max_seq_len,
        lora_rank=lora_rank,
        batch_size=batch_size,
        sft_data=sft_data,
        sft_split=sft_split,
        sft_limit=sft_limit,
        orpo_data=orpo_data,
        orpo_split=orpo_split,
        orpo_limit=orpo_limit,
        tokenizer=tokenizer,
    )


@app.command()
def prepare(
    dataset: Annotated[str, typer.Argument(help="HuggingFace dataset identifier")],
    format: Annotated[str, typer.Option("--format", "-f", help="Output format")] = "messages",
    output: Annotated[str, typer.Option("--output", "-o", help="Output JSONL file path")] = "",
    split: Annotated[str, typer.Option("--split", help="Dataset split")] = "train",
    limit: Annotated[int | None, typer.Option("--limit", help="Max rows to convert")] = None,
) -> None:
    """Convert HuggingFace dataset to JSONL for training."""
    from bit_axon.cli.prepare import prepare_cmd

    prepare_cmd(dataset, format, output, split, limit)


def main() -> None:
    """Entry point for the bit-axon CLI."""
    app()
