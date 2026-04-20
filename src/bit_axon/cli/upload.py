"""Upload Bit-Axon model to HuggingFace Hub."""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download
from rich.console import Console

from bit_axon.cli._console import print_error, print_info, print_success

console = Console()

_MODEL_CARD_TEMPLATE = """\
---
language:
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
model_type: bit-axon
tags:
- llm
- mlx
- ssm
- moe
- mixture-of-experts
- state-space-model
- apple-silicon
- quantized
- bit-axon
---

# Bit-Axon {param_count}

## Architecture

Bit-Axon is a hybrid SSM + MoE language model optimized for Apple Silicon (MLX framework).

**Sandwich Architecture:**
- **Layers 0-{ssm_end}**: Pure Axon-SSM (Mamba-style state space model, linear recurrence)
- **Layers {swa_start}-{swa_end}**: Sliding Window Attention (window={swa_window}) + Shared-Expert MoE ({n_experts} experts, top-{top_k})
- **Layers {moe_start}-{moe_end}**: Axon-SSM + Shared-Expert MoE ({n_experts} experts, top-{top_k})

## Model Details

- **Parameters**: ~{param_count}
- **Hidden dim**: {hidden_dim}
- **Layers**: {n_layers}
- **Vocab size**: {vocab_size}
- **Context length**: {max_seq_len:,}
- **Quantization**: {quantization_info}
- **Framework**: MLX (Apple Silicon)

## Usage

```bash
# Install
pip install bit-axon

# Download & run
bit-axon download {repo_id}
bit-axon run "Your prompt here" --model {repo_id} --tokenizer {tokenizer}

# Or use Python
from bit_axon.inference.loader import load_model
from bit_axon.tokenizer import QwenTokenizerWrapper

model = load_model("{repo_id}")
tokenizer = QwenTokenizerWrapper("{tokenizer}")
```

## Evaluation

{benchmark_section}

## Training

Initialized from Qwen2.5-3B weights via structured weight porting:
- Embedding extraction with vocabulary mapping (152K → 32K)
- RMSNorm dimension padding (2048 → 2560)
- MLP → shared-expert projection with structured truncation
- Routed expert initialization from shared expert + perturbation

Fine-tuned with QLoRA (rank=16) on UltraChat 200K, followed by ORPO preference alignment.
"""


def _generate_model_card(
    config: object,
    repo_id: str,
    tokenizer: str,
    benchmark_results: dict[str, float] | None = None,
    model_path: str | None = None,
) -> str:
    d = config.__dict__ if hasattr(config, "__dict__") else asdict(config)

    # Auto-load evaluation results from model directory
    if benchmark_results is None and model_path is not None:
        eval_json = Path(model_path) / "evaluation_results.json"
        if eval_json.exists():
            with open(eval_json) as f:
                eval_data = json.load(f)
            benchmark_results = {b["name"]: b["accuracy"] for b in eval_data.get("benchmarks", [])}

    hidden_dim = d.get("hidden_dim", 2560)
    n_layers = d.get("num_layers", 24)
    vocab_size = d.get("vocab_size", 32000)
    max_seq_len = d.get("max_seq_len", 65536)
    n_experts = d.get("moe_num_experts", 8)
    top_k = d.get("moe_top_k", 2)
    swa_window = d.get("swa_window_size", 4096)

    total_params = _estimate_params(d)
    param_count = _format_param_count(total_params)

    ssm_end = n_layers // 3 - 1
    swa_start = n_layers // 3
    swa_end = 2 * n_layers // 3 - 1
    moe_start = 2 * n_layers // 3
    moe_end = n_layers - 1

    if benchmark_results:
        lines = ["| Benchmark | Accuracy |", "|-----------|----------|"]
        for name, acc in benchmark_results.items():
            lines.append(f"| {name} | {acc:.1%} |")
        benchmark_section = "\n".join(lines)
    else:
        benchmark_section = "_Evaluation results pending._"

    return _MODEL_CARD_TEMPLATE.format(
        param_count=param_count,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        n_experts=n_experts,
        top_k=top_k,
        swa_window=swa_window,
        quantization_info="NF4 (4-bit, group_size=64)",
        ssm_end=ssm_end,
        swa_start=swa_start,
        swa_end=swa_end,
        moe_start=moe_start,
        moe_end=moe_end,
        repo_id=repo_id,
        tokenizer=tokenizer,
        benchmark_section=benchmark_section,
    )


def _format_param_count(total_params: int) -> str:
    """Render a parameter count as a short human string.

    <1M → "{N}K", <1B → "{N}M", otherwise "{N}B". The default "{X:.1f}B"
    collapses small configs (e.g. 5M params) to "0.0B" which is misleading
    on model cards for the small/medium presets.
    """
    if total_params < 1_000_000:
        return f"{total_params / 1_000:.0f}K"
    if total_params < 1_000_000_000:
        return f"{total_params / 1_000_000:.0f}M"
    return f"{total_params / 1_000_000_000:.1f}B"


def _estimate_params(config_dict: dict) -> int:
    h = config_dict.get("hidden_dim", 2560)
    n_layers = config_dict.get("num_layers", 24)
    vocab = config_dict.get("vocab_size", 32000)
    ssm_expand = config_dict.get("ssm_expand", 3)
    ssm_d_state = config_dict.get("ssm_d_state", 16)
    moe_inter = config_dict.get("moe_intermediate_dim", 4096)
    n_experts = config_dict.get("moe_num_experts", 8)
    d_source = config_dict.get("d_source_model", 2048)

    embed = vocab * d_source
    ssm_per_layer = h * ssm_expand * h * 3 + h * ssm_d_state * 3
    swa_per_layer = 4 * h * h
    shared_expert = 3 * h * moe_inter
    routed_expert = n_experts * 3 * h * moe_inter
    moe_gate = h * n_experts
    rms_norm_per_layer = 2 * h

    third = n_layers // 3
    total = embed
    total += third * (ssm_per_layer + rms_norm_per_layer)
    total += third * (ssm_per_layer + swa_per_layer + shared_expert + routed_expert + moe_gate + 2 * rms_norm_per_layer)
    total += third * (ssm_per_layer + shared_expert + routed_expert + moe_gate + 2 * rms_norm_per_layer)
    total += h * d_source
    return total


def _download_tokenizer_files(tokenizer_id: str, target_dir: Path) -> None:
    files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]
    for fname in files:
        try:
            local_path = hf_hub_download(repo_id=tokenizer_id, filename=fname)
            shutil.copy2(local_path, target_dir / fname)
            print_info(f"Copied {fname}")
        except Exception:
            print_info(f"Skipped {fname} (not found in {tokenizer_id})")


def stage_upload_dir(
    model_path: str,
    repo_id: str,
    tokenizer: str = "Qwen/Qwen2.5-3B",
    benchmark_results: str | None = None,
    upload_dir_name: str = "_hf_upload",
) -> Path:
    """Stage a HuggingFace-ready upload directory without pushing.

    Copies weights, config, tokenizer files, and renders a model card into
    ``<model_path>/<upload_dir_name>/``. Returns the staged path.
    """
    from bit_axon.config import BitAxonConfig

    model_dir = Path(model_path)
    if not model_dir.exists():
        print_error(f"Model directory not found: {model_path}")
        raise SystemExit(1)

    print_info(f"Staging upload folder for {repo_id}...")

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = BitAxonConfig(**json.load(f))
        print_success(f"Loaded config: hidden_dim={config.hidden_dim}, layers={config.num_layers}")
    else:
        config = BitAxonConfig()
        print_info("Using default BitAxonConfig")

    upload_dir = model_dir / upload_dir_name
    upload_dir.mkdir(parents=True, exist_ok=True)

    for sf in model_dir.glob("*.safetensors"):
        shutil.copy2(sf, upload_dir / sf.name)
        print_info(f"Copied {sf.name}")

    if config_path.exists():
        shutil.copy2(config_path, upload_dir / "config.json")
    else:
        with open(upload_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

    print_info(f"Downloading tokenizer files from {tokenizer}...")
    try:
        _download_tokenizer_files(tokenizer, upload_dir)
    except Exception as e:
        print_info(f"Tokenizer download skipped: {e}")

    # Copy LICENSE / NOTICE from the repo root so the HF repo carries the
    # same legal files the project ships with, not just the license field
    # declared in the model card frontmatter.
    repo_root = Path(__file__).resolve().parents[4]
    for fname in ("LICENSE", "NOTICE"):
        src = repo_root / fname
        if src.exists():
            shutil.copy2(src, upload_dir / fname)
            print_info(f"Copied {fname}")

    bench_dict: dict[str, float] | None = None
    if benchmark_results is not None:
        bench_dict = {}
        for pair in benchmark_results.split(","):
            if "=" not in pair:
                raise ValueError(f"Malformed benchmark result entry: {pair!r}. Expected format 'name=score', e.g. 'mmlu=0.45'.")
            name, acc = pair.strip().split("=")
            bench_dict[name.strip()] = float(acc.strip())
        print_info(f"Benchmark results: {bench_dict}")

    readme = _generate_model_card(config, repo_id, tokenizer, bench_dict, model_path=model_path)
    (upload_dir / "README.md").write_text(readme)
    print_success(f"Staged upload dir: {upload_dir}")

    return upload_dir


def upload_cmd(
    model_path: str,
    repo_id: str,
    tokenizer: str = "Qwen/Qwen2.5-3B",
    private: bool = False,
    commit_message: str = "Upload Bit-Axon 3.2B model",
    benchmark_results: str | None = None,
) -> None:
    """Upload model to HuggingFace Hub."""
    upload_dir = stage_upload_dir(model_path, repo_id, tokenizer, benchmark_results)

    if not (upload_dir / "tokenizer.json").exists():
        print_error("tokenizer.json not found in upload directory. Upload aborted.")
        raise SystemExit(1)

    api = HfApi()
    print_info(f"Creating repo {repo_id} (private={private})...")
    create_repo(repo_id=repo_id, private=private, exist_ok=True)

    print_info("Uploading files to HuggingFace Hub...")
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        commit_message=commit_message,
    )

    print_success(f"Model uploaded to https://huggingface.co/{repo_id}")
    with contextlib.suppress(OSError):
        shutil.rmtree(upload_dir)
