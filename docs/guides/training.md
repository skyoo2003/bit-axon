# Training Guide

Fine-tune Bit-Axon on your own data using thermal-aware QLoRA on Apple Silicon. The entire pipeline — SFT, ORPO alignment, checkpointing — runs locally on a fanless MacBook Air M4 with 16 GB unified memory.

---

## Quick Start

```bash
bit-axon train data.json \
    --model-weights ./model \
    --tokenizer Qwen/Qwen2.5-3B
```

This loads the 4-bit quantized model, applies DoRA adapters (rank 8), tokenizes your dataset, and starts the training loop with thermal monitoring enabled.

---

## Overview

Bit-Axon supports two training modes:

| Mode | Purpose | Trainer Class | Loss |
|------|---------|---------------|------|
| **SFT** | Supervised fine-tuning | `Trainer` | Cross-entropy on assistant tokens |
| **ORPO** | Preference alignment | `ORPOTrainer` | NLL + odds-ratio penalty |

Both modes use **4-bit quantized base weights** with trainable **LoRA** or **DoRA** adapters. Only adapter parameters receive gradients — the base model stays frozen at NF4 precision, keeping total memory under 3.7 GB.

### Training Modules

All training logic lives under `src/bit_axon/training/`:

| Module | Purpose |
|--------|---------|
| `config.py` | `TrainingConfig` dataclass — all hyperparameters |
| `trainer.py` | `Trainer` class — SFT training loop |
| `lora.py` | `LoRALinear`, `apply_lora_to_model()` |
| `dora.py` | `DoRALinear` — weight-decomposed LoRA |
| `data.py` | `SFTDataset`, `AlpacaDataset`, `ORPODataset` |
| `cooling.py` | `CoolingScheduler`, `ThermalPolicy` |
| `orpo_trainer.py` | `ORPOTrainer` — preference alignment loop |
| `orpo_loss.py` | `compute_orpo_loss()`, `orpo_loss()` |
| `packing.py` | `SequencePacker` — concatenate examples into fixed-length sequences |
| `checkpoint.py` | `save_checkpoint()`, `load_checkpoint()`, `get_latest_checkpoint()` |
| `scheduler.py` | Cosine decay with linear warmup |
| `collate.py` | `iterate_batches()`, `BatchCollator` |
| `merging.py` | `merge_adapters()`, `load_and_merge()` |

---

## Dataset Formats

### SFT: Chat Messages (JSONL)

Each line is a single conversation in OpenAI-style messages format:

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain quantum entanglement."}, {"role": "assistant", "content": "Quantum entanglement is a phenomenon..."}]}
```

Supported roles: `system`, `user`, `assistant`. The `SFTDataset` applies the Qwen2.5 chat template and computes a binary loss mask — gradients only flow through assistant response tokens. System and user tokens are ignored.

### SFT: Alpaca Format

The `AlpacaDataset` class accepts standard Alpaca instruction data and converts it internally:

```json
{"instruction": "Summarize the following text.", "input": "Long text here...", "output": "Short summary."}
```

The `input` field is optional. When present, it is appended to the instruction with a double newline.

### ORPO: Preference Pairs

Each example contains a prompt, a chosen response, and a rejected response:

```json
{"prompt": [{"role": "user", "content": "Write a haiku about debugging."}], "chosen": [{"role": "assistant", "content": "Silent cursor blinks / Stack trace scrolls through the night / Bug found at line three"}], "rejected": [{"role": "assistant", "content": "Debugging is when you fix bugs in your code."}]}
```

A simpler string format is also accepted:

```json
{"prompt": "Write a haiku about debugging.", "chosen": "Silent cursor blinks...", "rejected": "Debugging is when you fix bugs..."}
```

!!! tip "Converting datasets"
    Use `bit-axon prepare` to convert HuggingFace datasets into JSONL:
    ```bash
    bit-axon prepare HuggingFaceH4/ultrachat --format messages --output train.jsonl --split train
    ```

---

## LoRA and DoRA Adapters

### LoRA (Low-Rank Adaptation)

Adds a trainable rank-`r` decomposition to target linear layers. The base weight `W` is frozen:

```
output = W·x + scale · (dropout(x) · A) · B
```

- `A` shape: `(input_dims, r)` — initialized with uniform noise
- `B` shape: `(r, output_dims)` — initialized to zeros (adapter starts as identity)
- `scale` default: 20.0

The `LoRALinear.from_base(linear, r=8, scale=20.0)` factory wraps an existing `nn.Linear` or `nn.QuantizedLinear` layer, preserving its weights.

### DoRA (Weight-Decomposed LoRA)

Enabled by default. Extends LoRA by preserving the magnitude of the original weight:

```
adapted_W = W + scale · B^T · A^T
output = (m / ||adapted_W||) · (W·x + scale · (dropout(x) · A) · B)
```

The magnitude vector `m` stores the per-row L2 norm of the frozen base weight (computed once at init). During forward, the output is re-normalized so the adapted weight preserves the original magnitude characteristics. This often produces better results than plain LoRA, especially on tasks requiring fine-grained output calibration.

!!! important "DoRA is the default"
    `use_dora=True` in `TrainingConfig`. Pass `--no-dora` to fall back to plain LoRA.

### Target Layers

Adapters are applied to these linear layer types:

| Target Layer | Location |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention Q/K/V/O projections |
| `in_proj`, `out_proj` | Combined attention projections |
| `gate_proj`, `up_proj`, `down_proj` | Feed-forward (expert) projections |
| `input_proj`, `output_proj` | SSM input/output projections |

### Excluded Layers

The following are **never** adapted, regardless of target matching:

| Exclusion | Match Type | Reason |
|---|---|---|
| `switch_mlp` | Path contains | MoE router internals |
| `lm_head` | Path contains | Output head — tied with embeddings |
| `gate` | Name equals | MoE gating |
| `shared_expert_gate` | Name equals | Shared expert gating |
| `x_proj` | Name equals | SSM-specific projection |
| `dt_proj` | Name equals | SSM delta projection |

The exclusion logic lives in `lora.py`:

```python
LORA_EXCLUDED_PATHS = ("switch_mlp", "lm_head")
LORA_EXCLUDED_NAMES = ("x_proj", "dt_proj", "gate", "shared_expert_gate")
```

### Applying Adapters

```python
from bit_axon.training.lora import apply_lora_to_model

# Apply DoRA (default) — returns list of wrapped layer paths
wrapped = apply_lora_to_model(model, rank=8, dropout=0.0, scale=20.0, use_dora=True)

# Apply plain LoRA
wrapped = apply_lora_to_model(model, rank=8, dropout=0.0, scale=20.0, use_dora=False)
```

### Fusing Adapters

After training, fuse adapters back into base weights:

```python
from bit_axon.training.merging import merge_adapters

# LoRA fuse: W_fused = W + scale · B^T · A^T
# DoRA fuse: W_fused = (m / ||W + delta||) · (W + delta)
merge_adapters(model)
```

---

## Sequence Packing

To maximize GPU utilization, the `SequencePacker` concatenates multiple short examples into fixed-length sequences of `max_seq_len` tokens:

```
Example 1 tokens | EOS | Example 2 tokens | EOS | Example 3 tokens | PAD
[   loss_mask=1  |  0  | loss_mask=1      |  0  | loss_mask=1      |  0 ]
```

The binary loss mask ensures:

- Separator EOS tokens (ID 151645) do not contribute to loss
- Padding tokens are masked with `-100` ignore index
- Only response tokens from the original examples produce gradients

Packing runs automatically inside `iterate_batches()`. No manual configuration needed.

```python
from bit_axon.training.packing import SequencePacker

packer = SequencePacker(max_seq_len=2048, eos_token_id=151645)

for token_ids, loss_mask in dataset:
    packed_batches = packer.add_example(token_ids, loss_mask)
    for batch in packed_batches:
        process(batch)  # PackedBatch(token_ids, loss_mask)

final = packer.flush()  # Remaining tokens, padded to max_seq_len
```

!!! info "ORPO does not use packing"
    Preference pairs are kept intact via `iterate_orpo_batches()`. Each chosen/rejected pair is processed as a unit.

---

## Thermal-Aware Training

Training on a fanless MacBook generates sustained heat. The `CoolingScheduler` reads SoC temperature via macOS `powermetrics` and applies a three-tier thermal policy before every training step.

### Thermal Tiers

| Tier | Temperature | Action |
|------|-------------|--------|
| Normal | < 75°C | Full-speed training |
| Warm | 75–85°C | `should_reduce_batch()` returns `True` (signal for batch reduction) |
| Hot | ≥ 85°C | Training **pauses** — sleeps in 0.5s intervals until temperature drops |
| Critical | ≥ 95°C | Training **stops immediately** with `ThermalShutdownError` |

### Python API

```python
from bit_axon.training.cooling import CoolingScheduler, ThermalPolicy, ThermalShutdownError

policy = ThermalPolicy(
    max_speed_temp=75.0,   # Batch reduction zone starts
    pause_temp=85.0,       # Training pauses above this
    stop_temp=95.0,        # Training stops above this
    pause_duration=0.5,    # Sleep interval during pause (seconds)
)

cooling = CoolingScheduler(monitor, policy)

# Called before each training step:
cooling.check_before_step(step)  # Pauses or raises ThermalShutdownError

# Check total time spent in thermal pauses:
print(f"Paused {cooling.total_pause_time:.1f}s for cooling")
```

### CLI Configuration

```bash
# Custom thermal thresholds
bit-axon train data.json --model-weights ./model --tokenizer Qwen/Qwen2.5-3B \
    --temp-pause 80 --temp-stop 90

# Disable thermal monitoring (use only on machines with active cooling)
bit-axon train data.json --model-weights ./model --tokenizer Qwen/Qwen2.5-3B \
    --no-thermal
```

!!! warning "Fanless MacBook Air"
    On a MacBook Air M4 with no fan, sustained training can push SoC temperature above 90°C. The default thresholds (pause at 85°C, stop at 95°C) are calibrated for safe operation. Do not disable thermal monitoring unless you have active cooling or are running a short test.

---

## ORPO Preference Optimization

ORPO (Odds Ratio Preference Optimization) performs simultaneous SFT and preference alignment. Unlike DPO, it requires **no reference model**, saving ~50% memory during alignment.

### Loss Function

The ORPO loss combines two terms:

```
L_total = L_NLL(chosen) - log σ(β · log_odds_ratio)

where:
  log_odds_ratio = log(p_chosen / (1 - p_chosen)) - log(p_rejected / (1 - p_rejected))
```

- **NLL loss**: Standard next-token prediction on the chosen response (SFT signal)
- **Odds-ratio penalty**: Log-sigmoid that pushes the model toward higher log-prob on chosen vs. rejected
- **β** (default `0.1`): Controls preference strength. Higher values push harder toward chosen responses

### Running ORPO

```python
from bit_axon.training.config import TrainingConfig
from bit_axon.training.data import ORPODataset
from bit_axon.training.orpo_trainer import ORPOTrainer

config = TrainingConfig(
    training_mode="orpo",
    beta=0.1,
    max_steps=2000,
)

dataset = ORPODataset("prefs.jsonl", tokenizer, max_seq_len=2048)
trainer = ORPOTrainer(model, config, dataset, cooling_scheduler=cooling)
result = trainer.train()

# Result keys:
# step, loss, grad_norm, chosen_reward, rejected_reward, reward_margin, reward_accuracy
```

The `ORPOTrainer` runs two forward passes per batch (chosen + rejected), computes averaged log-probabilities for both via `get_logps()`, and combines NLL loss with the odds-ratio penalty.

!!! tip "Monitor reward margin"
    `reward_margin = chosen_reward - rejected_reward`. A growing margin means the model is learning to prefer better responses. Use this to decide when to stop ORPO training.

---

## Training Configuration

### TrainingConfig Dataclass

All hyperparameters live in `TrainingConfig`:

```python
from bit_axon.training.config import TrainingConfig

config = TrainingConfig(
    # Optimizer
    learning_rate=1e-4,       # Peak LR after warmup
    weight_decay=0.01,        # AdamW weight decay
    warmup_steps=100,         # Linear warmup steps
    max_steps=10_000,         # Total training steps
    max_grad_norm=1.0,        # Gradient clipping threshold
    grad_accum_steps=4,       # Gradient accumulation steps

    # LoRA / DoRA
    lora_rank=8,              # Low-rank decomposition rank
    lora_dropout=0.0,         # Dropout on adapter path
    lora_scale=20.0,          # Adapter output scaling
    use_dora=True,            # Use DoRA (weight-decomposed LoRA)

    # ORPO alignment
    beta=0.1,                 # ORPO preference strength
    training_mode="sft",      # "sft" or "orpo"

    # Quantization
    quantize_bits=4,          # Base weight bit-width (NF4)
    quantize_group_size=64,   # Quantization group size

    # Data
    batch_size=1,             # Sequences per batch
    max_seq_len=2048,         # Packing target length

    # Checkpointing
    save_every=500,           # Save checkpoint every N steps
    eval_every=500,           # Evaluate every N steps
    output_dir="checkpoints", # Checkpoint directory

    # Thermal thresholds (°C)
    temp_max_speed=75.0,      # Batch reduction zone
    temp_pause=85.0,          # Pause training
    temp_stop=95.0,           # Stop training
    temp_poll_interval=1.0,   # Temperature poll interval (seconds)

    # Misc
    seed=42,                  # Random seed
)
```

### Learning Rate Schedule

Cosine decay with linear warmup:

- Steps 0–`warmup_steps`: LR ramps linearly from 0 to `learning_rate`
- Steps `warmup_steps`–`max_steps`: Cosine decay from `learning_rate` down to 0

### Batch Size and Gradient Accumulation

With `batch_size=1` and `grad_accum_steps=4`, the effective batch size is 4:

```
effective_batch_size = batch_size × grad_accum_steps = 1 × 4 = 4
```

Gradients accumulate over 4 forward passes before a single optimizer update. This keeps per-step memory low while maintaining a reasonable effective batch size.

### Full CLI Options

```bash
bit-axon train --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-weights` / `-w` | required | Path to model weights directory |
| `--tokenizer` / `-t` | `Qwen/Qwen2.5-3B` | Tokenizer identifier |
| `--val-data` | None | Validation JSONL file |
| `--lora-rank` | 8 | Adapter rank |
| `--lora-dropout` | 0.0 | Adapter dropout |
| `--lora-scale` | 20.0 | Adapter scaling |
| `--no-dora` | False | Use LoRA instead of DoRA |
| `--learning-rate` / `-lr` | 1e-4 | Peak learning rate |
| `--max-steps` | 10,000 | Total training steps |
| `--batch-size` | 1 | Sequences per batch |
| `--grad-accum-steps` | 4 | Gradient accumulation |
| `--max-seq-len` | 2048 | Maximum sequence length |
| `--warmup-steps` | 100 | Warmup steps |
| `--max-grad-norm` | 1.0 | Gradient clipping |
| `--seed` | 42 | Random seed |
| `--no-thermal` | False | Disable thermal monitoring |
| `--temp-pause` | 85.0 | Pause threshold (°C) |
| `--temp-stop` | 95.0 | Stop threshold (°C) |
| `--output-dir` / `-o` | `checkpoints` | Checkpoint directory |
| `--save-every` | 500 | Checkpoint interval |
| `--eval-every` | 500 | Evaluation interval |
| `--resume` | False | Resume from latest checkpoint |
| `--config-small` | False | Use small model for testing |

---

## Python API

### Full SFT Training Example

```python
import mlx.core as mx
from bit_axon import BitAxonConfig, BitAxonModel
from bit_axon.tokenizer import QwenTokenizerWrapper
from bit_axon.training import TrainingConfig, Trainer, apply_lora_to_model
from bit_axon.training.data import SFTDataset, CacheDataset
from bit_axon.training.cooling import CoolingScheduler, ThermalPolicy

# 1. Load model
model_config = BitAxonConfig()
model = BitAxonModel(model_config)

# 2. Apply DoRA adapters
wrapped_layers = apply_lora_to_model(
    model,
    rank=8,
    dropout=0.0,
    scale=20.0,
    use_dora=True,
)
mx.eval(model.parameters())

# 3. Prepare data
tokenizer = QwenTokenizerWrapper("Qwen/Qwen2.5-3B")
dataset = CacheDataset(SFTDataset("data.json", tokenizer, max_seq_len=2048))
val_dataset = SFTDataset("val.json", tokenizer, max_seq_len=2048)

# 4. Configure training
config = TrainingConfig(
    learning_rate=1e-4,
    max_steps=5000,
    grad_accum_steps=4,
    save_every=500,
    eval_every=500,
    output_dir="checkpoints/my-run",
)

# 5. Set up thermal monitoring
policy = ThermalPolicy(pause_temp=85.0, stop_temp=95.0)
cooling = CoolingScheduler(thermal_monitor, policy)

# 6. Train
trainer = Trainer(model, config, dataset, val_dataset, cooling_scheduler=cooling)
result = trainer.train()

print(f"Step {result['step']}: loss={result['loss']:.4f}, grad_norm={result['grad_norm']:.4f}")
```

### ORPO Training Example

```python
from bit_axon.training.config import TrainingConfig
from bit_axon.training.data import ORPODataset
from bit_axon.training.orpo_trainer import ORPOTrainer

config = TrainingConfig(
    training_mode="orpo",
    beta=0.1,
    max_steps=2000,
    save_every=500,
)

dataset = ORPODataset("prefs.jsonl", tokenizer, max_seq_len=2048)
trainer = ORPOTrainer(model, config, dataset, cooling_scheduler=cooling)
result = trainer.train()

print(f"Reward margin: {result['reward_margin']:.4f}")
print(f"Reward accuracy: {result['reward_accuracy']:.2f}")
```

### Manual Adapter Application

```python
from bit_axon.training.lora import apply_lora_to_model, LoRALinear
from bit_axon.training.dora import DoRALinear

# Apply to all target layers (returns list of wrapped paths)
wrapped = apply_lora_to_model(model, rank=8, scale=20.0, use_dora=True)
print(f"Wrapped {len(wrapped)} layers: {wrapped[:3]}...")

# Individual layer control
dora_layer = DoRALinear.from_base(existing_linear, r=8, scale=20.0)
lora_layer = LoRALinear.from_base(existing_linear, r=8, scale=20.0)

# Fuse adapter back into base weight
fused_linear = dora_layer.fuse()  # Re-normalizes with magnitude vector
```

---

## Checkpointing and Resume

### Automatic Checkpoints

Checkpoints save every `save_every` steps (default 500). Each checkpoint contains:

| File | Contents |
|------|----------|
| `adapters.safetensors` | All model parameters (adapter weights identifiable by `lora_a`, `lora_b`, `.m` keys) |
| `optimizer_state.safetensors` | AdamW momentum and variance buffers |
| `training_state.json` | `{"step": int, "loss": float}` |

A rotation policy keeps the 3 most recent checkpoints and deletes older ones:

```
checkpoints/
├── step_00000500/
│   ├── adapters.safetensors
│   ├── optimizer_state.safetensors
│   └── training_state.json
├── step_00001000/
│   ├── adapters.safetensors
│   ├── optimizer_state.safetensors
│   └── training_state.json
└── step_00001500/
    ├── adapters.safetensors
    ├── optimizer_state.safetensors
    └── training_state.json
```

### Resuming Training

The trainer finds the highest-step checkpoint in `output_dir`, restores adapter weights and optimizer state, and continues from that step:

```bash
bit-axon train data.json --model-weights ./model --tokenizer Qwen/Qwen2.5-3B \
    --output-dir ./checkpoints --resume
```

### Python API for Checkpoints

```python
from bit_axon.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    save_adapter_only,
)

# Save a checkpoint manually
ckpt_path = save_checkpoint(model, optimizer, step=1500, loss=1.23, output_dir="checkpoints")

# Find the latest checkpoint
latest = get_latest_checkpoint("checkpoints")  # Returns Path or None

# Load a checkpoint (restores model weights + optimizer state)
step, loss = load_checkpoint(model, optimizer, latest)

# Export only adapter weights for sharing or deployment
save_adapter_only(model, "my_adapter.safetensors")
```

---

## Adapter Merging

After training, fuse adapter weights back into the base model for deployment:

### CLI

```bash
bit-axon merge ./model \
    --adapter ./checkpoints/final_adapter.safetensors \
    --output ./merged-model
```

By default, the merged model is re-quantized to 4-bit. Pass `--no-re-quantize` to keep full-precision weights.

### Python API

```python
from bit_axon.training.merging import merge_adapters, save_merged_model, load_and_merge

# One-step: load base + adapter, merge, re-quantize, save
load_and_merge(
    base_model_path="./model",
    adapter_path="./checkpoints/final_adapter.safetensors",
    output_dir="./merged-model",
    quantize_after_merge=True,
    bits=4,
    group_size=64,
    lora_rank=8,
)

# Manual step-by-step
merge_adapters(model)
save_merged_model(model, "./merged-model", config=model_config)
```

---

## Caching

Wrap any dataset with `CacheDataset` to avoid redundant tokenization across epochs:

```python
from bit_axon.training.data import SFTDataset, CacheDataset

raw = SFTDataset("train.jsonl", tokenizer, max_seq_len=2048)
cached = CacheDataset(raw)

# First access tokenizes and caches; subsequent accesses hit the cache
for token_ids, loss_mask in cached:
    train_step(token_ids, loss_mask)
```

This is especially useful when `loop=True` in the batch iterator, which cycles through the dataset multiple times during training.

---

## Training Pipeline Summary

The full SFT pipeline executed by `bit-axon train`:

| Step | Action | Module |
|------|--------|--------|
| 1 | Build `BitAxonConfig` | `config.py` |
| 2 | Build `TrainingConfig` | `training/config.py` |
| 3 | Load `BitAxonModel` and weights | `model.py` |
| 4 | Quantize to 4-bit (NF4) | `quantization/nf4.py` |
| 5 | Freeze all weights, apply LoRA/DoRA adapters | `training/lora.py`, `training/dora.py` |
| 6 | Load tokenizer and datasets | `training/data.py` |
| 7 | Start `ThermalMonitor` via `powermetrics` | `profiling/thermal.py` |
| 8 | Run training loop with gradient accumulation | `training/trainer.py` |
| 9 | Save final adapter weights | `training/checkpoint.py` |
| 10 | Print results | CLI |

---

## See also

- [CLI Reference](../cli/reference.md) — Full command documentation for `train`, `merge`, `quantize`
- [Quantization Guide](quantization.md) — NF4 quantization, QLoRA training memory, and merge workflows
- [Architecture](../architecture/index.md) — Model design, sandwich structure, and memory layout
- [Thermal Training Paper](../papers/thermal-training.md) — Mathematical foundations of the cooling scheduler
- [Inference Guide](inference.md) — Run the fine-tuned model with CLI or Python API
- [API — Training](../api/training.md) — `Trainer`, `TrainingConfig`, `LoRALinear`, `DoRALinear` Python API
