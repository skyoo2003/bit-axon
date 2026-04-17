# CLI Reference

Bit-Axon ships a single `bit-axon` entry point powered by [Typer](https://typer.tiangolo.com/). Every subcommand is typed with `typer.Argument` and `typer.Option` annotations, so `bit-axon --help` and `bit-axon <command> --help` always reflect the latest signatures.

---

## Inference

### `bit-axon run`

Generate text from a prompt (or from stdin).

```bash
bit-axon run "Explain entropy in one sentence"
bit-axon run --chat
echo "What is 2+2?" | bit-axon run
```

**Usage**

```
bit-axon run [PROMPT] [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--model` / `-m` | `str` | `skyoo2003/bit-axon` | Model identifier (Hugging Face repo or local path) |
| `--tokenizer` / `-t` | `str` | `None` | Override tokenizer (defaults to the model's own) |
| `--max-tokens` | `int` | `512` | Maximum tokens to generate |
| `--temperature` | `float` | `0.6` | Sampling temperature |
| `--top-k` | `int` | `50` | Top-K filtering |
| `--top-p` | `float` | `0.95` | Nucleus (top-p) filtering threshold |
| `--seed` | `int` | `None` | Random seed for reproducible output |
| `--chat` / `-c` | `bool` | `False` | Launch an interactive chat session |
| `--no-stream` | `bool` | `False` | Print the full response at once instead of streaming tokens |
| `--config-small` | `bool` | `False` | Use the small-model configuration |

---

## Training

### `bit-axon train`

Fine-tune a model on a JSONL dataset using LoRA.

```bash
bit-axon train data/train.jsonl -w skyoo2003/bit-axon -o checkpoints/my-run
```

**Usage**

```
bit-axon train DATA [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--model-weights` / `-w` | `str` | *required* | Base model weights to fine-tune |
| `--val-data` | `str` | `None` | Validation dataset path |
| `--tokenizer` / `-t` | `str` | `Qwen/Qwen2.5-3B` | Tokenizer identifier |
| `--lora-rank` | `int` | `8` | LoRA rank |
| `--lora-dropout` | `float` | `0.0` | LoRA dropout probability |
| `--lora-scale` | `float` | `20.0` | LoRA scaling factor |
| `--no-dora` | `bool` | `False` | Disable DoRA (use standard LoRA instead) |
| `--learning-rate` / `-lr` | `float` | `1e-4` | Peak learning rate |
| `--max-steps` | `int` | `10000` | Maximum training steps |
| `--batch-size` | `int` | `1` | Per-device batch size |
| `--grad-accum-steps` | `int` | `4` | Gradient accumulation steps |
| `--max-seq-len` | `int` | `2048` | Maximum sequence length |
| `--warmup-steps` | `int` | `100` | Linear warmup steps |
| `--max-grad-norm` | `float` | `1.0` | Gradient clipping norm |
| `--seed` | `int` | `42` | Random seed |
| `--no-thermal` | `bool` | `False` | Disable thermal management |
| `--temp-pause` | `float` | `85.0` | Temperature (°C) at which training pauses |
| `--temp-stop` | `float` | `95.0` | Temperature (°C) at which training stops |
| `--output-dir` / `-o` | `str` | `checkpoints` | Directory to save checkpoints |
| `--save-every` | `int` | `500` | Save a checkpoint every N steps |
| `--eval-every` | `int` | `500` | Run evaluation every N steps |
| `--resume` | `bool` | `False` | Resume from the latest checkpoint |
| `--config-small` | `bool` | `False` | Use the small-model configuration |

**Training pipeline (10 steps)**

1. Load the base model weights and tokenizer.
2. Apply LoRA (or DoRA) adapters to the target modules.
3. Load and tokenize the training dataset.
4. Set up the optimizer with the configured learning rate and warmup schedule.
5. Configure gradient accumulation to simulate a larger effective batch size.
6. Optionally enable thermal monitoring to pause or stop training if the GPU exceeds the configured thresholds.
7. Run the training loop, evaluating and saving checkpoints at the configured intervals.
8. If interrupted or completed, write a final checkpoint.
9. Log training metrics (loss, learning rate, throughput) at each step.
10. Exit and report the path to the best or latest checkpoint.

---

## Model Management

### `bit-axon quantize`

Quantize a model to lower precision (e.g. 4-bit integer).

```bash
bit-axon quantize skyoo2003/bit-axon -o models/bit-axon-q4 -b 4 -g 64
```

**Usage**

```
bit-axon quantize MODEL_PATH [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--output` / `-o` | `str` | `""` | Output directory for the quantized model |
| `--bits` / `-b` | `int` | `4` | Quantization bit-width |
| `--group-size` / `-g` | `int` | `64` | Group size for grouped quantization |
| `--config-small` | `bool` | `False` | Use the small-model configuration |

### `bit-axon merge`

Merge a LoRA adapter back into the base model, optionally re-quantizing the result.

```bash
bit-axon merge skyoo2003/bit-axon -a checkpoints/my-run/adapter -o models/merged
```

**Usage**

```
bit-axon merge BASE_MODEL [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--adapter` / `-a` | `str` | *required* | Path to the LoRA adapter to merge |
| `--output` / `-o` | `str` | `""` | Output directory for the merged model |
| `--no-re-quantize` | `bool` | `False` | Skip re-quantization after merging |
| `--bits` / `-b` | `int` | `4` | Bit-width if re-quantizing |
| `--group-size` / `-g` | `int` | `64` | Group size if re-quantizing |
| `--lora-rank` / `-r` | `int` | `8` | LoRA rank of the adapter |

### `bit-axon download`

Download a model (or dataset) from Hugging Face.

```bash
bit-axon download skyoo2003/bit-axon -d models/bit-axon
```

**Usage**

```
bit-axon download [REPO_ID] [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--local-dir` / `-d` | `str` | `None` | Local directory to save files into |
| `--include` | `list[str]` | `None` | Glob patterns of files to include |

### `bit-axon upload`

Upload a model to Hugging Face Hub.

```bash
bit-axon upload models/merged -r skyoo2003/bit-axon -t Qwen/Qwen2.5-3B
```

**Usage**

```
bit-axon upload MODEL_PATH [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--repo-id` / `-r` | `str` | `skyoo2003/bit-axon` | Hugging Face repository ID |
| `--tokenizer` / `-t` | `str` | `Qwen/Qwen2.5-3B` | Tokenizer name or path |
| `--private` | `bool` | `False` | Create a private repository |
| `--commit-message` / `-m` | `str` | `Upload Bit-Axon 3.2B model` | Commit message for the upload |
| `--benchmark-results` | `str` | `None` | Comma-separated benchmark results, e.g. `mmlu=0.45,gsm8k=0.32` |

### `bit-axon port-weights`

Port model weights to the Bit-Axon format.

```bash
bit-axon port-weights models/bit-axon-ported
```

**Usage**

```
bit-axon port-weights OUTPUT [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--config-small` | `bool` | `False` | Use the small-model configuration |

---

## Evaluation

### `bit-axon benchmark`

Measure generation throughput across multiple sequence lengths.

```bash
bit-axon benchmark -s "128,512,1024,2048" -i 10
```

**Usage**

```
bit-axon benchmark [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--seq-lengths` / `-s` | `str` | `128,512,1024,2048` | Comma-separated sequence lengths to benchmark |
| `--batch-size` | `int` | `1` | Batch size for each benchmark run |
| `--warmup` / `-w` | `int` | `2` | Warmup iterations (excluded from timing) |
| `--iterations` / `-i` | `int` | `5` | Timed iterations per sequence length |
| `--config-small` | `bool` | `False` | Use the small-model configuration |

### `bit-axon evaluate`

Run evaluation on a model and print aggregate metrics.

```bash
bit-axon evaluate models/bit-axon-q4 -t Qwen/Qwen2.5-3B
```

**Usage**

```
bit-axon evaluate MODEL_PATH [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--max-tokens` | `int` | `100000` | Token budget for the full evaluation run |
| `--seq-length` | `int` | `2048` | Maximum sequence length |
| `--tokenizer` / `-t` | `str` | `None` | Override tokenizer |
| `--batch-size` | `int` | `4` | Evaluation batch size |
| `--config-small` | `bool` | `False` | Use the small-model configuration |

### `bit-axon pipeline`

Run the end-to-end training and alignment pipeline on a built-in dataset.

```bash
bit-axon pipeline -o pipeline_output --max-steps 200
```

**Usage**

```
bit-axon pipeline [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--output-dir` / `-o` | `str` | `pipeline_output` | Root output directory |
| `--max-steps` | `int` | `100` | Maximum SFT training steps |
| `--orpo-steps` | `int` | `50` | Maximum ORPO alignment steps |
| `--max-seq-len` | `int` | `32` | Maximum sequence length |
| `--lora-rank` | `int` | `8` | LoRA rank for both SFT and ORPO phases |
| `--batch-size` | `int` | `1` | Per-device batch size |

**Pipeline stages (7 stages)**

1. Download (or verify) the built-in training dataset.
2. Preprocess and tokenize the data for supervised fine-tuning.
3. Run SFT (supervised fine-tuning) with LoRA for the configured number of steps.
4. Generate preference pairs from the SFT checkpoint.
5. Run ORPO (odds-ratio preference optimization) on the preference pairs.
6. Merge the final adapter weights back into the base model.
7. Write the finished model and a summary report to the output directory.

---

## See also

- [Training Guide](../guides/training.md) — Full fine-tuning walkthrough with examples
- [Inference Guide](../guides/inference.md) — Generation, chat mode, and streaming
- [Quantization Guide](../guides/quantization.md) — Weight quantization and adapter merging
- [Benchmarking Guide](../guides/benchmarking.md) — Performance measurement and interpretation
- [API Reference](../api/index.md) — Python API for all modules
