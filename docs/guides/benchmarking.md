# Profiling & Benchmarking

Bit-Axon ships with a built-in profiling toolkit for measuring memory, speed, and thermal behavior on Apple Silicon. The tools work through two interfaces: a CLI command for quick results and a Python API for custom workflows.

## CLI Benchmark

The fastest way to get a performance snapshot:

```bash
bit-axon benchmark --seq-lengths 128,512,1024,2048
```

This runs the full `BenchmarkSuite` across four sequence lengths and prints a Rich table with tokens/sec, latency, memory usage, and SoC temperature.

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--seq-lengths, -s` | `128,512,1024,2048` | Comma-separated sequence lengths |
| `--batch-size` | `1` | Batch size per forward pass |
| `--warmup, -w` | `2` | Untimed warmup iterations |
| `--iterations, -i` | `5` | Timed iterations per sequence length |
| `--config-small` | `false` | Use a tiny config (4 layers, 256 dim) for quick testing |

### Examples

Quick smoke test with a small model:

```bash
bit-axon benchmark --config-small --seq-lengths 32,64,128
```

Full benchmark with more iterations for stable numbers:

```bash
bit-axon benchmark -s 128,256,512,1024,2048,4096 -i 10 -w 3
```

Batch inference profiling:

```bash
bit-axon benchmark --batch-size 4 -s 256,1024
```

## Python API

All four profiler classes live under `bit_axon.profiling`.

### MemoryProfiler

Tracks GPU memory through MLX Metal APIs. Three metrics are available: active memory (currently allocated), peak memory (high-water mark since last reset), and cache memory (reclaimed but not yet freed back to the OS).

```python
from bit_axon.profiling.memory import MemoryProfiler

profiler = MemoryProfiler()

# Model weight footprint
info = profiler.profile_model(model)
print(f"Weights: {info['weight_memory_gb']:.2f} GB, {info['param_count']:,} params")

# Memory during a forward pass at sequence length 2048
profiler.reset_peak()
result = profiler.profile_forward(model, seq_len=2048)
print(f"Peak: {result['peak_memory_gb']:.2f} GB")
print(f"Activations: {result['activation_memory_gb']:.2f} GB")
```

The `profile_forward` method runs a single forward pass with random input and reports peak and active memory. Activation memory is estimated by subtracting the model weight footprint from the peak reading.

Device info is available too:

```python
info = profiler.device_info()
print(info)  # {'architecture': 'apple-m4', ...}
```

### SpeedProfiler

Measures inference throughput in two modes: prefill (batch processing of the full prompt) and autoregressive (one token at a time, simulating decode).

```python
from bit_axon.profiling.speed import SpeedProfiler

profiler = SpeedProfiler()

# Prefill speed: how fast can the model ingest a prompt?
result = profiler.benchmark_tokens_per_sec(
    model,
    seq_len=1024,
    num_warmup=2,
    num_iterations=5,
)
print(f"Prefill: {result['tokens_per_sec']:.0f} tok/s")
print(f"Latency: {result['latency_ms']:.1f} ms (±{result['std_latency_ms']:.1f})")

# Autoregressive speed: how fast can it generate?
result = profiler.benchmark_autoregressive(
    model,
    total_tokens=64,
)
print(f"Decode: {result['tokens_per_sec']:.0f} tok/s")
print(f"Per-token: {result['mean_per_token_ms']:.1f} ms")
```

Prefill benchmarks report mean latency and standard deviation across iterations. Autoregressive benchmarks simulate real generation: the model produces tokens one at a time using KV cache, and the result reflects the sustained decode throughput.

### ThermalMonitor

Reads the Apple Silicon SoC die temperature via `sudo powermetrics`. It supports both one-shot reads and continuous background polling with trend detection.

```python
from bit_axon.profiling.thermal import ThermalMonitor

monitor = ThermalMonitor(poll_interval=1.0, history_size=60)

# One-shot read
temp = monitor.get_soc_temperature()
print(f"SoC: {temp}°C" if temp else "Temperature unavailable")

# Continuous background polling (for training loops)
monitor.start()
# ... run training ...
print(f"Current: {monitor.temperature}°C")
print(f"History: {monitor.get_history()}")
print(f"Rising? {monitor.is_rising(window=5)}")
print(f"Above 95°C? {monitor.is_above(95.0)}")
monitor.stop()
```

!!! warning "Requires sudo"
    `ThermalMonitor` calls `sudo powermetrics` internally. Make sure your terminal session has sudo privileges, or run the script with `sudo`. Without sudo, temperature reads return `None`.

The `is_rising` method uses linear regression over the last N readings to detect upward trends. This is useful for thermal-aware training: if the temperature is climbing toward throttling territory, you can reduce batch size or insert pauses.

### BenchmarkSuite

Combines all three profilers into a single orchestrated run across multiple sequence lengths.

```python
from bit_axon.profiling.benchmark import BenchmarkSuite, BenchmarkResult

suite = BenchmarkSuite()  # uses default BitAxonConfig

results: BenchmarkResult = suite.benchmark_sequence_lengths(
    seq_lengths=[128, 512, 1024, 2048],
    batch_size=1,
    num_warmup=2,
    num_iterations=5,
)

# Print ASCII table
print(results.to_table())

# Access individual results
for name, metrics in results.results.items():
    print(f"{name}: {metrics['tokens_per_sec']} tok/s, "
          f"{metrics['peak_memory_gb']:.2f} GB peak")
```

Each entry in `results.results` contains `seq_len`, `tokens_per_sec`, `latency_ms`, `peak_memory_gb`, `active_memory_gb`, and `soc_temp_c`.

## Interpreting Results

### Speed

On a MacBook Air M4 with 16 GB unified memory, expect roughly:

| Metric | Typical Range | Notes |
|--------|---------------|-------|
| Prefill (128 tokens) | 800 to 2,000 tok/s | Short sequences are memory-bandwidth bound |
| Prefill (1024 tokens) | 300 to 800 tok/s | Longer sequences amortize overhead better |
| Prefill (2048 tokens) | 200 to 600 tok/s | Approaches compute-bound regime |
| Autoregressive decode | 30 to 80 tok/s | Single-token forward passes have higher per-token overhead |

Actual numbers vary by Mac model (M2 vs M4, Air vs Pro), memory pressure from other apps, and whether thermal throttling has kicked in. The Pro and Max chips sustain higher throughput for longer because of their cooling systems.

### Memory

For the full Bit-Axon configuration (3.2B parameters, Q4 quantized):

| Metric | Expected Range |
|--------|---------------|
| Weight memory | ~1.7 to 1.8 GB |
| Total at 4K context | ~2.4 to 2.6 GB |
| Total at 64K context | ~2.8 to 3.2 GB |

If active memory is significantly higher than peak memory, the MLX allocator is holding onto cached buffers. Call `mx.clear_cache()` between runs if you need a clean measurement.

### Temperature

| Range | Meaning |
|-------|---------|
| 40 to 60 °C | Idle to light load. Normal for short benchmarks. |
| 60 to 80 °C | Moderate load. Sustained inference or small-batch training. |
| 80 to 95 °C | Heavy load. Approaching thermal limits on fanless Macs. |
| 95+ °C | Throttling likely. Expect clock speed reductions and lower throughput. |

Fanless machines (MacBook Air) hit thermal limits faster than actively cooled models (MacBook Pro, Mac Studio). If you see temperatures climbing above 90 °C during training, consider reducing batch size or increasing the poll interval to allow cooldown between steps.

## Tips

- **Always warm up.** MLX compiles kernels lazily. The first few forward passes are slow because of Metal shader compilation. The profilers handle this with `num_warmup`, but if you write custom measurement code, account for it.
- **Synchronize before timing.** MLX operations are asynchronous. Always call `mx.synchronize()` before starting and stopping the timer, as the profilers do internally.
- **Close other apps** before benchmarking. GPU memory and thermal headroom are shared across the system. A browser with 50 tabs will skew your numbers.
- **Run multiple iterations.** A single measurement can be noisy due to OS scheduling and thermal state changes. Five or more iterations give a stable mean.
- **Use `--config-small` for iteration speed.** When you are debugging the benchmark pipeline itself rather than measuring real performance, the small config loads near-instantly and keeps iteration times under a second.

---

## See also

- [Inference Guide](inference.md) — Run generation and measure output quality
- [Training Guide](training.md) — Thermal monitoring during training runs
- [Memory Budget](../architecture/memory-budget.md) — Expected memory ranges for inference
- [API — Profiling](../api/profiling.md) — `MemoryProfiler`, `SpeedProfiler`, `ThermalMonitor` Python API
