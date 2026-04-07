# Thermal-Aware Training on Fanless Hardware

**Status**: :fontawesome-solid-circle-check:{ .green } Implemented
**Source**: [`src/bit_axon/profiling/thermal.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/profiling/thermal.py), [`src/bit_axon/training/cooling.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/training/cooling.py)

## Abstract

Training language models on fanless Apple Silicon MacBooks introduces a unique constraint: sustained GPU compute causes thermal throttling that can degrade throughput by 40–60% or trigger hardware shutdown. Bit-Axon addresses this with a thermal-aware training system consisting of a background thermal monitor (`ThermalMonitor`) and a cooling-gated training scheduler (`CoolingScheduler`). The monitor polls SoC die temperature via macOS `powermetrics`, and the scheduler pauses training steps when the temperature exceeds configurable thresholds—enabling sustained multi-hour fine-tuning on a MacBook Air M4 without thermal throttling.

## Key Contributions

1. **Background thermal polling** — A daemon thread continuously reads SoC temperature with configurable intervals, maintaining a rolling history buffer.
2. **Three-tier thermal policy** — Separate thresholds for batch-size reduction, training pause, and emergency shutdown.
3. **Trend detection** — Linear regression on the temperature history window determines whether the SoC is heating up, enabling proactive throttling.
4. **Zero-overhead on non-macOS** — Gracefully degrades to no-op when `powermetrics` is unavailable.

## Mathematical Foundations

### Temperature Sampling Model

Let $T_k$ denote the SoC die temperature at sample $k$. The monitor maintains a rolling window:

$$
\mathcal{H} = [T_{k - n + 1}, T_{k - n + 2}, \ldots, T_k]
$$

where $n = |\mathcal{H}|$ is the history buffer size (default: 60 samples).

### Trend Detection via Linear Regression

To determine whether the SoC is heating up, the monitor fits a simple linear model over the last $w$ readings:

$$
T_i = \alpha + \beta \cdot i + \varepsilon_i, \quad i \in \{k - w + 1, \ldots, k\}
$$

The slope estimate is:

$$
\hat{\beta} = \frac{\sum_{i=0}^{w-1}(i - \bar{i})(T_{k-w+1+i} - \bar{T})}{\sum_{i=0}^{w-1}(i - \bar{i})^2}
$$

where $\bar{i} = (w-1)/2$ and $\bar{T}$ is the mean of the window. The SoC is considered to be in a **rising** trend when $\hat{\beta} > 0$.

### Three-Tier Thermal Policy

The `ThermalPolicy` defines three temperature thresholds:

$$
T_{\text{speed}} < T_{\text{pause}} < T_{\text{stop}}
$$

| Threshold | Default | Action |
|-----------|---------|--------|
| $T_{\text{speed}} = 75°C$ | `max_speed_temp` | Signal to reduce batch size |
| $T_{\text{pause}} = 85°C$ | `pause_temp` | Pause training for $\delta t$ seconds |
| $T_{\text{stop}} = 95°C$ | `stop_temp` | Raise `ThermalShutdownError` |

### Pause Loop

When $T \geq T_{\text{pause}}$, the scheduler enters a sleep loop:

$$
\text{while } T \geq T_{\text{pause}}: \quad \text{sleep}(\delta t), \quad t_{\text{total}} \mathrel{+}= \delta t
$$

where $\delta t = 0.5$ seconds (configurable). The loop exits when the temperature drops below $T_{\text{pause}}$ or the reading becomes unavailable.

### Batch Size Reduction Signal

When $T_{\text{speed}} \leq T < T_{\text{pause}}$, the scheduler signals that the batch size should be reduced:

$$
\text{reduce\_batch} = \begin{cases}
\text{True} & \text{if } T_{\text{speed}} \leq T < T_{\text{pause}} \\
\text{False} & \text{otherwise}
\end{cases}
$$

## Implementation in Bit-Axon

### ThermalMonitor

| Component | Description |
|-----------|-------------|
| `get_soc_temperature()` | Reads die temperature via `sudo powermetrics --samplers smc -i 1 -n 1` |
| `start()` / `stop()` | Daemon thread lifecycle for background polling |
| `temperature` | Thread-safe property returning the latest reading |
| `get_history()` | Returns a snapshot of the temperature buffer |
| `is_rising(window=5)` | Linear regression slope test over last $w$ readings |
| `is_above(threshold)` | Simple threshold comparison |

### CoolingScheduler

| Component | Description |
|-----------|-------------|
| `check_before_step(step)` | Pre-step temperature gate; pauses or raises shutdown error |
| `should_reduce_batch()` | Signal for adaptive batch sizing |
| `total_pause_time` | Cumulative pause time for training logs |

### ThermalPolicy

```python
@dataclass
class ThermalPolicy:
    max_speed_temp: float = 75.0   # Signal batch reduction
    pause_temp: float = 85.0       # Pause training
    stop_temp: float = 95.0        # Emergency shutdown
    pause_duration: float = 0.5    # Seconds per pause iteration
```

Invariant enforced at initialization: $T_{\text{speed}} < T_{\text{pause}} < T_{\text{stop}}$.

### Training Loop Integration

```python
from bit_axon.profiling.thermal import ThermalMonitor
from bit_axon.training.cooling import CoolingScheduler, ThermalPolicy

monitor = ThermalMonitor(poll_interval=1.0, history_size=60)
monitor.start()

scheduler = CoolingScheduler(monitor, ThermalPolicy())

for step in range(num_steps):
    scheduler.check_before_step(step)  # Pauses or raises if too hot
    # ... training step ...
```

## References

- Apple Developer Documentation. *powermetrics — System Power Metrics*.
- Apple Machine Learning Research. *MLX: An array framework for machine learning on Apple silicon*.
- You, Y., et al. (2017). *Large Batch Training of Convolutional Networks*. arXiv:1708.03888. (Related: learning rate scaling under hardware constraints.)
