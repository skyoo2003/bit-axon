# Axon-SSM: Selective State Space Model for Apple Silicon

**Status**: :fontawesome-solid-circle-check:{ .green } Implemented
**Source**: [`src/bit_axon/layers/axon_ssm.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/layers/axon_ssm.py)

## Abstract

Axon-SSM is a Mamba-style selective state space model layer designed and compiled for Apple Silicon via MLX. It replaces traditional self-attention with linear-recurrence-based sequence modeling, achieving $\mathcal{O}(1)$ memory per token during autoregressive decoding—no KV cache required. The layer integrates causal depthwise convolution, input-dependent parameter selection, SiLU gating, and hardware-aware compilation through `@mx.compile`.

## Key Contributions

1. **Hardware-aware compilation** — Core SSM kernels (`_ssm_fma`, `_compute_dt`) are decorated with `@mx.compile` for MLX graph optimization on the Apple GPU.
2. **Selective scan mechanism** — Input-dependent $\Delta t$, $B$, and $C$ matrices allow the model to dynamically control how much information to retain or forget at each timestep.
3. **Causal convolution prefix** — A depthwise 1D convolution with kernel size 4 provides local context before the recurrent scan.
4. **Dual-branch gating** — SiLU-gated output branch multiplies the SSM output, following the Mamba design where the input projection splits into $x$ and $z$ branches.

## Mathematical Foundations

### Continuous-Time SSM

A structured state space model maps a 1D input $x(t) \in \mathbb{R}$ to an output $y(t) \in \mathbb{R}$ through a latent state $h(t) \in \mathbb{R}^N$:

$$
h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)
$$

where $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$, and $D \in \mathbb{R}^{1 \times 1}$.

### Discretization via Zero-Order Hold

Given a timestep $\Delta t$, the continuous system is discretized using zero-order hold (ZOH):

$$
\bar{A} = \exp(\Delta t \cdot A), \quad \bar{B} = (\Delta t \cdot A)^{-1}(\exp(\Delta t \cdot A) - I) \cdot B
$$

In Axon-SSM, the implementation uses the simplified first-order approximation:

$$
\bar{A} = \exp(\Delta t \cdot A), \quad \bar{B} = \Delta t \cdot B
$$

The recurrent update at each step becomes:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t + D x_t
$$

### Selective Mechanism

The key innovation from Mamba is making $B$, $C$, and $\Delta t$ **input-dependent** rather than fixed:

$$
(B_t, C_t, \Delta t_t) = f_{\text{proj}}(x_t)
$$

where $f_{\text{proj}}$ is a linear projection from the convolution output to the SSM parameters. The step size $\Delta t$ is further processed:

$$
\Delta t_t = \text{softplus}(\Delta t_{\text{raw}} + b_{\Delta t}), \quad \text{clipped to } [\epsilon, \Delta t_{\max}]
$$

with $\epsilon = 10^{-4}$ and $\Delta t_{\max} = 100.0$ in the current implementation.

### Diagonal State Matrix

The $A$ matrix is constrained to be diagonal, initialized as:

$$
A_{i,j} = \begin{cases} -\exp(\text{A\_log}_{i}) & \text{if } i = j \\ 0 & \text{otherwise} \end{cases}
$$

where $\text{A\_log}$ is initialized to $\log(\text{arange}(1, N+1))$, giving $A$ diagonal entries $-1, -2, \ldots, -N$. This initialization provides a range of decay rates from slow ($-1$) to fast ($-N$).

### Gating

The layer uses a SiLU-gated dual-branch structure. The input projection produces two branches:

$$
(x_{\text{branch}}, z_{\text{branch}}) = \text{split}(W_{\text{in}} \cdot x)
$$

The final output is:

$$
y_{\text{out}} = W_{\text{out}} \cdot (y_{\text{ssm}} \odot \text{SiLU}(z_{\text{branch}}))
$$

## Implementation in Bit-Axon

### Layer Configuration

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Hidden dimension | $D$ | 2,560 |
| SSM expansion ratio | — | 3 |
| SSM intermediate dimension | $E = D \times 3$ | 7,680 |
| State dimension | $N$ | 16 |
| Convolution kernel | $K$ | 4 |

### Code Mapping

| Component | Source Location |
|-----------|----------------|
| SSM FMA kernel | `_ssm_fma()` — compiled with `@mx.compile` |
| Step size computation | `_compute_dt()` — compiled with `@mx.compile` |
| Causal conv1d | `_causal_conv1d()` with cache support |
| Recurrent scan | `_ssm_scan()` — sequential loop over timesteps |
| Full forward pass | `__call__()` — orchestrates projection, conv, scan, gating |

### Autoregressive Decoding

The layer supports cached inference. The cache tuple `[conv_cache, ssm_state]` carries forward the convolution padding and SSM hidden state between timesteps:

- **conv_cache**: Shape $(B, K-1, E)$ — stores the last $K-1$ positions for causal convolution.
- **ssm_state**: Shape $(B, E, N)$ — the recurrent hidden state $h$.

## References

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. ICLR 2022.
- Apple MLX Documentation. *MLX: Compile and Graph Optimization*.

---

## See also

- [Architecture — Axon-SSM](../architecture/axon-ssm.md) — Implementation details, memory properties, and JIT kernels
- [SWA + MoE](../architecture/swa-moe.md) — Attention and sparse experts paired with SSM in the sandwich design
- [Sandwich Architecture Paper](sandwich-architecture.md) — How Axon-SSM fits into the three-zone layout
- [API — Layers](../api/layers.md) — `AxonSSM` Python class documentation
