# Research Papers

This section documents the theoretical foundations and key innovations behind Bit-Axon. Each page provides the mathematical formulation, design rationale, and implementation mapping for a core component of the system.

## Papers

| # | Paper | Status | Key Idea |
|:--|:------|:-------|:---------|
| 1 | [Axon-SSM: Selective SSM for Apple Silicon](axon-ssm.md) | ![Implemented](https://img.shields.io/badge/status-Implemented-success) | Mamba-style selective state space model with `@mx.compile` fused kernels, $\mathcal{O}(1)$ memory per token |
| 2 | [24-Layer Sandwich Architecture](sandwich-architecture.md) | ![Implemented](https://img.shields.io/badge/status-Implemented-success) | Three-zone hybrid: SSM → SWA+MoE → SSM+MoE, dimension bridge $d_{\text{src}}=2048$ |
| 3 | [Thermal-Aware Training](thermal-training.md) | ![Implemented](https://img.shields.io/badge/status-Implemented-success) | `CoolingScheduler` + macOS `powermetrics`, pause at 85°C, halt at 95°C |
| 4 | [TurboQuant KV Cache Compression](turboquant.md) | ![Planned](https://img.shields.io/badge/status-Planned-yellow) | Compress KV cache for 64K contexts; ICLR 2026 reference |

## Scope

These papers focus on the **mathematical foundations** and **algorithmic design** of each component. For API usage and integration details, see the [Architecture](../architecture/index.md) section.

## Referenced Work

- **Mamba**: Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- **MLX**: Apple Machine Learning Research. *MLX: An array framework for machine learning on Apple silicon*.
- **Qwen2.5**: Qwen Team (2024). *Qwen2.5 Technical Report*.
