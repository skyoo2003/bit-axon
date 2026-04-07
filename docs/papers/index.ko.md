# 연구 논문

이 섹션은 Bit-Axon의 이론적 기반과 핵심 혁신 사항을 정리합니다. 각 페이지는 시스템의 핵심 구성 요소에 대한 수학적 정식화, 설계 근거, 구현 매핑을 제공합니다.

## 논문 목록

| # | 논문 | 상태 | 핵심 아이디어 |
|:--|:-----|:-----|:------------|
| 1 | [Axon-SSM: Selective SSM for Apple Silicon](axon-ssm.ko.md) | ![Implemented](https://img.shields.io/badge/status-Implemented-success) | `@mx.compile` 퓨즈드 커널을 적용한 Mamba 스타일 선택적 상태 공간 모델, 토큰당 $\mathcal{O}(1)$ 메모리 |
| 2 | [24-Layer Sandwich Architecture](sandwich-architecture.ko.md) | ![Implemented](https://img.shields.io/badge/status-Implemented-success) | 3영역 하이브리드: SSM → SWA+MoE → SSM+MoE, 차원 브리지 $d_{\text{src}}=2048$ |
| 3 | [Thermal-Aware Training](thermal-training.ko.md) | ![Implemented](https://img.shields.io/badge/status-Implemented-success) | `CoolingScheduler` + macOS `powermetrics`, 85°C에서 일시 정지, 95°C에서 중단 |
| 4 | [TurboQuant KV Cache Compression](turboquant.ko.md) | ![Planned](https://img.shields.io/badge/status-Planned-yellow) | 64K 컨텍스트를 위한 KV 캐시 압축; ICLR 2026 참조 |

## 범위

이 논문들은 각 구성 요소의 **수학적 기반**과 **알고리즘 설계**에 초점을 맞춥니다. API 사용법 및 통합 세부 사항은 [아키텍처](../architecture/index.md) 섹션을 참조하세요.

## 참고 문헌

- **Mamba**: Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- **MLX**: Apple Machine Learning Research. *MLX: An array framework for machine learning on Apple silicon*.
- **Qwen2.5**: Qwen Team (2024). *Qwen2.5 Technical Report*.
