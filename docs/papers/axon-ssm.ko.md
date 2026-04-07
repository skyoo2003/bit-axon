# Axon-SSM: Selective State Space Model for Apple Silicon

**상태**: :fontawesome-solid-circle-check:{ .green } 구현됨
**소스**: [`src/bit_axon/layers/axon_ssm.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/layers/axon_ssm.py)

## 요약

Axon-SSM은 MLX를 통해 Apple Silicon에 맞게 설계 및 컴파일된 Mamba 스타일 선택적 상태 공간 모델 레이어입니다. 기존의 셀프 어텐션을 선형 순환 기반 시퀀스 모델링으로 대체하며, 자기회귀 디코딩 시 토큰당 $\mathcal{O}(1)$ 메모리를 달성합니다. KV 캐시가 필요하지 않습니다. 이 레이어는 인과적 깊이별(depthwise) 합성곱, 입력 종속 매개변수 선택, SiLU 게이팅, 그리고 `@mx.compile`을 통한 하드웨어 인식 컴파일을 통합합니다.

## 핵심 기여

1. **하드웨어 인식 컴파일** — 핵심 SSM 커널(`_ssm_fma`, `_compute_dt`)은 Apple GPU에서 MLX 그래프 최적화를 위해 `@mx.compile`로 데코레이션됩니다.
2. **선택적 스캔 메커니즘** — 입력 종속 $\Delta t$, $B$, $C$ 행렬을 통해 모델이 각 타임스텝에서 보유하거나 망각할 정보의 양을 동적으로 제어합니다.
3. **인과적 합성곱 접두부** — 커널 크기 4의 깊이별 1D 합성곱이 순환 스캔 이전에 로컬 컨텍스트를 제공합니다.
4. **이중 분기 게이팅** — SiLU 게이팅된 출력 분기가 SSM 출력에 곱해지며, 입력 투영이 $x$와 $z$ 분기로 분할되는 Mamba 설계를 따릅니다.

## 수학적 기반

### 연속 시간 SSM

구조화 상태 공간 모델은 1차원 입력 $x(t) \in \mathbb{R}$을 잠재 상태 $h(t) \in \mathbb{R}^N$를 거쳐 출력 $y(t) \in \mathbb{R}$로 매핑합니다:

$$
h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)
$$

여기서 $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$, $D \in \mathbb{R}^{1 \times 1}$입니다.

### 영차 유지(ZOH) 이산화

타임스텝 $\Delta t$가 주어지면, 연속 시스템은 영차 유지(ZOH, Zero-Order Hold)를 사용하여 이산화됩니다:

$$
\bar{A} = \exp(\Delta t \cdot A), \quad \bar{B} = (\Delta t \cdot A)^{-1}(\exp(\Delta t \cdot A) - I) \cdot B
$$

Axon-SSM에서는 간소화된 1차 근사를 사용합니다:

$$
\bar{A} = \exp(\Delta t \cdot A), \quad \bar{B} = \Delta t \cdot B
$$

각 스텝의 순환 갱신은 다음과 같습니다:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t + D x_t
$$

### 선택적 메커니즘

Mamba의 핵심 혁신은 $B$, $C$, $\Delta t$를 고정값이 아닌 **입력 종속**으로 만드는 것입니다:

$$
(B_t, C_t, \Delta t_t) = f_{\text{proj}}(x_t)
$$

여기서 $f_{\text{proj}}$는 합성곱 출력에서 SSM 매개변수로의 선형 투영입니다. 스텝 크기 $\Delta t$는 추가로 처리됩니다:

$$
\Delta t_t = \text{softplus}(\Delta t_{\text{raw}} + b_{\Delta t}), \quad \text{clipped to } [\epsilon, \Delta t_{\max}]
$$

현재 구현에서 $\epsilon = 10^{-4}$, $\Delta t_{\max} = 100.0$입니다.

### 대각 상태 행렬

$A$ 행렬은 대각행렬로 제약되며, 다음과 같이 초기화됩니다:

$$
A_{i,j} = \begin{cases} -\exp(\text{A\_log}_{i}) & \text{if } i = j \\ 0 & \text{otherwise} \end{cases}
$$

여기서 $\text{A\_log}$는 $\log(\text{arange}(1, N+1))$로 초기화되어, $A$의 대각 성분이 $-1, -2, \ldots, -N$이 됩니다. 이 초기화는 느린($-1$) 것부터 빠른($-N$) 것까지 다양한 감쇠율을 제공합니다.

### 게이팅

이 레이어는 SiLU 게이팅된 이중 분기 구조를 사용합니다. 입력 투영은 두 분기를 생성합니다:

$$
(x_{\text{branch}}, z_{\text{branch}}) = \text{split}(W_{\text{in}} \cdot x)
$$

최종 출력은 다음과 같습니다:

$$
y_{\text{out}} = W_{\text{out}} \cdot (y_{\text{ssm}} \odot \text{SiLU}(z_{\text{branch}}))
$$

## Bit-Axon에서의 구현

### 레이어 설정

| 매개변수 | 기호 | 값 |
|----------|------|-----|
| 은닉 차원 | $D$ | 2,560 |
| SSM 확장 비율 | — | 3 |
| SSM 중간 차원 | $E = D \times 3$ | 7,680 |
| 상태 차원 | $N$ | 16 |
| 합성곱 커널 | $K$ | 4 |

### 코드 매핑

| 구성 요소 | 소스 위치 |
|-----------|-----------|
| SSM FMA 커널 | `_ssm_fma()` — `@mx.compile`로 컴파일 |
| 스텝 크기 계산 | `_compute_dt()` — `@mx.compile`로 컴파일 |
| 인과적 conv1d | 캐시 지원을 갖춘 `_causal_conv1d()` |
| 순환 스캔 | `_ssm_scan()` — 타임스텝에 대한 순차 루프 |
| 전체 순전파 | `__call__()` — 투영, 합성곱, 스캔, 게이팅을 총괄 |

### 자기회귀 디코딩

이 레이어는 캐시된 추론을 지원합니다. 캐시 튜플 `[conv_cache, ssm_state]`는 타임스텝 사이에 합성곱 패딩과 SSM 은닉 상태를 전달합니다:

- **conv_cache**: 형태 $(B, K-1, E)$ — 인과적 합성곱을 위해 마지막 $K-1$ 위치를 저장합니다.
- **ssm_state**: 형태 $(B, E, N)$ — 순환 은닉 상태 $h$입니다.

## 참고 문헌

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. ICLR 2022.
- Apple MLX Documentation. *MLX: Compile and Graph Optimization*.
