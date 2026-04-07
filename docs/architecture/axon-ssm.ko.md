# Axon-SSM: 선택적 상태 공간 모델

Axon-SSM은 Bit-Axon의 Mamba 스타일 선택적 상태 공간 모델입니다. 표준 Transformer 자기 어텐션을 선형 순회로 대체하여, 토큰당 $O(1)$ 메모리를 달성하고 KV cache를 완전히 제거합니다.

## 개요

| 속성 | 값 |
|:---------|:------|
| 상태 차원 ($d_\text{state}$) | 16 |
| 합성곱 커널 ($d_\text{conv}$) | 4 |
| 확장 비율 ($\text{ssm\_expand}$) | 3 |
| 중간 차원 ($E$) | $2560 \times 3 = 7680$ |
| 토큰당 메모리 | $O(1)$ — 고정 상태 벡터 |
| KV cache | **없음** |

Axon-SSM은 24개 층 중 16개에 등장합니다:

- **층 1–8**: 순수 SSM (`AxonSSMBlock`) — SSM의 내부 확장이 FFN/MLP 역할을 완전히 대체
- **층 17–24**: SSM + MoE (`AxonSSMMoEBlock`) — SSM이 순회를 담당하고 MoE가 희소 연산을 추가

## 알고리즘

Axon-SSM의 순전파는 다음 단계를 따릅니다:

```
입력 x: (batch, seq_len, hidden_dim=2560)
            │
            ▼
    ┌─── in_proj ───┐
    │  (D → 2E)     │
    └───────┬───────┘
            │ split
      ┌─────┴─────┐
      ▼           ▼
  x_branch    z_branch     (각 차원 E=7680)
      │           │
      ▼           │
 ┌──────────┐     │
 │ conv1d   │     │   인과적, kernel=4, groups=E
 │ (depthwise)│   │
 └────┬─────┘     │
      ▼           │
  ┌───────┐       │
  │ SiLU  │       │
  └───┬───┘       │
      │           │
      ▼           │
 ┌─── x_proj ───┐ │
 │ (E → 2·d_state+1) │
 └──────┬───────┘ │
   ┌────┼────┐    │
   ▼    ▼    ▼    │
   B    C   dt_raw   (B, C: 각각 d_state; dt_raw: 1)
   │    │    │      │
   │    │    ▼      │
   │    │ ┌────────┐│
   │    │ │dt_proj ││  (1 → E)
   │    │ │+softplus││
   │    │ │+clip   ││
   │    │ └───┬────┘│
   │    │     ▼     │
   │    │    dt     │  (채널별 스텝 크기, 차원 E)
   │    │     │     │
   ▼    ▼     ▼     │
 ┌──────────────┐   │
 │  SSM 스캔    │   │  seq_len에 대한 순차 순회
 │  (아래 참조) │   │
 └──────┬───────┘   │
        ▼           │
        y           │
        │           │
        ▼           ▼
    ┌─────────────────┐
    │  y * SiLU(z)    │   게이팅: SSM 출력에 활성화된 z 브랜치를 곱함
    └────────┬────────┘
             ▼
    ┌─── out_proj ───┐
    │  (E → D=2560)  │
    └────────┬───────┘
             ▼
      출력: (batch, seq_len, 2560)
```

### 핵심 투영

| 층 | 형태 | 목적 |
|:------|:------|:--------|
| `in_proj` | $(D, 2E)$ | $x$와 $z$ 브랜치로 분할 (게이팅) |
| `conv1d` | Depthwise, kernel=4 | SSM 이전의 국소 인과적 컨텍스트 |
| `x_proj` | $(E, 2 \cdot d_\text{state} + 1)$ | $B$, $C$, 원시 $\Delta t$ 생성 |
| `dt_proj` | $(1, E)$ | 바이어스가 있는 채널별 스텝 크기 |
| `out_proj` | $(E, D)$ | 은닉 차원으로 역투영 |

## SSM 순회

핵심 스캔은 각 타임스텝 $t$에서 이산화된 선형 순회를 계산합니다:

$$h_t = \exp(\Delta t_t \cdot A) \cdot h_{t-1} + \Delta t_t \cdot B_t \cdot x_t$$

$$y_t = C_t \cdot h_t + D \cdot x_t$$

여기서:

| 기호 | 형태 | 설명 |
|:-------|:------|:------------|
| $h_t$ | $(B_\text{batch}, E, d_\text{state})$ | 시간 $t$에서의 은닉 상태 |
| $A$ | $(E, d_\text{state})$ | 대각 상태 행렬 (학습 가능, $\log$로 저장) |
| $B_t$ | $(B_\text{batch}, d_\text{state})$ | 시간 $t$에서의 입력 선택적 행렬 |
| $C_t$ | $(B_\text{batch}, d_\text{state})$ | 시간 $t$에서의 출력 선택적 행렬 |
| $\Delta t_t$ | $(B_\text{batch}, E)$ | 시간 $t$에서의 채널별 스텝 크기 |
| $D$ | $(E,)$ | 스킵 연결 (초기값: 1) |

### 이산화

스텝 크기 $\Delta t$는 softplus 투영과 클램핑을 통해 계산됩니다:

$$\Delta t = \text{clip}(\text{softplus}(\text{dt\_proj}(dt_\text{raw}) + \text{dt\_bias}),\ 10^{-4},\ 100)$$

이를 통해 $\Delta t$가 수치적으로 안정적인 범위를 유지하면서 입력 의존적(선택적)으로 유지됩니다.

### 상태 초기화

$A$ 행렬은 반복된 대각행렬로 초기화됩니다:

$$A_{\log} = \log\!\Big(\text{repeat}\big(\text{arange}(1,\ d_\text{state}+1)\big)_{\text{reps}=E}\Big)$$

런타임에: $A = -\exp(A_{\log})$, $-1$에서 $-d_\text{state}$까지의 대각선을 생성합니다. 음의 지수는 시간에 따른 은닉 상태의 안정적인 감쇠를 보장합니다.

## 메모리 특성

### 토큰당 일정한 메모리

KV cache가 시퀀스 길이에 따라 $O(n)$으로 증가하는 표준 어텐션과 달리, SSM은 고정 크기 상태를 유지합니다:

| 구성 요소 | 크기 |
|:----------|:-----|
| SSM 상태 | $(B_\text{batch},\ E=7680,\ d_\text{state}=16)$ |
| Conv cache | $(B_\text{batch},\ K{-}1=3,\ E=7680)$ |
| **층당 합계** | **약 ~1.5 MB** (FP16, batch=1) |
| **16개 SSM 층 합계** | **약 ~24 MB** |

16개 어텐션 층에 대해 64K 컨텍스트로 전체 KV cache를 사용할 경우 수 GB가 필요한 것과 비교해 보세요.

### KV cache 없음

SSM 층은 캐시로 `[conv_cache, ssm_state]`를 반환합니다 — 작고 고정 크기의 텐서입니다. 모델의 `_create_caches()` 메서드는 모든 SSM 층에 대해 `None`을 반환하고, 8개 SWA 층(9–16)에 대해서만 `KVCache` 객체를 반환합니다.

## JIT 컴파일된 커널

두 리프 함수는 퓨즈된 Metal 커널 생성을 위해 `@mx.compile`로 데코레이션됩니다 (Jamba 패턴을 따름):

### `_ssm_fma`

```python
@mx.compile
def _ssm_fma(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    return a * b + c    # dA * h + dB * x_t  (퓨즈된 곱셈-덧셈)
```

이는 상태 갱신 $h_t = dA \cdot h_{t-1} + dB \cdot x_t$를 단일 커널로 퓨즈하여 중간 텐서 할당을 피합니다.

### `_compute_dt`

```python
@mx.compile
def _compute_dt(dt: mx.array, dt_bias: mx.array, lo: float, hi: float) -> mx.array:
    return mx.clip(nn.softplus(dt + dt_bias), lo, hi)
```

바이어스 덧셈, softplus 활성화, 클램핑을 하나의 커널로 퓨즈합니다.

## 자기회귀 디코딩

점진적(토큰별) 생성 중 캐시 메커니즘은 다음과 같이 작동합니다:

1. **첫 호출** (prefill, `cache=None`): 전체 프롬프트를 처리하고, `ssm_state`를 0으로 초기화하며, 마지막 $K-1$ 위치에서 `conv_cache`를 구축합니다.
2. **이후 호출** (decode, `cache=[conv_cache, ssm_state]`): `conv_cache`와 새로운 단일 토큰을 연결하고, 이전 `ssm_state`를 사용하여 스캔의 한 스텝을 실행한 뒤 갱신된 캐시를 반환합니다.

스캔 루프는 `seq_len` 위치만큼 반복됩니다 — prefill 중에는 전체 프롬프트 길이이고, decode 중에는 정확히 1입니다.

## Parameters

SSM 층당 parameter 수 ($D=2560$, $E=7680$, $d_\text{state}=16$ 기준):

| Parameter | 형태 | 수 |
|:----------|:------|:------|
| `in_proj.weight` | $(2E, D)$ | 39.3M |
| `conv1d.weight` | $(E, 1, 4)$ | 30.7K |
| `conv1d.bias` | $(E,)$ | 7.7K |
| `x_proj.weight` | $(33, E)$ | 253.4K |
| `dt_proj.weight` | $(E, 1)$ | 7.7K |
| `dt_proj.bias` | $(E,)$ | 7.7K |
| `A_log` | $(E, 16)$ | 122.9K |
| `D` | $(E,)$ | 7.7K |
| `out_proj.weight` | $(D, E)$ | 19.7M |
| **SSM 층당 합계** | | **약 ~60.2M** |

SSM을 포함하는 16개 층(8 순수 + 8 MoE 포함)으로, SSM은 전체 3.2B 중 약 **960M parameters**를 차지합니다.

[← 아키텍처로 돌아가기](index.ko.md)
