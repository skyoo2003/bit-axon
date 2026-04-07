# SWA + MoE: 어텐션과 희소 전문가

층 9–16은 국소 추론을 위한 슬라이딩 윈도우 어텐션(SWA)과 희소 연산을 위한 공유 전문가 mixture-of-experts(MoE)를 결합합니다. 층 17–24는 어텐션을 제거하고 Axon-SSM과 동일한 MoE 설계를 결합합니다.

## 슬라이딩 윈도우 어텐션

### 개요

| 속성 | 값 |
|:---------|:------|
| 은닉 차원 | 2,560 |
| 헤드 수 | 32 |
| 헤드 차원 ($d_k$) | 80 ($2560 / 32$) |
| 윈도우 크기 | 4,096 |
| 복잡도 | $O(n^2)$ 대신 $O(n \cdot w)$ |
| 층 | 9–16만 |
| KV cache | 점진적 디코딩을 위한 외부 `KVCache` |

### 알고리즘

```
입력 x: (batch, seq_len, 2560)
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  q_proj  k_proj  v_proj     (각 2560 → 2560)
    │       │       │
    ▼       ▼       ▼
  reshape + transpose  →  (batch, 32, seq_len, 80)
    │       │       │
    │       └───┬───┘
    │           │
    │    ┌──────┴──────┐
    │    │ 캐시 갱신    │  (디코딩 시 K, V 추가)
    │    └──────┬──────┘
    │           │
    ▼           ▼
    Q          K, V
    │           │
    └─────┬─────┘
          ▼
  ┌───────────────┐
  │  scores = QKᵀ │   1/√d_k로 스케일
  │    / √80      │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │ + sliding     │   인과적 AND 윈도우 마스크
  │   window mask │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │   softmax     │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │   × V         │
  └───────┬───────┘
          ▼
  reshape + transpose  →  (batch, seq_len, 2560)
          │
          ▼
      o_proj (2560 → 2560)
          │
          ▼
      출력: (batch, seq_len, 2560)
```

### 어텐션 공식

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

여기서 $M$은 어텐션 점수에 요소별로 적용되는 결합 마스크입니다.

### 슬라이딩 윈도우 마스크

각 위치는 **윈도우 내**의 이전 위치에만 어텐션할 수 있습니다:

$$M_{i,j} = \begin{cases} 0 & \text{if } j \leq i \text{ and } i - j < w \\ -\infty & \text{otherwise} \end{cases}$$

마스크는 두 가지 조건을 결합합니다:

1. **인과적**: $j \leq i$ — 토큰은 미래 위치에 어텐션할 수 없음
2. **윈도우**: $i - j < 4096$ — 토큰은 윈도우 너머에 어텐션할 수 없음

코드:

```python
causal_mask = k_pos[None, :] <= (q_pos[:, None] + causal_offset)
window_mask = (q_pos[:, None] + causal_offset) - k_pos[None, :] < self.window_size
mask = mx.where(causal_mask & window_mask, 0.0, -mx.inf)
```

`causal_offset`은 자기회귀 디코딩 중 `kv_len > seq_len`인 경우 KV cache 길이를 고려합니다.

### 점진적 디코딩을 위한 KV cache

8개 SWA 층(9–16)만 KV cache를 생성합니다. Prefill 시 전체 프롬프트의 K, V 텐서가 저장됩니다. Decode 시 각 새 토큰의 K/V 행이 추가됩니다:

```
Prefill:  K, V 형태 = (batch, 32, prompt_len, 80)
Decode:   K, V 형태 = (batch, 32, prompt_len + n_decoded, 80)
```

캐시는 생성된 토큰에 따라 선형적으로 증가합니다. 긴 컨텍스트의 경우 TurboQuant가 이 KV 텐서를 6배 이상 압축합니다 ([메모리 예산](memory-budget.ko.md) 참조).

## 공유 전문가 MoE

### 개요

| 속성 | 값 |
|:---------|:------|
| 총 전문가 | 8 라우팅 + 1 공유 |
| Top-$k$ | 2 |
| 전문가 FFN 차원 | 4,096 |
| 토큰당 활성 parameters | 약 ~1.4B (전체 3.2B의 약 ~44%) |
| 층 | 9–16 (SWA와 함께) 및 17–24 (SSM과 함께) |

### 아키텍처

```
입력 x: (batch, seq_len, 2560)
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
 ┌─────────┐   ┌─────────────┐
 │  라우터   │   │공유 전문가    │   항상 활성
 │ (gate)   │   │  (MLP+GLU)  │
 └────┬────┘   └──────┬──────┘
      │               │
      ▼               │
 ┌──────────────┐     │
 │ softmax gate  │     │
 │ → top-2 선택  │     │
 └──────┬───────┘     │
        │             │
        ▼             │
  ┌──────────┐        │
  │ 전문가 0  │        │
  │ 전문가 3  │  ← 라우팅으로 선택됨
  │   ...    │        │
  └────┬─────┘        │
       │              ▼
       ▼         sigmoid 게이트 × shared_out
  가중합              │
       │              │
       └──────┬───────┘
              ▼
           y + shared_out
              │
              ▼
      출력: (batch, seq_len, 2560)
```

### 라우팅

게이트는 선형 투영 + softmax를 통해 전문가 점수를 생성합니다:

```python
gates = self.gate(x)                    # (batch, seq_len, 8)
gates = mx.softmax(gates, axis=-1)
inds = mx.argpartition(-gates, kth=k-1)[..., :k]   # top-2 인덱스
scores = mx.take_along_axis(gates, inds, axis=-1)   # top-2 점수
```

top-2 인덱스(`inds`)는 라우팅 기울기 불안정성을 방지하기 위해 `stop_gradient`를 통과합니다. 전문가 출력은 softmax 점수로 가중치가 부여되어 합산됩니다.

### SwitchGLU: 전문가 라우팅 SwiGLU

각 라우팅 전문가는 `SwitchGLU`로 구현됩니다 — MLX의 `gather_mm`를 사용하여 토큰을 전문가별 가중치 행렬로 라우팅하는 SwiGLU MLP입니다:

```
입력 x + 전문가 인덱스
        │
   ┌────┼────┐
   ▼    ▼    ▼
 gate  up   down      (전문가별 SwitchLinear)
  │    │    │
  ▼    ▼    │
SwiGLU: SiLU(gate) × up
        │
        ▼
     down_proj
        │
        ▼
    전문가 출력
```

**Gather-sort 최적화**: 토큰 수가 64를 초과하면 `gather_mm` 전에 토큰을 전문가 인덱스로 정렬합니다. 이렇게 하면 동일 전문가의 토큰이 연속적으로 배치되어 메모리 접근 패턴이 개선됩니다:

```python
if indices.size >= 64:
    x_sorted, idx_flat, inv_order = _gather_sort(x_exp, indices)
    # ... 정렬된 데이터 처리 ...
    y = _scatter_unsort(y_flat, inv_order, shape=(B, L, K))
```

### 공유 전문가

공유 전문가는 **모든** 토큰에 무조건 적용되는 표준 SwiGLU MLP(`gate_proj`, `up_proj`, `down_proj`)입니다. 그 출력은 학습된 sigmoid로 게이팅됩니다:

$$y_\text{shared} = \sigma(W_\text{gate} \cdot x) \cdot \text{MLP}(x)$$

이를 통해 모델은 공유 지식(항상 사용 가능)과 라우팅된 전문가 지식(특화됨)을 동적으로 혼합할 수 있습니다:

```python
shared_out = self.shared_expert(x)
shared_out = mx.sigmoid(self.shared_expert_gate(x)) * shared_out
```

### 왜 공유 전문가가 필요한가?

| 공유 전문가 없음 | 공유 전문가 있음 |
|:---------------------|:-------------------|
| 모든 지식을 라우팅해야 함 | 일반 지식이 항상 사용 가능 |
| 라우터 오류 시 정보 손실 | 공유 전문가가 안전망 역할 |
| 부하 균형이 중요함 | 라우팅 품질에 덜 민감 |
| 각 전문가가 공통 패턴을 재학습해야 함 | 공유 전문가가 공통 패턴 처리, 전문가는 특화 |

## 블록 구성

SWA+MoE 블록과 SSM+MoE 블록은 동일한 MoE 설계를 공유하지만 첫 번째 하위 층이 다릅니다:

### AxonSWAMoEBlock (층 9–16)

```
x → RMSNorm → SWA → (+잔차) → RMSNorm → MoE → (+잔차) → 출력
                              ↑                    ↑
                          KV cache            캐시 없음
```

### AxonSSMMoEBlock (층 17–24)

```
x → RMSNorm → SSM → (+잔차) → RMSNorm → MoE → (+잔차) → 출력
                              ↑                    ↑
                         SSM 상태             캐시 없음
```

두 블록 모두 각 하위 층에 대해 별도의 `RMSNorm` 인스턴스를 사용하는 pre-norm 잔차 연결을 사용합니다.

## 희소성과 열 이점

8개 전문가 중 top-2를 사용하므로 토큰당 라우팅된 전문가 parameters의 25%만 활성화됩니다. 공유 전문가와 결합하면:

- **토큰당 활성 parameters**: 약 ~1.4B (전체 3.2B의 44%)
- **휴면 parameters**: 약 ~1.8B (56%) — 연산 없음, 메모리 대역폭 소비 없음
- **열 영향**: 칩 활용도가 낮아 팬 없는 MacBook Air에서 지속적인 추론/학습 가능

`swiglu` 활성화 함수 역시 `@mx.compile`로 JIT 컴파일됩니다:

```python
@mx.compile
def swiglu(x: mx.array, gate: mx.array) -> mx.array:
    return nn.silu(gate) * x
```

[← 아키텍처로 돌아가기](index.ko.md)
