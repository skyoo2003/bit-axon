# 24-Layer Sandwich Architecture

**상태**: :fontawesome-solid-circle-check:{ .green } 구현됨
**소스**: [`src/bit_axon/model.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/model.py), [`src/bit_axon/layers/block.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/layers/block.py)

## 요약

Bit-Axon은 24층 샌드위치 아키텍처를 채택하며, 네트워크가 처리 파이프라인에서 각각 고유한 역할을 수행하는 세 개의 기능적 영역으로 나뉩니다. 첫 번째 영역은 $\mathcal{O}(1)$ 메모리 컨텍스트 흡수를 위해 순수 SSM 레이어를 사용합니다. 중간 영역은 4K 토큰 윈도우에 걸쳐 집중적인 추론을 위해 슬라이딩 윈도우 어텐션과 Mixture-of-Experts를 결합합니다. 마지막 영역은 어텐션을 완전히 배제하고, 빠른 출력 합성을 위해 SSM과 MoE를 사용합니다. 차원 브리지는 Qwen2.5 가중치 호환성을 위해 소스 모델 차원(2,048)과 내부 은닉 차원(2,560) 사이를 투영합니다.

## 핵심 기여

1. **기능적 레이어 구역화** — 서로 다른 계산 기본 요소가 정보 처리 특성에 따라 네트워크 깊이별로 배정됩니다.
2. **어텐션 없는 출력 영역** — 마지막 8개 층은 어텐션 메커니즘을 사용하지 않아, 출력 생성 시 토큰당 $\mathcal{O}(1)$ 메모리를 가능하게 합니다.
3. **차원 브리지** — $d_{\text{source}} = 2048$과 $d_{\text{model}} = 2560$ 사이의 입력 및 출력 투영이 Qwen2.5-3B로부터의 가중치 이식을 허용합니다.
4. **캐시 이질성** — 중간 8개 층만 KV 캐시를 유지하며, SSM 레이어는 내부 순환 상태를 사용하여 긴 컨텍스트 추론 시 메모리를 크게 절감합니다.

## 수학적 기반

### 레이어 배정 함수

인덱스 $i \in \{0, 1, \ldots, 23\}$에 대한 레이어 유형은 다음과 같이 결정됩니다:

$$
\text{type}(i) = \begin{cases}
\text{SSM} & \text{if } i < \lfloor L/3 \rfloor \\
\text{SWA+MoE} & \text{if } \lfloor L/3 \rfloor \leq i < \lfloor 2L/3 \rfloor \\
\text{SSM+MoE} & \text{otherwise}
\end{cases}
$$

여기서 $L = 24$는 총 레이어 수입니다.

### 전체 순전파

입력 토큰 인덱스 $\mathbf{t} \in \mathbb{Z}^{B \times S}$가 주어지면:

$$
\mathbf{x}_0 = W_{\text{embed}}[\mathbf{t}] \in \mathbb{R}^{B \times S \times d_{\text{source}}}
$$

$$
\mathbf{x}_0' = \mathbf{x}_0 W_{\text{in}} \in \mathbb{R}^{B \times S \times d_{\text{model}}}
$$

각 레이어 $i$에 대해:

$$
\mathbf{x}_{i+1}' = \text{Block}_i(\mathbf{x}_i')
$$

출력 투영은 다시 매핑합니다:

$$
\mathbf{x}_{\text{out}} = \mathbf{x}_{24}' W_{\text{out}} \in \mathbb{R}^{B \times S \times d_{\text{source}}}
$$

$$
\mathbf{o} = \mathbf{x}_{\text{out}} W_{\text{lm\_head}} \in \mathbb{R}^{B \times S \times V}
$$

가중치 공유(weight tying) 시, $W_{\text{lm\_head}} = W_{\text{embed}}^T$입니다.

### 영역 1: 순수 SSM (레이어 0–7)

각 블록은 잔차 연결과 함께 RMSNorm 후 Axon-SSM을 적용합니다:

$$
\mathbf{x}_{i+1}' = \mathbf{x}_i' + \text{AxonSSM}(\text{RMSNorm}(\mathbf{x}_i'))
$$

**토큰당 메모리**: $\mathcal{O}(d_{\text{model}} \cdot N_{\text{state}}) = \mathcal{O}(2560 \times 16) = \mathcal{O}(1)$ — 시퀀스 길이에 관계없이 일정합니다.

### 영역 2: SWA + MoE (레이어 8–15)

각 블록은 각각 자체 잔차를 갖는 어텐션과 MoE를 순차적으로 적용합니다:

$$
\mathbf{x}_{i+1}^{(1)} = \mathbf{x}_i' + \text{SWA}(\text{RMSNorm}(\mathbf{x}_i'))
$$

$$
\mathbf{x}_{i+1}' = \mathbf{x}_{i+1}^{(1)} + \text{MoE}(\text{RMSNorm}(\mathbf{x}_{i+1}^{(1)}))
$$

슬라이딩 윈도우 어텐션은 윈도우 크기 $W = 4096$, 차원 $d_h = 80$의 헤드 $H = 32$개를 사용합니다:

$$
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_h}} \odot \mathbf{M}_{\text{SWA}}\right)\mathbf{V}
$$

여기서 $\mathbf{M}_{\text{SWA}}$는 슬라이딩 윈도우 마스크로, $|i - j| > W$일 때 $M_{ij} = -\infty$입니다.

**토큰당 메모리**: KV 캐시를 위해 $\mathcal{O}(W \cdot d_{\text{model}})$ (최대 4K 위치로 제한).

### 영역 3: SSM + MoE (레이어 16–23)

각 블록은 각각 잔차를 갖는 SSM과 MoE를 순차적으로 적용합니다:

$$
\mathbf{x}_{i+1}^{(1)} = \mathbf{x}_i' + \text{AxonSSM}(\text{RMSNorm}(\mathbf{x}_i'))
$$

$$
\mathbf{x}_{i+1}' = \mathbf{x}_{i+1}^{(1)} + \text{MoE}(\text{RMSNorm}(\mathbf{x}_{i+1}^{(1)}))
$$

**토큰당 메모리**: $\mathcal{O}(1)$ — 어텐션 KV 캐시 없이 SSM 순환 상태만 사용합니다.

### 매개변수 예산

| 영역 | 레이어 | 레이어당 매개변수 | 역할 |
|------|--------|-------------------|------|
| 1 (SSM) | 0–7 | SSM 투영 + 합성곱 | 컨텍스트 흡수 |
| 2 (SWA+MoE) | 8–15 | 어텐션 + 8개 전문가 + 공유 전문가 | 심층 추론 |
| 3 (SSM+MoE) | 16–23 | SSM + 8개 전문가 + 공유 전문가 | 출력 합성 |

MoE는 중간 차원 4,096의 8개 전문가에 대해 공유 전문가 top-2 라우팅을 사용합니다. 공유 전문가는 항상 활성화되어 희소 전문가와 함께 조밀한 용량을 제공합니다.

## Bit-Axon에서의 구현

### 블록 변형

세 개의 블록 클래스가 영역 유형을 구현합니다:

| 클래스 | 영역 | 소스 |
|--------|------|------|
| `AxonSSMBlock` | 1 (SSM) | `layers/block.py` |
| `AxonSWAMoEBlock` | 2 (SWA+MoE) | `layers/block.py` |
| `AxonSSMMoEBlock` | 3 (SSM+MoE) | `layers/block.py` |

### 캐시 관리

```python
# model.py — SWA+MoE 레이어만 KV 캐시를 생성합니다
def _create_caches(self) -> list:
    caches = []
    for i in range(self.config.num_layers):
        if self._get_layer_type(i, self.config.num_layers) == "swa_moe":
            caches.append(KVCache())
        else:
            caches.append(None)
    return caches
```

### 차원 브리지

$d_{\text{source}} = 2048$ 차원은 Qwen2.5-3B로부터의 가중치 이식을 가능하게 합니다:

| 투영 | 형태 | 목적 |
|------|------|------|
| `embed_tokens` | $(V, 2048)$ | 토큰 임베딩 (lm_head와 공유) |
| `input_proj` | $(2048, 2560)$ | 소스 → 내부 차원 |
| `output_proj` | $(2560, 2048)$ | 내부 → 소스 차원 |
| `lm_head` | $(2048, V)$ | 로짓 (임베딩과 가중치 공유) |

## 참고 문헌

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 23.
- Qwen Team (2024). *Qwen2.5 Technical Report*.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). *Longformer: The Long-Document Transformer*. arXiv:2004.05150.
