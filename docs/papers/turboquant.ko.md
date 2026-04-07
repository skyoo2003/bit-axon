# TurboQuant: KV Cache Compression

**상태**: :fontawesome-solid-clock:{ . amber } 예정
**소스**: [`src/bit_axon/quantization/turboquant.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/quantization/turboquant.py) *(스텁)*

## 요약

TurboQuant는 Bit-Axon에서 긴 컨텍스트 추론의 메모리 사용량을 줄이기 위해 계획된 KV 캐시 압축 기술입니다. 컨텍스트 길이가 목표인 64K 토큰으로 증가함에 따라, 8개 슬라이딩 윈도우 어텐션 레이어의 KV 캐시가 주요 메모리 소비원이 됩니다. TurboQuant는 캐시된 키와 값 텐서를 품질 손실을 최소화하면서 낮은 정밀도로 압축하는 것을 목표로 하며, 전체 64K 컨텍스트를 16 GB MacBook Air의 메모리 예산 내에 맞추는 것을 가능하게 합니다.

!!! warning "예정 기능"

    TurboQuant는 ICLR 2026 제출 논문에서 참조되었으며 아직 구현되지 않았습니다. 소스 파일은 현재 스텁을 포함하고 있습니다. 아래의 세부 사항은 계획된 설계를 설명합니다.

## 핵심 기여 (예정)

1. **KV 캐시 양자화** — 캐시된 $\mathbf{K}$ 및 $\mathbf{V}$ 텐서를 FP16에서 4비트 표현으로 압축합니다.
2. **SWA 레이어와의 통합** — KV 캐시가 유지되는 8개 슬라이딩 윈도우 어텐션 레이어(영역 2)에 선택적으로 적용됩니다.
3. **메모리 목표** — 64K 컨텍스트의 총 추론 메모리를 약 2,900 MB에서 2,500 MB 미만으로 감소시킵니다.

## 수학적 기반

### KV 캐시 메모리 모델

윈도우 크기 $W$인 슬라이딩 윈도우 어텐션의 경우, 레이어당 KV 캐시는 다음을 필요로 합니다:

$$
M_{\text{KV}} = 2 \times B \times W \times d_{\text{model}} \times \text{sizeof}(\text{dtype})
$$

여기서 계수 2는 별개의 $\mathbf{K}$ 및 $\mathbf{V}$ 텐서를 고려합니다. Bit-Axon 영역 2 레이어의 경우:

- $W = 4096$, $d_{\text{model}} = 2560$, $B = 1$ (단일 배치)
- KV 캐시를 갖는 8개 레이어
- FP16 (요소당 2바이트):

$$
M_{\text{KV}}^{\text{FP16}} = 8 \times 2 \times 4096 \times 2560 \times 2 = 335.5 \text{ MB}
$$

### 양자화된 KV 캐시

TurboQuant는 KV 캐시의 4비트 양자화를 목표로 합니다. 압축 비율은 다음과 같습니다:

$$
r = \frac{\text{sizeof}(\text{FP16})}{\text{sizeof}(\text{Q4})} = \frac{16}{4} = 4\times
$$

양자화된 KV 캐시 메모리:

$$
M_{\text{KV}}^{\text{Q4}} = \frac{M_{\text{KV}}^{\text{FP16}}}{4} = 83.9 \text{ MB}
$$

### 양자화 함수

계획된 양자화는 FP16 값을 4비트 인덱스로 매핑합니다:

$$
\mathbf{K}_{\text{quantized}} = Q(\mathbf{K}) = \text{argmin}_{\mathbf{K}' \in \mathcal{C}_{4\text{-bit}}} \|\mathbf{K} - \mathbf{K}'\|_2
$$

여기서 $\mathcal{C}_{4\text{-bit}}$는 4비트 형식에서 표현 가능한 값의 집합입니다. 역양자화는 근사값을 복원합니다:

$$
\hat{\mathbf{K}} = DQ(\mathbf{K}_{\text{quantized}}) \approx \mathbf{K}
$$

### 양자화 하의 어텐션 품질

양자화된 KV를 사용한 어텐션 연산:

$$
\text{Attn}(\mathbf{Q}, \hat{\mathbf{K}}, \hat{\mathbf{V}}) = \text{softmax}\left(\frac{\mathbf{Q}\hat{\mathbf{K}}^T}{\sqrt{d_h}}\right)\hat{\mathbf{V}}
$$

품질 손실은 양자화 오차에 의해 제한됩니다:

$$
\|\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) - \text{Attn}(\mathbf{Q}, \hat{\mathbf{K}}, \hat{\mathbf{V}})\| \leq f(\|\mathbf{K} - \hat{\mathbf{K}}\|, \|\mathbf{V} - \hat{\mathbf{V}}\|)
$$

구체적인 양자화 방식(NF4, 균일, 또는 학습 기반)은 구현 중에 결정될 예정입니다.

## 구현 계획

### 통합 지점

| 구성 요소 | 통합 |
|-----------|------|
| `SlidingWindowAttention` | FP16 KV 캐시를 양자화된 캐시로 교체 |
| `KVCache` | 양자화/역양자화 메서드 추가 |
| `turboquant.py` | 핵심 양자화 프리미티브 |

### 계획된 API

```python
# 계획됨 (아직 구현되지 않음)
from bit_axon.quantization.turboquant import TurboQuant

quantizer = TurboQuant(bits=4)
# 추론 중:
# quantizer.compress(kv_cache)  # 각 어텐션 스텝 후 압축
# quantizer.decompress(kv_cache)  # 어텐션 연산을 위해 역양자화
```

### 메모리 예산 영향

| 구성 | KV 캐시 메모리 | 총 추론 메모리 |
|------|---------------|---------------|
| FP16, 4K 컨텍스트 | 335.5 MB | ~2,500 MB |
| FP16, 64K 컨텍스트 | N/A (윈도우 초과) | ~2,900 MB |
| TurboQuant Q4, 64K 컨텍스트 | ~83.9 MB | ~2,500 MB (목표) |

## 참고 문헌

- Dettmers, T., et al. (2024). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. (관련: NF4 양자화.)
- Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. (관련: KV 캐시 관리.)
- Liu, Z., et al. (2024). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. arXiv:2402.02750. (관련: KV 캐시 양자화.)
