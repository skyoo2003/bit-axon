# 양자화

32억 파라미터 모델을 6.4GB에서 1.76GB로 축소하여 16GB MacBook에 편안하게 올립니다. 이 가이드에서는 NF4 양자화(구현 완료), QLoRA 학습 워크플로우, 병합 및 재양자화 파이프라인, 그리고 계획 중인 양자화 방법들을 다룹니다.

---

## 왜 양자화하나요?

FP16 정밀도의 32억 파라미터 모델은 가중치만으로 약 6.4GB의 메모리가 필요합니다. 통합 메모리 16GB의 MacBook Air M4에서 KV cache, 활성화 값, 운영체제를 위한 공간이 거의 남지 않습니다.

양자화는 가중치 정밀도를 16-bit 부동소수점에서 4-bit 정수로 낮추어, 정확도 손실을 최소화하면서 가중치 메모리를 약 4배 줄입니다.

!!! tip "진정한 병목은 RAM, 연산이 아닙니다"
    Apple Silicon에는 충분한 연산 대역폭이 있습니다. 병목은 모델을 macOS, 컨텍스트 윈도우, KV cache와 함께 16GB 통합 메모리에 맞추는 것입니다. 양자화가 이를 가능하게 합니다.

## 메모리 절감

| 구성 | 가중치 메모리 | 추론 메모리 (4K ctx) | 추론 메모리 (64K ctx) |
|:---|---:|---:|---:|
| FP16 (미양자화) | ~6,400 MB | 메모리 부족 | 메모리 부족 |
| **Q4 (NF4)** | **~1,760 MB** | **~2,500 MB** | **~2,900 MB** |
| QLoRA 학습 (4-bit 베이스) | ~1,760 MB | ~3,200 – 3,700 MB | — |

Q4는 가중치 저장을 6.4GB에서 1.76GB로 낮춥니다. **3.6배 감소**이며, 64K 컨텍스트 윈도우를 KV cache와 함께 3GB 미만으로 유지할 수 있습니다.

---

## NF4 양자화 (구현 완료)

Bit-Axon은 **4-bit NormalFloat (NF4)** 양자화를 사용합니다. 정규 분포 신경망 가중치에 최적화된 아핀 양자화 방식으로, 가중치를 `group_size`(기본값 64) 단위의 블록으로 그룹화하고 그룹별 스케일과 바이어스를 계산한 뒤 각 가중치를 4비트로 패킹합니다.

내부적으로 Bit-Axon은 MLX 프리미티브를 사용합니다.

- **`mx.quantize(weight, group_size, bits=4)`**: FP16 가중치를 그룹별 스케일과 바이어스와 함께 4-bit 정수로 패킹
- **`mx.dequantize(packed, scales, biases, group_size, bits)`**: 다시 FP16으로 언패킹
- **`nn.QuantizedLinear.from_linear(linear, group_size, bits)`**: `nn.Linear` 레이어를 Apple Silicon에서 4-bit matmul을 네이티브로 실행하는 양자화 버전으로 교체

### CLI

```bash title="모델을 4-bit로 양자화"
bit-axon quantize ./model --bits 4 --group-size 64
```

이 명령은 `./model`에서 FP16 모델을 로드하고, 모든 `nn.Linear`를 `nn.QuantizedLinear`로 교체한 뒤 양자화된 가중치를 `./model/q4`에 저장합니다.

```bash title="전체 옵션"
bit-axon quantize ./model \
  --output ./model-q4 \
  --bits 4 \
  --group-size 64
```

| 플래그 | 기본값 | 설명 |
|:-----|:--------|:------------|
| `--output` / `-o` | `<model>/q4` | 출력 디렉토리 |
| `--bits` / `-b` | `4` | 양자화 비트 폭 |
| `--group-size` / `-g` | `64` | 아핀 양자화의 그룹 크기 |

### Python API

#### `quantize_nf4`

단일 가중치 텐서를 4-bit NormalFloat 형식으로 패킹합니다.

```python
import mlx.core as mx
from bit_axon.quantization import quantize_nf4, dequantize_nf4

# weight: mx.array, 형태 (output_dim, input_dim), dtype float16
packed, scales, biases = quantize_nf4(weight, group_size=64)

# packed: uint32 배열 (각 요소는 8개의 4-bit 가중치를 저장)
# scales: float16 배열, 형태 (output_dim, input_dim // group_size)
# biases: float16 배열, 형태 (output_dim, input_dim // group_size)

# FP16으로 언패킹
restored = dequantize_nf4(packed, scales, biases, group_size=64, bits=4)
```

#### `replace_linear_with_quantized`

모델을 재귀적으로 순회하며 모든 `nn.Linear` 레이어를 `nn.QuantizedLinear`로 교체합니다.

```python
from bit_axon import BitAxonModel, BitAxonConfig
from bit_axon.quantization import replace_linear_with_quantized

config = BitAxonConfig()
model = BitAxonModel(config)

# 모든 nn.Linear를 nn.QuantizedLinear로 교체 (in-place)
model = replace_linear_with_quantized(model, group_size=64, bits=4)

# 모델이 완전히 양자화됨: 추론 준비 완료
```

!!! note "MoE 지원"
    `replace_linear_with_quantized`는 MoE expert 목록을 올바르게 처리합니다. dict 스타일 자식(이름이 있는 레이어)과 list 스타일 자식(`MixtureOfExperts` 내부의 expert 배열) 모두를 순회하며 모든 expert의 선형 레이어를 양자화합니다.

### 동작 원리

```
FP16 가중치 행렬
┌──────────────────────────────┐
│ w₁  w₂  w₃  w₄  w₅  w₆ ... │  형태: (out, in), float16
└──────────────────────────────┘
           │
           ▼  64개씩 그룹으로 분할
┌──────────────────────────────┐
│ Group 0: [w₁..w₆₄]  → scale₀, bias₀, 4-bit 코드
│ Group 1: [w₆₅..w₁₂₈] → scale₁, bias₁, 4-bit 코드
│ ...                          │
└──────────────────────────────┘
           │
           ▼  8개의 코드를 하나의 uint32로 패킹
┌──────────────────────────────┐
│ packed: uint32 배열         │  float16보다 4× 작음
│ scales: float16 배열        │  64개 그룹당 1개의 스케일
│ biases: float16 배열        │  64개 그룹당 1개의 바이어스
└──────────────────────────────┘
```

64개 가중치의 각 그룹은 자체 아핀 매핑을 갖습니다: `w_quantized = (w - bias) / scale`. 4-bit 코드는 하나의 `uint32` 워드당 8개씩 패킹됩니다. 추론 중 `nn.QuantizedLinear`은 즉시 언패킹하고 양자화 영역에서 matmul을 계산합니다. 가중치 행렬에 대해 FP16 중간값을 사용하지 않습니다.

---

## QLoRA: 학습 워크플로우에서의 양자화

QLoRA (Quantized Low-Rank Adaptation)는 베이스 모델을 Q4로 고정하고 그 위에 작은 LoRA 또는 DoRA 어댑터만 학습합니다. 전체 FP16 학습에 근접한 파인튜닝 품질을 유지하면서 메모리 사용량을 ~3.2~3.7GB로 유지합니다.

```
┌─────────────────────────────────────────────┐
│ 베이스 모델 (고정, Q4)                      │
│  ┌───────────────────────────────────────┐  │
│  │ nn.QuantizedLinear (4-bit 가중치)    │  │
│  └──────────────┬────────────────────────┘  │
│                 │                           │
│                 ▼                           │
│  ┌───────────────────────────────────────┐  │
│  │ LoRA: A @ B (랭크 8, float16)        │  │  ← 학습됨
│  │ 또는 DoRA: 크기 + 방향               │  │  ← 학습됨
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### QLoRA로 학습

```bash title="QLoRA로 파인튜닝 (4-bit 베이스 + LoRA 어댑터)"
bit-axon train data.json \
  --lora-rank 8 \
  --quantize-bits 4 \
  --quantize-group-size 64
```

내부적으로 학습 파이프라인은 다음을 수행합니다.

1. **모델 로드** (FP16)
2. **양자화**: 모든 `nn.Linear` → `nn.QuantizedLinear` (Q4, group_size=64)
3. **LoRA/DoRA 적용**: 고정된 양자화 레이어 위에 어댑터 적용
4. **학습**: 어댑터 파라미터(`lora_a`, `lora_b`, 선택적으로 DoRA 크기 `.m`)만 학습
5. **저장**: 어댑터 가중치만 저장 (몇 MB)

### Python API

```python
from bit_axon import BitAxonModel, BitAxonConfig
from bit_axon.quantization import replace_linear_with_quantized
from bit_axon.training import apply_lora_to_model

# 1단계: 모델 로드
config = BitAxonConfig()
model = BitAxonModel(config)

# 2단계: 베이스를 Q4로 양자화
model = replace_linear_with_quantized(model, group_size=64, bits=4)

# 3단계: LoRA 어댑터로 래핑 (랭크 8)
model = apply_lora_to_model(
    model,
    rank=8,
    alpha=16,
    use_dora=False,
    target_modules=["attention", "moe"],
)

# 4단계: 어댑터 파라미터만 학습
# lora_a, lora_b (및 DoRA의 .m)만 학습 가능합니다.
# 베이스 양자화 가중치는 고정됩니다.
```

!!! warning "QLoRA 중 베이스 가중치를 업데이트하지 마세요"
    `Trainer.get_trainable_params()`는 어댑터 파라미터로 엄격하게 필터링합니다. 사용자 정의 학습 루프를 작성하는 경우 양자화된 베이스 가중치를 반드시 고정하세요. 그렇지 않으면 정밀도가 저하된 양자화 matmul을 통해 기울기를 계산하게 됩니다.

---

## 병합 및 재양자화

학습 후 LoRA/DoRA 어댑터를 베이스 모델에 병합하고 효율적인 추론을 위해 재양자화합니다.

```
Q4 베이스 + LoRA 어댑터
         │
         ▼  merge_adapters()
FP16 베이스 (LoRA 퓨즈됨, 양자화 해제됨)
         │
         ▼  quantize_model()
Q4 병합 모델 (배포 준비 완료)
```

### CLI

```bash title="어댑터 병합 및 재양자화"
bit-axon merge ./base-model \
  --adapter ./adapter-checkpoint \
  --output ./merged-model \
  --bits 4 \
  --group-size 64
```

기본적으로 병합 명령은 어댑터 퓨징 후 재양자화합니다. 병합된 모델을 FP16으로 유지하려면(예: 추가 처리용) `--no-re-quantize`를 사용하세요.

```bash title="재양자화 없이 병합"
bit-axon merge ./base-model \
  --adapter ./adapter-checkpoint \
  --output ./merged-model \
  --no-re-quantize
```

### Python API

```python
from bit_axon.training import load_and_merge

# 엔드투엔드: 베이스 + 어댑터 로드, 병합, 재양자화, 저장
load_and_merge(
    base_model_path="./base-model",
    adapter_path="./adapter-checkpoint",
    output_dir="./merged-model",
    quantize_after_merge=True,
    bits=4,
    group_size=64,
    lora_rank=8,
)
```

각 단계를 더 세밀하게 제어하려면:

```python
from bit_axon.training import (
    merge_adapters,
    dequantize_model,
    quantize_model,
    save_merged_model,
)

# 1단계: LoRA/DoRA 어댑터를 베이스에 병합
model = merge_adapters(model)  # 모든 LoRALinear/DoRALinear에서 .fuse() 호출

# 2단계: Q4에서 FP16으로 양자화 해제
model = dequantize_model(model)  # QuantizedLinear → nn.Linear (float16)

# 3단계: Q4로 재양자화
model = quantize_model(model, bits=4, group_size=64)

# 4단계: 저장
save_merged_model(model, output_dir="./merged-model", config=config, tokenizer=tokenizer)
```

!!! tip "평가를 위해 병합 후 별도로 양자화하세요"
    전체 파이프라인은 재양자화 전 병합된(미양자화) 모델에서 perplexity를 평가합니다. 이렇게 하면 양자화 노이즈 없이 깔끔한 품질 메트릭을 얻을 수 있습니다. 최종 배포 모델만 재양자화됩니다.

---

## 계획 중인 양자화 방법

Bit-Axon에는 두 가지 추가 양자화 방식이 개발 중입니다. 이들은 **아직 구현되지 않았습니다**. 해당 모듈에는 스텁만 포함되어 있습니다.

### 삼진 양자화 (1.58-bit BitNet)

**파일:** `src/bit_axon/quantization/ternary.py` (스텁)

삼진(1.58-bit) 양자화는 각 가중치를 세 가지 값 중 하나로 표현합니다: `{-1, 0, +1}`. 이는 matmul에서 곱셈을 완전히 제거하여 부호 반전과 덧셈으로 대체하며, BitNet b1.58의 핵심 아이디어입니다.

| 정밀도 | 가중치당 비트 | 메모리 (3.2B) |
|:----------|----------------:|--------------:|
| FP16 | 16 | ~6,400 MB |
| NF4 | 4 | ~1,760 MB |
| **삼진** | **1.58** | **~700 MB** |

!!! info "상태"
    삼진 모듈(`quantization/ternary.py`)은 구현이 없는 스텁입니다. 향후 릴리즈에 계획되어 있습니다.

### TurboQuant KV Cache 압축

**파일:** `src/bit_axon/quantization/turboquant.py` (스텁)

TurboQuant는 시퀀스 길이에 따라 선형적으로 증가하는 KV cache를 압축하여 긴 컨텍스트 추론의 메모리 사용량을 줄입니다. 이 기법은 학습된 KV cache 양자화에 대한 ICLR 2026 연구를 기반으로 합니다.

64K 컨텍스트 윈도우에서 KV cache 메모리가 지배적일 수 있습니다. TurboQuant는 최대 컨텍스트 길이에서도 전체 추론 메모리를 3GB 이하로 유지하는 것을 목표로 합니다.

!!! info "상태"
    TurboQuant 모듈(`quantization/turboquant.py`)은 구현이 없는 스텁입니다. 향후 릴리즈에 계획되어 있습니다.

---

## 빠른 참조

```bash
# 모델 양자화
bit-axon quantize ./model --bits 4 --group-size 64

# QLoRA로 학습
bit-axon train data.json --lora-rank 8

# 어댑터 병합 및 재양자화
bit-axon merge ./base-model --adapter ./adapter --output ./merged

# 추론 실행 (로드 시 자동 양자화)
bit-axon run --model ./model --prompt "Hello, world!"
```

```python
# 양자화
from bit_axon.quantization import quantize_nf4, replace_linear_with_quantized
packed, scales, biases = quantize_nf4(weight, group_size=64)
model = replace_linear_with_quantized(model, group_size=64, bits=4)

# 병합
from bit_axon.training import load_and_merge
load_and_merge("./base", "./adapter", "./output", quantize_after_merge=True)
```
