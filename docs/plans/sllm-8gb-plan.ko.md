# Bit-Axon 3B: Master Plan

**"최소 비트, 최대 임펄스."**
_선형(Linear), 희소(Sparse), 양자화(Quantized) 아키텍처를 위한 확장 가능한 sLLM 엔진._

> **문서 버전**: v3.0 (통합 마스터 플랜)
> **대상 장치**: MacBook Air M4 (16GB 통합 메모리, 모델에 ~8GB 가용)
> **총 기간**: ~6개월 (5개 Phase)

---

## 1. 프로젝트 비전

Bit (최소 비트)와 Axon (최대 임펄스)의 결합입니다.

- **Bit (Minimal Bits):** 극단적인 양자화(Quantization)를 통해 메모리 대역폭의 한계를 최소화합니다. 가중치를 4-bit 또는 1.58-bit로 압축하여 16GB 통합 메모리에서도 여유롭게 구동합니다.
- **Axon (Maximal Impulse):** 인간의 신경 축삭(Axon)처럼 빠르고 효율적인 정보 전달. 선형(Linear) 및 희소(Sparse) 연산을 통해 최소한의 전력/발열로 최대의 추론/학습 속도를 달성합니다.
- **Target Environment:** 쿨링팬이 없는 MacBook Air M4 환경에서의 **발열 없는 로컬 학습 및 초고속 추론**.

**3대 핵심 아키텍처:**
1. **Linear Architecture**: Mamba-3 기반 SSM(State Space Model). 문맥 길이가 늘어도 메모리 $O(N)$으로 고정. KV-Cache 제거.
2. **Sparse Architecture**: Shared-Expert MoE. 총 3.2B 파라미터 중 토큰당 ~1.4B만 활성화. 60% 이상 수면(Sleep) 상태 유지.
3. **Quantized Architecture**: Bit-DoRA(학습) + NF4/1.58-bit(추론) + TurboQuant(KV 캐시 압축). 메모리 한계 돌파.

**5-Phase Roadmap (6개월):**
- Phase 1: Core Primitives (Week 1-4)
- Phase 2: Architecture Synthesis (Week 5-8)
- Phase 3: Thermal-Aware Training (Week 9-14)
- Phase 4: Alignment & Merging (Week 15-18)
- Phase 5: App & CLI Release (Week 19-24)

**기대 효과:**
1. **"No GPU, No Cloud"**: 값비싼 Nvidia GPU나 클라우드 서버 없이 팬리스 MacBook Air에서 학습-추론-배포의 풀 사이클 완성.
2. **지속 가능성 (Green AI)**: Sparse와 Quantized 기술 결합으로 전력 소모를 획기적으로 줄여, 배터리 상태에서도 장시간 LLM 구동.
3. **확장성 (Scalable)**: M4 Air에서 최소 단위로 동작하지만, Mac Studio(M4 Ultra) 등 하드웨어가 커지면 선형적으로 파라미터 확장 가능.

---

## 2. 아키텍처 설계

### 2-1. 모델 개요

| 속성 | 값 | 비고 |
|:-----|:---|:-----|
| 총 파라미터 | ~3.2B | 24-layer hybrid |
| 활성 파라미터/토큰 | ~1.4B | MoE top-2 + 1 shared |
| 레이어 수 | 24 | 샌드위치 구조 |
| Hidden dim | 2,560 | d_source_model=2048 (Qwen) |
| MoE 전문가 수 | 8 | top-2 + 1 shared expert |
| Context Window | 최대 64K | SSM + SWA hybrid |
| FP16 모델 크기 | ~6,400 MB | 전체 파라미터 |
| Q4 모델 크기 | ~1,760 MB | MLX affine, g=64 |

MoE의 희소성(Sparsity) 덕분에 토큰당 실제 활성 파라미터는 ~1.4B에 불과하여, FP16 기준 활성 가중치만 ~2.8GB로 8GB 제약에 잘 맞습니다. 추론 시 Q4 양자화로 ~1.76GB, 학습 시 QLoRA로 ~3.2-3.7GB 사용.

---

### 2-2. 3대 핵심 모듈

#### 2-2-1. Linear Module: Axon-SSM (State Space Model)

기존 Transformer의 Self-Attention을 제거하고 Mamba-3/Griffin 구조를 차용한 선형 순환 신경망. 과거 정보를 고정 크기의 'Hidden State'로 압축하여 저장하므로 KV-Cache가 필요 없습니다. 문맥이 수만 토큰으로 길어져도 메모리 증가가 $O(1)$입니다.

#### 2-2-2. Sparse Module: Shared-Expert MoE

MLP 층을 8개의 전문가 네트워크로 쪼갭니다. 항상 활성화되는 Shared Expert(범용 지식, ~0.2B)와 라우팅에 의해 선택되는 top-2 Routed Expert(~0.4B)를 결합합니다. 물리적 파라미터의 약 60%가 수면 상태를 유지하여 칩 온도를 낮게 유지합니다.

#### 2-2-3. Quantized Module: Bit-DoRA

LLM 속도는 연산력보다 '메모리 대역폭'에 달려 있습니다. 추론 시 NF4(4-bit NormalFloat) 또는 1.58-bit(Ternary, {-1, 0, 1})로 가중치를 극단적으로 압축합니다. 학습 시에는 DoRA(Weight-Decomposed LoRA)를 사용하여 가중치의 '크기(Magnitude)'와 '방향(Direction)'을 분리해 학습합니다. LoRA 수준의 극히 적은 메모리(1~2GB)만으로도 Full Fine-tuning에 버금가는 정확도를 냅니다.

#### 2-2-4. Quantization Module: TurboQuant

Google Research의 ICLR 2026 논문 기반 근-최적 벡터 양자화 엔진입니다. 두 단계로 동작합니다: Stage 1은 PolarQuant(무작위 회전 + Beta 분포 + 좌표별 최적 양자화), Stage 2는 QJL(1-bit 잔차 보정, 불편향 추정기). 왜곡률이 Shannon 하한의 약 2.7배 이내로 보장됩니다. 데이터 의존성 없이 즉시 적용 가능(Training-free). Layer 9-16의 SWA KV 캐시를 압축하여 6배 이상 메모리 절감. (자세한 내용은 Section 6 참조)

---

### 2-3. 24-Layer 샌드위치 구조

전체 24개 레이어를 세 구간으로 나누는 샌드위치 구조입니다. SSM 레이어는 KV 캐시가 없어 메모리 증가가 $O(1)$이고, SWA 레이어만 KV 캐시를 소모합니다.

```
Layer  1-8:  ████████████████████ Pure Axon-SSM (Linear, no KV cache)
Layer  9-16: ████████████████████ SWA + MoE (Attention Window + Sparse)
Layer 17-24: ████████████████████ SSM + MoE (Linear + Sparse)
```

| 구간 | 레이어 | 모듈 | KV 캐시 | 역할 |
|:-----|:------:|:-----|:-------:|:-----|
| 기초 지각망 | 1-8 | Axon-SSM | 없음 | 문맥 흡수, 선형 처리 |
| 심층 추론망 | 9-16 | SWA + MoE | 있음 | 국소 추론, 전문가 호출 |
| 최종 결합망 | 17-24 | SSM + MoE | 없음 | 출력 생성, 발열 제어 |

---

### 2-4. MLX 통합

Apple MLX Framework에 완벽히 결합(Native Integration)되도록 설계합니다. PyTorch로 구현하면 Mac에서 CUDA 최적화가 안 되어 비효율적입니다.

1. **MLX JIT (Just-In-Time) 컴파일:** 연산 그래프를 MLX의 `mx.compile` 데코레이터로 감싸서 M4의 NPU/GPU가 하나의 통합된 커널로 처리하도록 캐싱합니다. 모델 레벨 컴파일(모델 전체 forward pass)을 사용합니다. Layer-level 컴파일은 MLX의 모듈 참조 문제로 동작하지 않습니다.
2. **Unified Memory Zero-Copy:** MLX의 통합 메모리 특성(배열이 RAM과 VRAM을 구분하지 않음)을 활용하여, 양자화된 가중치를 메모리에 한 번만 로드하고 CPU/GPU/NPU가 데이터 복사 없이 동시에 접근합니다.

**주의사항:** MLX <= 0.31에서 `shapeless=True`는 matmul에 대해 깨져 있습니다. 출력 형상이 첫 번째 트레이스에서 캐시되어 이후 호출에서 잘못된 결과를 만듭니다. Shape-dependent 컴파일을 사용하며, 입력 형상이 바뀔 때 지연 재컴파일을 허용합니다.

---

### 2-5. 모델 설정

```python
from dataclasses import dataclass

@dataclass
class BitAxonConfig:
    """Bit-Axon 3B 모델 설정"""
    vocab_size: int = 32_000
    hidden_dim: int = 2_560          # d_model
    num_layers: int = 24
    num_heads: int = 32              # SWA heads, head_dim=64
    d_source_model: int = 2048       # Qwen2.5-3B bridge dimension

    # Axon-SSM (Mamba-3 style)
    ssm_d_state: int = 16            # 상태 벡터 차원
    ssm_d_conv: int = 4              # 1D 컨볼루션 커널
    ssm_expand: int = 3              # 확장 비율

    # Sliding Window Attention (Layer 9-16 only)
    swa_window_size: int = 4_096     # SWA 윈도우

    # Shared-Expert MoE
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_intermediate_dim: int = 4_096   # 전문가 FFN 차원
    moe_shared_expert: bool = True

    # General
    weight_tying: bool = True        # embedding = output head
    max_seq_len: int = 65_536        # 최대 64K 컨텍스트
    rms_norm_eps: float = 1e-6
```

### 2-6. MacBook Air M4 적합성

| 직면한 한계 (Air M4) | Bit-Axon의 해결책 | 적용 기술 |
|:---------------------|:------------------|:---------|
| 쿨링팬 부재 (발열) | 연산의 60%를 비활성화하여 칩 부하 최소화 | Sparse (Shared MoE) |
| 제한된 메모리 용량 | 가중치를 압축하고 긴 문맥 캐싱 방지 | Quantized (4-bit) + Linear (SSM) |
| 학습(SFT) 시 메모리 부족 | 모델 전체가 아닌 방향성/크기만 분리하여 학습 | Bit-DoRA (PEFT) |

---

## 3. 메모리 예산

### 3-1. 8GB 가용 메모리 분석

16GB 통합 메모리 중 모델에 할당 가능한 메모리를 정확히 산정합니다. macOS와 Metal의 동작 방식 때문에 "16GB 전체가 모델에 쓰이는 것"은 아닙니다.

| 항목 | 크기 | 비고 |
|:-----|:---:|:-----|
| **물리 RAM** | 16,384 MB | MacBook Air M4 16GB |
| macOS 시스템 사용 | ~2,500 MB | macOS 15 + services |
| 일반 앱 | ~2,000 MB | Browser, IDE, etc. |
| MLX 오버헤드 | ~500 MB | Framework runtime |
| Metal GPU 안전 마진 | ~1,500 MB | Swap prevention buffer |
| **모델 가용** | **~8,000 MB** | **이것이 제약 조건입니다** |

> 다른 앱을 모두 닫으면 ~10GB까지 늘어날 수 있지만, 안정적인 동작을 위해 ~8GB를 기준으로 계획합니다.

---

### 3-2. 추론 메모리

FP16으로 구동하면 4K 컨텍스트에서 ~7.2GB로 매우 빡빡합니다. Q4 양자화가 기본 전략이며, 긴 컨텍스트에서는 TurboQuant로 KV 캐시를 추가 압축합니다.

**모델 가중치별 메모리:**

| 정밀도 | 가중치 | 비고 |
|:------:|:------:|:-----|
| FP16 | 6,400 MB | 3.2B x 2 bytes |
| Q8 (affine) | ~3,200 MB | |
| Q4 (affine, g=64) | ~1,760 MB | MLX `nn.QuantizedLinear` |
| 1.58-bit (Ternary) | ~640 MB | BitNet, experimental |

**SSM 상태 메모리 (고정, Layer 1-8, 17-24):** ~1.2MB

**SWA KV 캐시 (Layer 9-16, 32 heads, head_dim=64, FP16):**

| 컨텍스트 길이 | KV 캐시 (FP16) | KV 캐시 (TurboQuant 3-bit) |
|:-------------:|:--------------:|:--------------------------:|
| 1,024 | 64 MB | ~11 MB |
| 4,096 | 256 MB | ~43 MB |
| 16,384 | 1,024 MB | ~171 MB |
| 32,768 | 2,048 MB | ~341 MB |
| 65,536 | 4,096 MB | ~683 MB |

> 계산식: 8 layers x 2 (K+V) x 32 heads x 64 dim x seq_len x 2 bytes = 65,536 x seq_len bytes

**추론 총 메모리:**

| 정밀도 | 4K ctx | 16K ctx | 32K ctx | 64K ctx |
|:------:|:------:|:-------:|:-------:|:-------:|
| FP16 (no TQ) | ~7,169 MB | ~7,925 MB | ~8,949 MB | ~10,901 MB |
| Q4 (no TQ) | ~2,529 MB | ~3,285 MB | ~4,309 MB | ~6,261 MB |
| Q4 + TQ 3-bit | ~2,306 MB | ~2,432 MB | ~2,602 MB | ~2,944 MB |

**핵심 결론**: Q4 양자화만으로도 32K까지 8GB 안에 수용. 64K에서는 TurboQuant 필수.

---

### 3-3. 학습 메모리

From-scratch 사전학습(FP16 전체 미세조정)은 ~38.4GB로 물리적으로 불가능합니다. QLoRA(Q4 동결 베이스 + LoRA 어댑터)로 학습 메모리를 ~3.2-3.7GB로 압축합니다.

**Full fine-tuning (불가):**

| 항목 | 크기 |
|:-----|:---:|
| 가중치 FP16 | 6,400 MB |
| 기울기 FP16 | 6,400 MB |
| Adam optimizer (m, v FP32) | 25,600 MB |
| **총계** | **~38,400 MB** |

**QLoRA 학습 (권장):**

| 항목 | 크기 | 비고 |
|:-----|:---:|:-----|
| 베이스 가중치 Q4 (동결) | ~1,760 MB | No gradients |
| LoRA 파라미터 (r=16, ~0.5%) | ~32 MB | Trainable |
| LoRA 기울기 (FP16) | ~32 MB | |
| LoRA Adam (m, v FP32) | ~256 MB | |
| 활성값 (checkpointed, B=1, T=2048) | ~500-1,000 MB | |
| SWA KV 캐시 (T=2048) | ~128 MB | |
| MLX 오버헤드 | ~500 MB | |
| **총계** | **~3,208-3,708 MB** | 8GB 안에 수용 |

---

## 4. 학습 계획

### 4-1. QLoRA 학습 전략

From-scratch 학습이 불가하므로 **듀얼 트랙 접근**을 사용합니다. 기존 3B 오픈소스 모델(예: Qwen2.5-3B)의 가중치를 Bit-Axon 아키텍처에 포팅한 후, QLoRA로 도메인 특화 파인튜닝합니다.

| 트랙 | 방식 | 메모리 | 목적 |
|:-----|:-----|:------:|:-----|
| **Track A: QLoRA SFT** | Q4 동결 + Bit-DoRA | ~3.5 GB | 도메인 특화 파인튜닝 |
| **Track B: ORPO 정렬** | Q4 동결 + ORPO | ~3.8 GB | 선호도 기반 정렬 |

> ORPO는 참조 모델 없이 SFT와 선호도 정렬을 동시에 수행하므로 8GB에서도 가능합니다.

---

### 4-2. QLoRA 학습 설정

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from mlx.nn import value_and_grad

def setup_qlora_training(model_path: str, lora_rank: int = 16):
    """QLoRA 학습 설정 (MLX)"""
    # 1. 모델 로드 및 Q4 양자화
    model = BitAxonModel.load(model_path)
    model.quantize(bits=4, group_size=64)  # MLX native Q4

    # 2. LoRA 어댑터 삽입 (동결 베이스 위에)
    #    Bit-DoRA 방식: 크기(Magnitude)와 방향(Direction) 분리
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            module.enable_lora(r=lora_rank, alpha=lora_rank * 2)
            module.enable_dora()  # 방향/크기 분리

    # 3. 옵티마이저 (어댑터 파라미터만)
    trainable_params = [
        (k, v) for k, v in model.parameters().items()
        if "lora" in k or "dora" in k
    ]
    optimizer = AdamW(
        learning_rate=2e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    return model, optimizer, trainable_params


def train_step(model, batch, optimizer, config):
    """QLoRA 단일 학습 스텝"""
    input_ids, labels = batch

    def loss_fn(model):
        logits = model(input_ids)
        logits = logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        loss = mx.mean(
            mx.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels_shifted.reshape(-1),
            )
        )
        return loss

    loss, grads = value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss
```

**QLoRA 하이퍼파라미터:**

| 파라미터 | 값 | 비고 |
|:---------|:---:|:-----|
| LoRA rank | 16 | 모델의 ~0.5% |
| LoRA alpha | 32 | rank x 2 |
| Learning rate | 2e-4 | 코사인 어닐링 |
| Batch size | 1 | B=1 (메모리 절약) |
| Gradient accumulation | 8 | 유효 배치 8 |
| Sequence length | 2,048 | |
| Max steps | ~10K | 도메인 SFT |
| Gradient clip | 1.0 | max_norm |

---

### 4-3. 열관리 스케줄러

3B 모델은 250M보다 더 많은 연산을 수행하므로 발열 관리가 더 중요합니다. macOS의 `powermetrics` 명령어나 IOKit을 Python(subprocess)으로 호출하여 SoC 온도를 실시간(1초 단위)으로 추적합니다. 온도 데이터와 MLX 학습 루프를 연동하여 발열을 제어합니다. MoE의 희소성이 칩 부하를 줄여주는 점을 활용합니다.

| 온도 | 동작 | 영향 |
|:----:|:-----|:-----|
| < 75C | 정상 속도 | 없음 |
| 75-85C | LoRA rank 절반 | 메모리는 동일, 연산 감소 |
| 85-95C | 0.5초 대기 | 약간 지연 |
| > 95C | 학습 정지 | 체크포인트 후 재시작 |

> **MoE 발열 이점**: 토큰당 ~1.4B 활성 파라미터이므로, 밀집(Dense) 3B 모델 대비 약 56%의 연산량만 사용. 팬리스 MacBook Air에서도 안정적인 학습 가능성이 높습니다.

---

## 5. 추론 계획

### 5-1. 양자화 전략

Q4 양자화를 기본으로 사용합니다. MLX의 `nn.QuantizedLinear`은 Metal-최적화 커널을 사용하므로, 커스텀 NF4 dequantize+matmul보다 훨씬 빠릅니다. MLX에 내장된 `nn.quantize()`를 사용하여 모든 `nn.Linear`를 `nn.QuantizedLinear`로 교체합니다.

```python
def prepare_inference_model(model_path: str, bits: int = 4):
    """추론용 모델 준비"""
    model = BitAxonModel.load(model_path)
    model.quantize(bits=bits, group_size=64)

    # 선택적: MLX JIT 컴파일 (속도 향상)
    # MoE numpy interop 패치가 필요합니다 (pure MLX MoE forward로 교체)
    model.compile()

    return model
```

| 방식 | MLX | Metal | 속도 | 용도 |
|:-----|:---:|:-----:|:----:|:----|
| FP16 | 네이티브 | 동작 | 기준 | 디버깅 |
| Q4 affine | 네이티브 | 동작 | ~2-3x 빠름 | **기본 추론** |
| 1.58-bit Ternary | 커스텀 | 동작 | ~4-6x 빠름 | 실험적 / 최적화 |

---

### 5-2. 컨텍스트 길이별 분석

사용 시나리오에 따라 권장 설정이 다릅니다. 일반적인 대화/코딩은 4K면 충분하며, 긴 문서 분석이 필요할 때 TurboQuant를 활성화합니다.

| 시나리오 | 컨텍스트 | 설정 | 총 메모리 |
|:---------|:--------:|:-----|:---------:|
| 챗봇 | 4K | Q4 | ~2.5 GB |
| 코드 생성 | 8K | Q4 | ~2.8 GB |
| 문서 요약 | 16K | Q4 | ~3.3 GB |
| PDF 분석 | 32K | Q4 | ~4.3 GB |
| 64K 전체 | 64K | Q4 + TurboQuant | ~2.9 GB |
| 64K (TQ 없음) | 64K | Q4 only | ~6.3 GB |

---

### 5-3. 예상 추론 성능

M4 메모리 대역폭 ~120 GB/s 기준 추정치입니다. Q4 가중치(~1.76GB) 로드 시간은 ~15ms입니다.

| 메트릭 | Q4 (MLX) | Q4 + Compile | 비고 |
|:-------|:--------:|:------------:|:-----|
| 모델 로드 | ~0.5s | ~0.8s | 컴파일 오버헤드 포함 |
| TTFT (4K) | ~200ms | ~100ms | Prefill |
| 토큰 생성 (tok/s) | ~40-60 | ~60-80 | Decode, B=1 |
| 메모리 (4K ctx) | ~2.5 GB | ~2.5 GB | 동일 |

---

## 6. TurboQuant 통합

### 6-1. 기술 개요

TurboQuant는 **Google Research**에서 개발하여 **ICLR 2026**에 발표된 온라인 벡터 양자화(Online Vector Quantization) 알고리즘입니다. 기존 양자화 방식(NF4, AWQ, GPTQ 등)의 한계를 극복하는 이론적 기반이 강건한 새로운 접근법을 제시합니다.

| 속성 | 기존 방식 (NF4, AWQ, GPTQ) | TurboQuant |
|:-----|:--------------------------|:-----------|
| **대상** | 모델 가중치 (정적, 1회 적용) | KV 캐시 (동적, 실시간) + 가중치 |
| **데이터 의존성** | 캘리브레이션 데이터 필요 | **Data-oblivious**: 데이터 불필요 |
| **메모리 오버헤드** | 블록별 양자화 상수 저장 | **제로 오버헤드**: 상수 저장 불필요 |
| **학습/튜닝** | 일부 방식은 미세조정 필요 | **학습 없음 (Training-free)** |
| **이론적 보장** | 경험적 (Empirical) | **정보이론적 근-최적 (Near-optimal)** |
| **적용 시점** | 사전 처리 (Offline) | **온라인 즉시 (Online)** |

---

### 6-2. 2단계 알고리즘 (PolarQuant + QJL)

#### Stage 1: PolarQuant (고품질 MSE 압축)

1. **Random Rotation**: 입력 벡터에 무작위 직교 행렬을 곱해 회전
2. **Beta 분포 유도**: 회전된 각 좌표가 집중된 Beta 분포를 따르도록 변환
3. **좌표별 최적 양자화**: 고차원에서 좌표들이 거의 독립적이 되므로, 각 좌표에 대해 개별적으로 최적 스칼라 양자화(Lloyd-Max) 적용
4. **극좌표 변환**: 직교 좌표를 반지름(Radius) + 각도(Angle)로 변환하여 정규화 오버헤드 제거

> 무작위 회전으로 데이터의 기하학적 구조를 단순화하고, 기존 방식에서 숨겨진 "메모리 오버헤드"를 원천적으로 제거합니다.

#### Stage 2: QJL — 1-bit 잔차 보정 (Quantized Johnson-Lindenstrauss)

MSE-optimal 양자화는 내적(Inner Product) 추정 시 편향(Bias)을 유발합니다. 이를 해결하기 위해:

1. Stage 1의 **잔차(Residual)** 에 1-bit QJL 변환 적용
2. 각 잔차 좌표를 단일 **부호 비트** ({+1, -1}) 로 압축
3. 특수한 **불편향(Unbiased) 추정기**로 내적 오차를 제거

> 결과: 편향이 없는 근-최적 내적 양자화기(Unbiased Inner Product Quantizer) 완성.

---

### 6-3. 이론적 성능 보장

TurboQuant의 왜곡률(Distortion)은 정보이론적 하한(Shannon Lower Bound)의 약 2.7배 이내로 보장됩니다. d(벡터 차원)가 클수록 내적 왜곡율이 더 낮아집니다.

| 비트폭 (b) | MSE 왜곡율 (D_mse) | 내적 왜곡율 (D_prod/d) |
|:-----------|:--------------------:|:----------------------:|
| 1-bit | 0.36 | 1.57/d |
| 2-bit | 0.117 | 0.56/d |
| 3-bit | 0.03 | 0.18/d |
| 4-bit | 0.009 | 0.047/d |

---

### 6-4. SWA KV 캐시 압축 적용

TurboQuant의 가장 큰 가치 창출 포인트는 **KV 캐시 압축**입니다. 24-layer 샌드위치 구조에서 Layer 9-16의 Sliding Window Attention(SWA)에서 생성되는 Key/Value 텐서를 TurboQuant로 실시간 압축합니다.

| 레이어 | 타입 | KV 캐시 | TurboQuant 적용 |
|:------:|:-----|:-------:|:----------------:|
| 1-8 | Pure SSM (Axon-SSM) | 없음 | 불필요 |
| 9-16 | SWA + MoE | 있음 | **적용 (주요 대상)** |
| 17-24 | SSM + MoE | 없음 | 불필요 |

**TurboQuant 설정별 효과:**
- **3-bit 설정**: 정확도 손실 제로(Zero quality loss), KV 캐시 **6x+ 메모리 절약**
- **4-bit 설정**: Attention logit 계산 속도 **최대 8x 향상**

**65,536 토큰 컨텍스트 기대 효과:**

| 메트릭 | 기존 FP16 KV | TurboQuant 3-bit | 개선 |
|:-------|:------------:|:----------------:|:-----|
| KV 캐시 메모리 | ~4.8 GB | ~0.8 GB | 6x 절감 |
| 64K 컨텍스트 처리 | OOM 가능성 | 안정 구동 | 메모리 한계 극복 |
| Perplexity 변화 | 기준 | < 0.3 | 무손실 수준 |

```python
def enable_turboquant_kv(model, bits: int = 3):
    """SWA 레이어(Layer 9-16)에 TurboQuant KV 캐시 압축 활성화"""
    for i in range(8, 16):  # SWA layers only
        layer = getattr(model, f"layer_{i}")
        if hasattr(layer, "attention"):
            layer.attention.enable_turboquant(bits=bits)
    return model

def disable_turboquant_kv(model):
    """TurboQuant 비활성화 (짧은 컨텍스트에서 메모리 절약)"""
    for i in range(8, 16):
        layer = getattr(model, f"layer_{i}")
        if hasattr(layer, "attention"):
            layer.attention.disable_turboquant()
    return model
```

**TurboQuant 필요성 판단:**

| 컨텍스트 | Q4만 | Q4 + TurboQuant | TQ 필요성 |
|:--------:|:----:|:----------------:|:----------:|
| 4K | ~2.5 GB | ~2.3 GB | 불필요 |
| 16K | ~3.3 GB | ~2.4 GB | 불필요 |
| 32K | ~4.3 GB | ~2.6 GB | 권장 |
| 64K | ~6.3 GB | ~2.9 GB | **필수** |

---

### 6-5. MLX 구현 계획

```
src/bit_axon/quantization/
├── nf4.py              # 기존 NF4 (유지)
├── ternary.py          # 기존 BitNet 1.58-bit (유지)
└── turboquant.py       # 신규: TurboQuant 구현
    ├── polarquant.py       # Stage 1: Random Rotation + 스칼라 양자화
    ├── qjl.py              # Stage 2: 1-bit 잔차 보정
    └── codebooks.py        # 비트별 최적 코드북 사전 계산
```

**구현 시 고려사항:**
- `mx.compile` 호환성: Random Rotation 행렬은 컴파일 캐시 외부에 고정하여 재컴파일 최소화
- Metal 커널 최적화: Random Rotation 행렬 곱셈, 스칼라 양자화 lookup, QJL sign 연산을 Metal 커널로 통합
- 통합 메모리 활용: 코드북(lookup table)은 한 번만 로드, CPU/GPU/NPU가 공유 접근
- 발열 영향: TurboQuant 연산이 Smart Cooling Scheduler와 어떻게 상호작용하는지 측정

#### 기존 오픈소스 구현체 참조

| 리포지토리 | 특징 | 참고 포인트 |
|:-----------|:-----|:------------|
| [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) | Metal fused kernels, 4.6x compression | Metal kernel design, M4 benchmarks |
| [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) | KV cache 전용 | KV cache integration pattern |
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | PyTorch (714 stars) | Accurate algorithm implementation, HuggingFace compatible |

---

### 6-6. 개발 로드맵 통합 (TQ-T1 ~ TQ-T7)

TurboQuant는 기존 양자화 모듈에 병렬 트랙으로 개발됩니다. 각 태스크의 상태는 모두 **진행 전(Pending)**입니다.

| 태스크 | 설명 | 권장 시기 | 상태 |
|:-------|:-----|:----------|:-----:|
| TQ-T1 | 알고리즘 연구 및 Bit-Axon 적용성 분석 | Phase 1 (Week 1-4) | Pending |
| TQ-T2 | MLX 코어 구현 (PolarQuant + QJL) | Phase 1-2 경계 (Week 4-6) | Pending |
| TQ-T3 | SWA 레이어 KV 캐시 TurboQuant 압축 적용 | Phase 3 (Week 9-14) | Pending |
| TQ-T4 | TurboQuant vs NF4 가중치 양자화 벤치마크 | Phase 2 (Week 5-8) | Pending |
| TQ-T5 | TurboQuant Metal 커널 최적화 (Apple Silicon) | Phase 3 (Week 9-14) | Pending |
| TQ-T6 | 저사양 GPU 환경 TurboQuant 통합 | Phase 8+ (미래 확장) | Future |
| TQ-T7 | TurboQuant 통합 벤치마크 및 문서화 | Phase 4 (Week 15-18) | Pending |

#### 참고 문헌

**논문:**
- **TurboQuant**: Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2026). "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." *ICLR 2026*. [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant**: Zandieh, A. et al. (2026). *AISTATS 2026*. [arxiv.org/abs/2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL**: Zandieh, A. et al. (2024). "Quantized Johnson-Lindenstrauss." *AAAI 2025*. [arxiv.org/abs/2406.03482](https://arxiv.org/abs/2406.03482)

**Google Research Blog:**
- [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (2026-03-24)

---

## 7. 개발 로드맵

> **모든 태스크 상태 = Pending (진행 전)**. 완료된 작업은 없습니다.

### Phase 1: Core Primitives (Week 1-4)

**기간:** 1주 차 ~ 4주 차 (1개월)
**목표:** Bit-Axon의 3대 아키텍처를 Apple MLX 프레임워크의 저수준 API로 구현

- **Week 1-2: 선형(Linear) SSM 커널 최적화**
  - `Axon-SSM` 클래스(Mamba-3 기반)를 MLX로 구현하여 Transformer Attention 블록 대체
  - MLX의 `mlx.core.scan` (누적 연산) 함수로 시퀀스 병렬 처리 로직 작성
  - **상태**: Pending

- **Week 3: 희소(Sparse) MoE 라우팅 구현**
  - `Shared-Expert MoE` 라우팅 로직: 전체 파라미터를 메모리에 올리고 연산 시 top-2 전문가만 활성화
  - MLX 배열 마스킹(Masking) 기법으로 GPU 병목 방지
  - **상태**: Pending

- **Week 4: 4-bit 양자화 및 Bit-DoRA 뼈대 구축**
  - 모델 가중치를 4-bit NormalFloat로 압축하는 로직 구현
  - DoRA(Weight-Decomposed LoRA)의 '크기(Magnitude)' 백터와 '방향(Direction)' 행렬 분리 초기화
  - **상태**: Pending

---

### Phase 2: Architecture Synthesis (Week 5-8)

**기간:** 5주 차 ~ 8주 차 (2개월)
**목표:** 3B 파라미터급 오픈소스 모델을 Bit-Axon 하이브리드 구조로 변환 및 벤치마크

**목표 지표:** 추론 속도 50-60 tokens/sec, 메모리 2.5GB 이하, 전력 5W 이하

#### Milestone 1: Weight Porting (Week 5-6)

- **T1.1: 가중치 포팅 스크립트 테스트** [Pending]
  - Qwen2.5-0.5B로 포팅 스크립트 테스트, 24 레이어 매핑 검증, 텐서 형상 확인
  - 성공 기준: 에러 없이 실행, 모든 필수 텐서 생성, 형상 일치

- **T1.2: 매핑 품질 검증** [Pending]
  - 포팅된 가중치를 BitAxonModel에 로드, 테스트셋에서 perplexity 평가
  - 성공 기준: 소스 모델 대비 perplexity delta < 20%, NaN/Inf 없음

- **T1.3: 타겟 모델 매핑 수정** [Pending]
  - Qwen2.5-3B 아키텍처 분석, RoPE 위치 임베딩 매핑, 정규화 레이어 처리
  - 성공 기준: Qwen2.5-3B 매핑 동작, RoPE/정규화 레이어 정확한 처리

- **T1.4: 가중치 검증 도구 생성** [Pending]
  - 소스 vs 타겟 가중치 비교 스크립트, 가중치 분포 시각화, 아웃라이어 탐지

#### Milestone 2: MLX Compilation & Memory Optimization (Week 7)

- **T2.1: @mx.compile JIT 통합** [Pending]
  - 모델 레벨 `mx.compile` 적용 (레이어 레벨 컴파일은 동작하지 않음)
  - MoE numpy interop를 pure MLX로 교체하는 패치 필요
  - 성공 기준: 에러 없이 컴파일, 2x+ 속도 향상, 컴파일 시간 < 10초

- **T2.2: 통합 메모리 Zero-Copy 프로파일링** [Pending]
  - `mx.metal.get_active_memory()`, `mx.metal.get_peak_memory()` 사용
  - 불필요한 메모리 복사 식별 및 제거
  - 성공 기준: 불필요한 복사 0개, 통합 메모리 이점 확인

- **T2.3: 데이터 전송 최적화** [Pending]
  - CPU/GPU 전송 최소화, `.numpy()` 변환 없이 `mx.array` 직접 사용
  - 성공 기준: 추론 시간의 10% 미만이 데이터 전송

- **T2.4: 컴파일 캐시 관리** [Pending]
  - 컴파일된 커널 디스크 저장, 모델 아키텍처 변경 시 캐시 무효화

#### Milestone 3: Bare-Metal Benchmarking (Week 8)

- **T3.1: 전체 모델 벤치마크** [Pending]
  - 1K, 4K, 16K, 32K, 64K 시퀀스 길이에서 테스트
  - 성공 기준: 50-60 tok/s (4K), 메모리 <= 2.5GB, 전력 <= 5W

- **T3.2: TurboQuant KV 프로파일링** [Pending]
  - SWA 레이어(9-16)에 TurboQuant 적용, 압축률 및 품질 측정
  - 성공 기준: KV 캐시 6x+ 압축, perplexity 증가 < 5%

- **T3.3: 열 동작 분석** [Pending]
  - 지속 추론 시 SoC 온도 모니터링, 쓰로틀링 이벤트 측정
  - 성공 기준: SoC 온도 < 85C, 10분간 쓰로틀링 없음

- **T3.4: 최적화 보고서** [Pending]
  - 벤치마크 결과 문서화, 목표 지표 대비 비교, Phase 3 우선순위 권장

---

### Phase 3: Thermal-Aware Training (Week 9-14)

**기간:** 9주 차 ~ 14주 차 (1.5개월)
**목표:** 팬리스 구조의 한계를 극복하는 '발열 제어형' SFT 학습 파이프라인 완성

- **Week 9: 발열 모니터링 데몬 개발** [Pending]
  - macOS `powermetrics` 또는 IOKit을 Python(subprocess)으로 호출
  - SoC 온도 및 전력 소모량을 1초 단위로 실시간 추적

- **Week 10-11: 스마트 쿨링 스케줄러 결합** [Pending]
  - 온도 데이터와 MLX 학습 루프 연동
  - 로직: 85C 초과 시 0.5초 대기, 95C 도달 시 학습 정지, 75C 이하 시 최고 속도

- **Week 12-14: 도메인 특화 학습 (본격 SFT)** [Pending]
  - 고품질 한국어 데이터셋 준비 (Packing 기법 적용)
  - Bit-DoRA 적용하여 전체 가중치의 ~1%만 학습
  - MacBook Air 전원 연결 상태로 밤새 무인 학습(Overnight SFT) 테스트

**검증 기준:**
- Loss 안정적 감소 (1000 스텝 이상)
- 학습 중 OOM 발생 없음
- SoC 온도 < 85C 유지

---

### Phase 4: Alignment & Merging (Week 15-18)

**기간:** 15주 차 ~ 18주 차 (1개월)
**목표:** 환각(Hallucination)을 줄이고 최종 배포를 위한 단일 파일로 압축

- **Week 15-16: ORPO 적용** [Pending]
  - RLHF/DPO는 참조 모델을 메모리에 올려야 하므로 16GB에서 OOM 가능성이 높음
  - 참조 모델 없이 SFT와 선호도 정렬을 동시에 수행하는 ORPO를 MLX에 포팅

- **Week 17-18: 어댑터 병합 및 최종 양자화** [Pending]
  - 4-bit DoRA 가중치를 베이스 모델에 병합
  - BF16으로 일시 업스케일 후 다시 4-bit(GGUF/MLX 포맷)로 다운스케일
  - 최종 산출물: `bit-axon-3b-q4.safetensors` (단일 파일)

---

### Phase 5: App & CLI Release (Week 19-24)

**기간:** 19주 차 ~ 24주 차 (1.5개월)
**목표:** 터미널 스크립트를 넘어 네이티브 macOS 애플리케이션으로 승화

- **Week 19-20: CLI 도구 개발** [Pending]
  - `bit-axon run "안녕?"`, `bit-axon train my_data.json`, `bit-axon quantize` 명령어
  - Rust 또는 Python 기반 터미널 유틸리티

- **Week 21-23: SwiftUI 네이티브 앱** [Pending]
  - Apple `MLX-Swift` 라이브러리로 macOS 전용 GUI 앱
  - 채팅 인터페이스 (실시간 토큰 속도, SoC 온도 표시)
  - 원클릭 파인튜닝 ("데이터를 드래그 앤 드롭하세요")

- **Week 24: 최종 릴리즈** [Pending]
  - GitHub `Project Bit-Axon` 리포지토리 구성
  - Hugging Face `mlx-community`에 완성된 한국어 3B 하이브리드 모델 업로드

---

### 일정 요약

| 단계 | 기간 | 누적 | 산출물 |
|:-----|:----:|:----:|:--------|
| Phase 1: Core Primitives | Week 1-4 | 4주 | SSM, MoE, DoRA primitives |
| Phase 2: Architecture Synthesis | Week 5-8 | 8주 | 포팅된 모델, Q4 양자화, 벤치마크 |
| Phase 3: Thermal-Aware Training | Week 9-14 | 14주 | QLoRA 학습된 어댑터 |
| Phase 4: Alignment & Merging | Week 15-18 | 18주 | 병합된 Q4 모델 |
| Phase 5: App & CLI Release | Week 19-24 | 24주 | CLI, 앱, 오픈소스 배포 |

총 예상 기간: **약 6개월**

---

## 8. 핵심 기술 발견

> 이 섹션은 이전 개발 과정에서 발견된 기술적 사실과 회피해야 할 함정(Pitfalls)을 문서화합니다. 완료 상태가 아닌 **지침(Guidelines)**으로 취급하세요.

### 8-1. MLX 플랫폼 발견

1. **Layer-level 컴파일은 동작하지 않습니다.** 개별 레이어 `forward` 메서드를 `mx.compile`로 감싸면 모듈 참조 문제가 발생합니다. MLX가 함수를 트레이스하지만 레이어 레벨에서 `self` 속성을 올바르게 해결할 수 없습니다. **대안: 모델 레벨 컴파일 사용.**

2. **`shapeless=True`는 MLX <= 0.31에서 matmul에 대해 깨져 있습니다.** 출력 형상이 첫 번째 트레이스에서 캐시되어 이후 호출에서 잘못된 결과를 만듭니다. **대안: Shape-dependent 컴파일 사용, 입력 형상 변경 시 지연 재컴파일 허용.**

3. **MoE numpy interop가 `mx.compile`을 깨뜨립니다.** `SharedExpertMoE.forward`가 `np.array()`, `np.where()` 등을 사용하면 내부적으로 `mx.eval()`이 호출되어 lazy tracing이 깨집니다. **대안: 모든 numpy 연산을 pure MLX dense dispatch로 교체한 `_pure_mlx_moe_forward()` 함수를 컴파일 전에 monkey-patch.**

4. **MLX `nn.Module`은 `mx.array` 속성을 자동으로 파라미터 트리에 등록합니다.** 별도의 `nn.Parameter` 클래스가 필요 없습니다. 모듈 속성으로 할당된 모든 `mx.array`가 `parameters()`에 나타납니다.

5. **MLX `parameters()`는 중첩 딕셔너리를 반환합니다.** 평탄화하려면 `mx.tree_flatten(parameters())`를 사용하여 `(key_path, array)` 쌍을 얻으세요.

6. **MLX에 내장된 `nn.QuantizedLinear`가 Metal 커널을 사용합니다.** `nn.quantize()`를 사용하면 커스텀 NF4 dequantization + matmul보다 훨씬 빠릅니다. 양자화된 matmul이 단일 fused Metal 커널로 실행됩니다.

7. **Python `for` 루프는 `mx.compile` 트레이싱과 호환됩니다.** 고정 바운드(예: `n_experts` 반복)가 있는 루프는 트레이서가 구체적인 값으로 실행하므로 올바르게 트레이스됩니다.

### 8-2. 회피해야 할 함정

#### 아키텍처 관련

- **중복 파라미터 키 버그:** `BitAxonModel.__init__`에서 `setattr(self, f"layer_{i}", layer)`와 `self.layers.append(layer)`를 동시에 사용하면 `parameters()`가 동일 레이어를 두 번 반환합니다. **해결: `self.layers` 리스트를 제거하고 `getattr`만 사용.**

- **Expert intermediate size 누락:** `ModelConfig`에 `expert_intermediate_size`가 없으면 `AxonSSMMoELayer`가 값을 받을 수 없습니다. `AxonSWAMoELayer`에도 전파해야 합니다. **해결: config에 `expert_intermediate_size: int = 4096` 추가.**

- **차원 프로젝션 레이어:** Qwen2.5-3B(hidden_size=2048)를 Bit-Axon(d_model=2560)으로 포팅할 때 차원 차이를 처리해야 합니다. `d_source_model=2048` 설정 시 `input_proj`와 `output_proj` 레이어가 생성됩니다. Embedding과 lm_head는 2048 차원에서 동작합니다.

#### 가중치 포팅 관련

- **"Llama-4-Mini"는 존재하지 않습니다.** Meta는 Llama-4 Scout(109B)와 Maverick(402B)을 출시했습니다. 소형 Llama-4 변형은 없습니다. Qwen2.5-3B를 대안으로 사용합니다.

- **Qwen2.5-3B → Bit-Axon 매핑:** Qwen의 36 transformer 레이어 중 처음 24개를 1:1 매핑하고, 레이어 24-35는 폐기합니다. SSM 파라미터는 무작위 초기화됩니다. MoE expert 가중치: shared expert는 Qwen의 MLP에서 가져오고, expert 0은 복사본, expert 1-7은 perturbed copy입니다. **총 517 타겟 키 생성.**

#### MLX 컴파일 관련

- **`CompilationConfig`, `CompilationCache`, `CompilationMetrics` 클래스는 불필요합니다.** 컴파일은 단일 함수 호출이며, 설정 가능한 하위 시스템이 아닙니다. MLX가 커널 캐싱을 내부적으로 처리합니다.

- **지속적 커널 캐싱(Persistent kernel caching)은 불필요합니다.** MLX가 이를 자체적으로 처리하므로 커스텀 캐시 레이어는 값이 없습니다.

### 8-3. 모델 차원 참조

| Parameter | Qwen2.5-3B (Source) | Bit-Axon (Target) |
|:----------|:---------------------|:------------------|
| hidden_size / d_model | 2048 | 2560 |
| num_layers | 36 | 24 (first 24 mapped, rest discarded) |
| intermediate_size | 11008 | 4096 (expert) / 8192 (shared) |
| vocab_size | 151936 | 32000 (projected via d_source_model=2048) |
| attention | GQA, 2 KV heads | SWA, 32 heads, window=4096 |
| n_experts / top_k | N/A (dense MLP) | 8 experts, top-2 + 1 shared |

**차원 브릿지:** `d_source_model=2048`가 `input_proj`와 `output_proj` 선형 레이어를 통해 `d_model=2560`으로 프로젝션합니다. Embedding과 lm_head는 2048 차원에서 동작합니다.

---

## 9. 미래 확장

> 이 섹션의 모든 내용은 **미래 고려 사항(Future Consideration)**이며, 현재 계획의 일부가 아닙니다.

### 9-1. 멀티 백엔드 전략

Apple 기기 전용인 MLX만으로는 범용 확장이 불가능합니다. 엔진 백엔드를 여러 환경에 맞게 스왑할 수 있는 멀티 백엔드 구조를 채택합니다.

| 타겟 하드웨어 | 엔진 백엔드 | 핵심 최적화 |
|:-------------|:-----------|:-----------|
| MacBook Air (M4) | Apple MLX | 통합 메모리 Zero-Copy, Smart Cooling |
| Intel/AMD 저사양 CPU | llama.cpp (GGML) | BitNet(1.58-bit) 순수 덧셈 연산, AVX-512 SIMD |
| 저사양 GPU 서버 (Nvidia) | vLLM / SGLang (PyTorch) | AWQ 3-bit 양자화, CUDA Graph, GaLore SFT |

**Bit-Axon 아키텍처의 저사양 CPU 적합성:**
- **Linear (SSM):** 저사양 PC의 8GB RAM에서도 KV-Cache로 인한 메모리 폭발 방지
- **Sparse (MoE):** CPU는 MatMul에 취약하지만 분기 처리(Routing)에는 강함. MoE 라우팅과 CPU 아키텍처 찰떡궁합
- **Quantized (1.58-bit Ternary):** 가중치를 {-1, 0, 1}로 쪼개면 부동소수점 곱셈이 순수 덧셈으로 변환. 구형 CPU의 AVX2/AVX-512로도 빠른 추론

### 9-2. 크로스 플랫폼 포팅

#### Phase 6: Universal Porting (7~8개월 차) — 미래 계획

- `mlx-to-gguf` 컨버터: Bit-Axon 하이브리드(Mamba+MoE) 아키텍처를 llama.cpp에서 인식할 수 있는 GGUF 포맷으로 매핑
- CPU 추론 최적화: GGML 백엔드에 1.58-bit(BitNet) 덧셈 연산 커널 추가, 구형 Intel i5/i7에서 30+ tok/s 튜닝
- Intel OpenVINO 및 ONNX 포맷 변환 스크립트

#### Phase 7: Omni-App Development (9개월 차) — 미래 계획

- SwiftUI 기반 Mac 전용 앱을 Tauri (Rust + React/Vue) 기반 데스크톱 앱으로 재작성 (Electron보다 메모리 1/10)
- 하드웨어 자동 감지: Mac M4 → MLX, Intel CPU → llama.cpp/OpenVINO
- "Bit-Axon Omni"로 GitHub 통합 릴리즈

### 9-3. 저사양 CPU용 마이크로 어댑터

- 전체 파라미터의 **0.01% 미만**만 업데이트하는 극소형 LoRA(VeRA - Vector-based Random Matrix Adaptation) 도입
- llama.cpp의 CPU 학습 기능으로 텍스트 10~50건의 사용자 페르소나/말투를 수십 분 내에 학습

### 9-4. Bit-Axon Swarm (이기종 분산 클러스터링)

#### Phase 8: 저사양 GPU 훈련 파이프라인 (10~11개월 차) — 미래 계획

- Unsloth 최적화 커널 기반 PyTorch Training Script (GaLore + DoRA)
- 구형 GPU(Turing, Ampere)에서 MoE 라우팅 병목 없이 동작하는 FlashInfer 기반 CUDA 커널
- AWQ 및 GPTQ 포맷 자동 양자화 모듈

**저사양 GPU 학습 기법:**
- **GaLore (Gradient Low-Rank Projection):** 기울기 자체를 저차원으로 투영하여 8GB VRAM GPU로 3B 모델 Full FT 수준 성능
- **LISA (Layer-wise Importance Sampled AdamW):** 한 번에 1개 레이어만 메모리에 올려 학습하고 스왑. VRAM 극단적 절감

#### Phase 9: Bit-Axon Swarm (12개월 차) — 미래 계획

- MacBook Air M4(UI/마스터), Windows 노트북(CPU 워커), 저가형 데스크탑(GTX 1660 워커)을 동일 네트워크에 연결
- **Swarm 모드 분업:**
  - Prefill(프롬프트 처리) → GPU 서버 담당
  - Decode(토큰 생성) + MoE 라우팅 → M4 Mac 담당
  - RAG(지식 검색) → CPU 노트북 담당
- 단일 기기로 불가능한 초거대 문맥 처리와 7B+ 모델 구동을 저사양 기기의 연합으로 극복

---

## 10. 위험 평가

8GB available 제약은 기존 16GB 계획 대비 추가적인 위험을 초래합니다. FP16 추론의 빡빡함과 TurboQuant 통합의 복잡성이 핵심 리스크입니다.

### 고위험

| 위험 | 확률 | 영향 | 완화 전략 |
|:-----|:----:|:----:|:---------|
| FP16 추론 시 8GB 초과 | 높음 | 중간 | Q4를 기본으로 사용. FP16은 디버깅 전용 |
| TurboQuant MLX 구현 복잡도 | 높음 | 높음 | 기존 구현체 참조 (arozanov/turboquant-mlx). 4K-16K에서는 TQ 없이 구동 |
| MoE + mx.compile 호환성 | 중간 | 높음 | Pure-MLX MoE forward 패치 적용 (numpy interop 제거) |
| QLoRA 학습 중 OOM | 낮음 | 높음 | B=1, T=2048로 안전 설정. Gradient checkpointing 필수 |
| 가중치 포팅 모델 품질 저하 | 중간 | 높음 | Perplexity 검증. 목표 < 20% delta. SSM 파라미터 무작위 초기화 주의 |
| 성능 목표 미달 (50-60 tok/s) | 중간 | 높음 | 광범위한 프로파일링, 핫패스 최적화, 아키텍처 수정 검토 |

### 중위험

| 위험 | 확률 | 영향 | 완화 전략 |
|:-----|:----:|:----:|:---------|
| 64K 컨텍스트 메모리 초과 | 중간 | 중간 | TurboQuant 필수 적용. TQ 없이 64K 구동 시도 금지 |
| 열 스로틀링 (3B 모델) | 중간 | 중간 | MoE 희소성으로 연산량 감소. Smart Cooling Scheduler 필수 |
| 1.58-bit Ternary 안정성 | 중간 | 낮음 | 실험적 기능으로 취급. Q4를 기본으로 유지 |
| 컴파일 재컴파일 오버헤드 | 중간 | 중간 | shapeless 대신 shape-dependent 사용. 프리필+디코드 2회 컴파일 허용 |
| MLX 컴파일 에러 (SSM 연산) | 낮음 | 높음 | 점진적 컴파일, 문제 연산 격리, 언컴파일 폴백 |

### 저위험

| 위험 | 확률 | 영향 | 완화 전략 |
|:-----|:----:|:----:|:---------|
| MLX 프레임워크 버그 | 낮음 | 중간 | 최신 안정 버전 사용, Apple MLX GitHub 이슈 트래킹 |
| Qwen2.5-3B 가중치 형식 변경 | 낮음 | 낮음 | 포팅 스크립트 버전 관리, 여러 모델 소스 지원 |
