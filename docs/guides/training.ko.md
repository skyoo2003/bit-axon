# 학습 가이드

Apple Silicon에서 온도 인식 QLoRA를 사용해 Bit-Axon을 직접 학습시킵니다. 전체 파이프라인(SFT, ORPO 정렬, 체크포인팅)은 통합 메모리 16GB의 팬리스 MacBook Air M4에서 로컬로 실행됩니다.

---

## 빠른 시작

```bash
bit-axon train data.json \
    --model-weights ./model \
    --tokenizer Qwen/Qwen2.5-3B
```

이 명령은 4-bit 양자화된 모델을 로드하고, DoRA 어댑터(랭크 8)를 적용하며, 데이터셋을 토큰화한 뒤 온도 모니터링이 활성화된 상태로 학습 루프를 시작합니다.

---

## 개요

Bit-Axon은 두 가지 학습 모드를 지원합니다.

| 모드 | 목적 | Trainer 클래스 | 손실 함수 |
|------|---------|---------------|------|
| **SFT** | 지도 파인튜닝 | `Trainer` | 어시스턴트 토큰에 대한 교차 엔트로피 |
| **ORPO** | 선호도 정렬 | `ORPOTrainer` | NLL + 승산비 페널티 |

두 모드 모두 **4-bit 양자화된 베이스 가중치**에 학습 가능한 **LoRA** 또는 **DoRA** 어댑터를 사용합니다. 어댑터 파라미터에만 기울기가 전파되며, 베이스 모델은 NF4 정밀도로 고정되어 전체 메모리를 3.7GB 이하로 유지합니다.

### 학습 모듈

모든 학습 로직은 `src/bit_axon/training/`에 있습니다.

| 모듈 | 용도 |
|--------|---------|
| `config.py` | `TrainingConfig` 데이터클래스: 모든 하이퍼파라미터 |
| `trainer.py` | `Trainer` 클래스: SFT 학습 루프 |
| `lora.py` | `LoRALinear`, `apply_lora_to_model()` |
| `dora.py` | `DoRALinear`: 가중치 분해 LoRA |
| `data.py` | `SFTDataset`, `AlpacaDataset`, `ORPODataset` |
| `cooling.py` | `CoolingScheduler`, `ThermalPolicy` |
| `orpo_trainer.py` | `ORPOTrainer`: 선호도 정렬 루프 |
| `orpo_loss.py` | `compute_orpo_loss()`, `orpo_loss()` |
| `packing.py` | `SequencePacker`: 예제를 고정 길이 시퀀스로 연결 |
| `checkpoint.py` | `save_checkpoint()`, `load_checkpoint()`, `get_latest_checkpoint()` |
| `scheduler.py` | 선형 웜업이 포함된 코사인 감쇠 |
| `collate.py` | `iterate_batches()`, `BatchCollator` |
| `merging.py` | `merge_adapters()`, `load_and_merge()` |

---

## 데이터셋 형식

### SFT: 채팅 메시지 (JSONL)

각 줄은 OpenAI 스타일 messages 형식의 단일 대화입니다.

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain quantum entanglement."}, {"role": "assistant", "content": "Quantum entanglement is a phenomenon..."}]}
```

지원되는 역할: `system`, `user`, `assistant`. `SFTDataset`은 Qwen2.5 채팅 템플릿을 적용하고 이진 loss 마스크를 계산합니다. 기울기는 어시스턴트 응답 토큰에만 전파됩니다. System 및 user 토큰은 무시됩니다.

### SFT: Alpaca 형식

`AlpacaDataset` 클래스는 표준 Alpaca 명령 데이터를 받아 내부적으로 변환합니다.

```json
{"instruction": "Summarize the following text.", "input": "Long text here...", "output": "Short summary."}
```

`input` 필드는 선택사항입니다. 존재하는 경우 지시어에 두 개의 빈 줄을 추가해 결합합니다.

### ORPO: 선호도 쌍

각 예제는 프롬프트, 선택된 응답, 거부된 응답을 포함합니다.

```json
{"prompt": [{"role": "user", "content": "Write a haiku about debugging."}], "chosen": [{"role": "assistant", "content": "Silent cursor blinks / Stack trace scrolls through the night / Bug found at line three"}], "rejected": [{"role": "assistant", "content": "Debugging is when you fix bugs in your code."}]}
```

더 간단한 문자열 형식도 지원됩니다.

```json
{"prompt": "Write a haiku about debugging.", "chosen": "Silent cursor blinks...", "rejected": "Debugging is when you fix bugs..."}
```

!!! tip "데이터셋 변환"
    HuggingFace 데이터셋을 JSONL로 변환하려면 `bit-axon prepare`를 사용하세요:
    ```bash
    bit-axon prepare HuggingFaceH4/ultrachat --format messages --output train.jsonl --split train
    ```

---

## LoRA 및 DoRA 어댑터

### LoRA (Low-Rank Adaptation)

대상 선형 레이어에 학습 가능한 랭크 `r` 분해를 추가합니다. 베이스 가중치 `W`는 고정됩니다.

```
output = W·x + scale · (dropout(x) · A) · B
```

- `A` 형태: `(input_dims, r)`: 균일 노이즈로 초기화
- `B` 형태: `(r, output_dims)`: 0으로 초기화 (어댑터는 항등 변환으로 시작)
- `scale` 기본값: 20.0

`LoRALinear.from_base(linear, r=8, scale=20.0)` 팩토리는 기존 `nn.Linear` 또는 `nn.QuantizedLinear` 레이어를 래핑하며 기존 가중치를 보존합니다.

### DoRA (Weight-Decomposed LoRA)

기본적으로 활성화됩니다. LoRA를 확장하여 원래 가중치의 크기를 보존합니다.

```
adapted_W = W + scale · B^T · A^T
output = (m / ||adapted_W||) · (W·x + scale · (dropout(x) · A) · B)
```

크기 벡터 `m`은 고정된 베이스 가중치의 행별 L2 노름을 저장합니다 (초기화 시 한 번 계산). 순전파에서 출력은 재정규화되어 적응된 가중치가 원래 크기 특성을 보존합니다. 이는 일반 LoRA보다 종종 더 좋은 결과를 만들어냅니다, 특히 세밀한 출력 보정이 필요한 작업에서.

!!! important "DoRA가 기본값입니다"
    `TrainingConfig`에서 `use_dora=True`입니다. 일반 LoRA로 되돌리려면 `--no-dora`를 전달하세요.

### 대상 레이어

어댑터는 다음 선형 레이어 유형에 적용됩니다.

| 대상 레이어 | 위치 |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention Q/K/V/O 프로젝션 |
| `in_proj`, `out_proj` | 결합된 어텐션 프로젝션 |
| `gate_proj`, `up_proj`, `down_proj` | Feed-forward (expert) 프로젝션 |
| `input_proj`, `output_proj` | SSM 입력/출력 프로젝션 |

### 제외 레이어

다음 레이어는 대상 매칭에 관계없이 **절대** 어댑터가 적용되지 않습니다.

| 제외 항목 | 매칭 유형 | 이유 |
|---|---|---|
| `switch_mlp` | 경로 포함 | MoE 라우터 내부 |
| `lm_head` | 경로 포함 | 출력 헤드: 임베딩과 결합(tied)됨 |
| `gate` | 이름 일치 | MoE 게이팅 |
| `shared_expert_gate` | 이름 일치 | 공유 expert 게이팅 |
| `x_proj` | 이름 일치 | SSM 전용 프로젝션 |
| `dt_proj` | 이름 일치 | SSM 델타 프로젝션 |

제외 로직은 `lora.py`에 있습니다.

```python
LORA_EXCLUDED_PATHS = ("switch_mlp", "lm_head")
LORA_EXCLUDED_NAMES = ("x_proj", "dt_proj", "gate", "shared_expert_gate")
```

### 어댑터 적용

```python
from bit_axon.training.lora import apply_lora_to_model

# DoRA 적용 (기본값): 래핑된 레이어 경로 목록 반환
wrapped = apply_lora_to_model(model, rank=8, dropout=0.0, scale=20.0, use_dora=True)

# 일반 LoRA 적용
wrapped = apply_lora_to_model(model, rank=8, dropout=0.0, scale=20.0, use_dora=False)
```

### 어댑터 퓨징

학습 후 어댑터를 베이스 가중치에 다시 퓨징합니다.

```python
from bit_axon.training.merging import merge_adapters

# LoRA 퓨즈: W_fused = W + scale · B^T · A^T
# DoRA 퓨즈: W_fused = (m / ||W + delta||) · (W + delta)
merge_adapters(model)
```

---

## 시퀀스 패킹

GPU 활용률을 극대화하기 위해 `SequencePacker`는 여러 짧은 예제를 `max_seq_len` 토큰의 고정 길이 시퀀스로 연결합니다.

```
Example 1 tokens | EOS | Example 2 tokens | EOS | Example 3 tokens | PAD
[   loss_mask=1  |  0  | loss_mask=1      |  0  | loss_mask=1      |  0 ]
```

이진 loss 마스크는 다음을 보장합니다.

- 구분자 EOS 토큰 (ID 151645)은 loss에 기여하지 않음
- 패딩 토큰은 `-100` 무시 인덱스로 마스킹됨
- 원래 예제의 응답 토큰만 기울기를 생성함

패킹은 `iterate_batches()` 내부에서 자동으로 실행됩니다. 별도의 설정이 필요 없습니다.

```python
from bit_axon.training.packing import SequencePacker

packer = SequencePacker(max_seq_len=2048, eos_token_id=151645)

for token_ids, loss_mask in dataset:
    packed_batches = packer.add_example(token_ids, loss_mask)
    for batch in packed_batches:
        process(batch)  # PackedBatch(token_ids, loss_mask)

final = packer.flush()  # 남은 토큰을 max_seq_len으로 패딩
```

!!! info "ORPO는 패킹을 사용하지 않습니다"
    선호도 쌍은 `iterate_orpo_batches()`를 통해 그대로 유지됩니다. 각 chosen/rejected 쌍이 단위로 처리됩니다.

---

## 온도 인식 학습

팬리스 MacBook에서 학습하면 지속적인 열이 발생합니다. `CoolingScheduler`는 macOS `powermetrics`를 통해 SoC 온도를 읽고, 매 학습 스텝 전에 3단계 온도 정책을 적용합니다.

### 온도 단계

| 단계 | 온도 | 동작 |
|------|-------------|--------|
| 정상 | < 75°C | 정속 학습 |
| 따뜻함 | 75–85°C | `should_reduce_batch()`가 `True` 반환 (배치 축소 신호) |
| 뜨거움 | ≥ 85°C | 학습 **일시 정지**: 온도가 낮아질 때까지 0.5초 간격으로 대기 |
| 임계 | ≥ 95°C | `ThermalShutdownError`와 함께 학습 **즉시 중지** |

### Python API

```python
from bit_axon.training.cooling import CoolingScheduler, ThermalPolicy, ThermalShutdownError

policy = ThermalPolicy(
    max_speed_temp=75.0,   # 배치 축소 구간 시작
    pause_temp=85.0,       # 이 온도 이상에서 학습 일시 정지
    stop_temp=95.0,        # 이 온도 이상에서 학습 중지
    pause_duration=0.5,    # 일시 정지 중 대기 간격 (초)
)

cooling = CoolingScheduler(monitor, policy)

# 각 학습 스텝 전에 호출:
cooling.check_before_step(step)  # 일시 정지 또는 ThermalShutdownError 발생

# 온도 대기로 소요된 총 시간 확인:
print(f"Paused {cooling.total_pause_time:.1f}s for cooling")
```

### CLI 설정

```bash
# 사용자 정의 온도 임계값
bit-axon train data.json --model-weights ./model --tokenizer Qwen/Qwen2.5-3B \
    --temp-pause 80 --temp-stop 90

# 온도 모니터링 비활성화 (활성 냉각이 있는 머신에서만 사용)
bit-axon train data.json --model-weights ./model --tokenizer Qwen/Qwen2.5-3B \
    --no-thermal
```

!!! warning "팬리스 MacBook Air"
    팬이 없는 MacBook Air M4에서 지속적인 학습은 SoC 온도를 90°C 이상으로 올릴 수 있습니다. 기본 임계값(85°C에서 일시 정지, 95°C에서 중지)은 안전한 동작을 위해 보정되어 있습니다. 활성 냉각이 있거나 짧은 테스트를 실행하는 경우가 아니면 온도 모니터링을 비활성화하지 마세요.

---

## ORPO 선호도 최적화

ORPO (Odds Ratio Preference Optimization)는 SFT와 선호도 정렬을 동시에 수행합니다. DPO와 달리 **레퍼런스 모델이 필요 없어** 정렬 중 메모리를 약 50% 절약합니다.

### 손실 함수

ORPO 손실은 두 항을 결합합니다.

```
L_total = L_NLL(chosen) - log σ(β · log_odds_ratio)

where:
  log_odds_ratio = log(p_chosen / (1 - p_chosen)) - log(p_rejected / (1 - p_rejected))
```

- **NLL 손실**: 선택된 응답에 대한 표준 next-token 예측 (SFT 신호)
- **승산비 페널티**: chosen 대 rejected에 대해 더 높은 로그 확률을 향하도록 밀어내는 로그 시그모이드
- **β** (기본값 `0.1`): 선호도 강도를 제어. 값이 높을수록 chosen 응답을 더 강하게 선호

### ORPO 실행

```python
from bit_axon.training.config import TrainingConfig
from bit_axon.training.data import ORPODataset
from bit_axon.training.orpo_trainer import ORPOTrainer

config = TrainingConfig(
    training_mode="orpo",
    beta=0.1,
    max_steps=2000,
)

dataset = ORPODataset("prefs.jsonl", tokenizer, max_seq_len=2048)
trainer = ORPOTrainer(model, config, dataset, cooling_scheduler=cooling)
result = trainer.train()

# Result keys:
# step, loss, grad_norm, chosen_reward, rejected_reward, reward_margin, reward_accuracy
```

`ORPOTrainer`는 배치당 두 번의 순전파(chosen + rejected)를 실행하고, `get_logps()`를 통해 양쪽의 평균 로그 확률을 계산하며, NLL 손실에 승산비 페널티를 결합합니다.

!!! tip "reward margin 모니터링"
    `reward_margin = chosen_reward - rejected_reward`. 증가하는 margin은 모델이 더 나은 응답을 선호하도록 학습하고 있음을 의미합니다. ORPO 학습을 언제 중지할지 결정하는 데 사용하세요.

---

## 학습 설정

### TrainingConfig 데이터클래스

모든 하이퍼파라미터는 `TrainingConfig`에 있습니다.

```python
from bit_axon.training.config import TrainingConfig

config = TrainingConfig(
    # 옵티마이저
    learning_rate=1e-4,       # 웜업 후 최대 LR
    weight_decay=0.01,        # AdamW 가중치 감쇠
    warmup_steps=100,         # 선형 웜업 스텝
    max_steps=10_000,         # 전체 학습 스텝
    max_grad_norm=1.0,        # 기울기 클리핑 임계값
    grad_accum_steps=4,       # 기울기 누적 스텝

    # LoRA / DoRA
    lora_rank=8,              # 저랭크 분해 랭크
    lora_dropout=0.0,         # 어댑터 경로의 dropout
    lora_scale=20.0,          # 어댑터 출력 스케일링
    use_dora=True,            # DoRA 사용 (가중치 분해 LoRA)

    # ORPO 정렬
    beta=0.1,                 # ORPO 선호도 강도
    training_mode="sft",      # "sft" 또는 "orpo"

    # 양자화
    quantize_bits=4,          # 베이스 가중치 비트 폭 (NF4)
    quantize_group_size=64,   # 양자화 그룹 크기

    # 데이터
    batch_size=1,             # 배치당 시퀀스 수
    max_seq_len=2048,         # 패킹 대상 길이

    # 체크포인팅
    save_every=500,           # N 스텝마다 체크포인트 저장
    eval_every=500,           # N 스텝마다 평가
    output_dir="checkpoints", # 체크포인트 디렉토리

    # 온도 임계값 (°C)
    temp_max_speed=75.0,      # 배치 축소 구간
    temp_pause=85.0,          # 학습 일시 정지
    temp_stop=95.0,           # 학습 중지
    temp_poll_interval=1.0,   # 온도 폴링 간격 (초)

    # 기타
    seed=42,                  # 난수 시드
)
```

### 학습률 스케줄

선형 웜업이 포함된 코사인 감쇠:

- 스텝 0~`warmup_steps`: LR이 0에서 `learning_rate`로 선형 증가
- 스텝 `warmup_steps`~`max_steps`: 코사인 감쇠로 `learning_rate`에서 0으로 감소

### 배치 크기와 기울기 누적

`batch_size=1`과 `grad_accum_steps=4`를 사용하면 유효 배치 크기는 4가 됩니다.

```
effective_batch_size = batch_size × grad_accum_steps = 1 × 4 = 4
```

4번의 순전파 동안 기울기가 누적된 후 단일 옵티마이저 업데이트가 이루어집니다. 이로써 스텝당 메모리는 낮게 유지하면서 합리적인 유효 배치 크기를 유지합니다.

### 전체 CLI 옵션

```bash
bit-axon train --help
```

| 옵션 | 기본값 | 설명 |
|--------|---------|-------------|
| `--model-weights` / `-w` | 필수 | 모델 가중치 디렉토리 경로 |
| `--tokenizer` / `-t` | `Qwen/Qwen2.5-3B` | Tokenizer 식별자 |
| `--val-data` | None | 검증 JSONL 파일 |
| `--lora-rank` | 8 | 어댑터 랭크 |
| `--lora-dropout` | 0.0 | 어댑터 dropout |
| `--lora-scale` | 20.0 | 어댑터 스케일링 |
| `--no-dora` | False | DoRA 대신 LoRA 사용 |
| `--learning-rate` / `-lr` | 1e-4 | 최대 학습률 |
| `--max-steps` | 10,000 | 전체 학습 스텝 수 |
| `--batch-size` | 1 | 배치당 시퀀스 수 |
| `--grad-accum-steps` | 4 | 기울기 누적 |
| `--max-seq-len` | 2048 | 최대 시퀀스 길이 |
| `--warmup-steps` | 100 | 웜업 스텝 |
| `--max-grad-norm` | 1.0 | 기울기 클리핑 |
| `--seed` | 42 | 난수 시드 |
| `--no-thermal` | False | 온도 모니터링 비활성화 |
| `--temp-pause` | 85.0 | 일시 정지 임계값 (°C) |
| `--temp-stop` | 95.0 | 중지 임계값 (°C) |
| `--output-dir` / `-o` | `checkpoints` | 체크포인트 디렉토리 |
| `--save-every` | 500 | 체크포인트 간격 |
| `--eval-every` | 500 | 평가 간격 |
| `--resume` | False | 최신 체크포인트에서 재개 |
| `--config-small` | False | 테스트용 소형 모델 사용 |

---

## Python API

### 전체 SFT 학습 예제

```python
import mlx.core as mx
from bit_axon import BitAxonConfig, BitAxonModel
from bit_axon.tokenizer import QwenTokenizerWrapper
from bit_axon.training import TrainingConfig, Trainer, apply_lora_to_model
from bit_axon.training.data import SFTDataset, CacheDataset
from bit_axon.training.cooling import CoolingScheduler, ThermalPolicy

# 1. 모델 로드
model_config = BitAxonConfig()
model = BitAxonModel(model_config)

# 2. DoRA 어댑터 적용
wrapped_layers = apply_lora_to_model(
    model,
    rank=8,
    dropout=0.0,
    scale=20.0,
    use_dora=True,
)
mx.eval(model.parameters())

# 3. 데이터 준비
tokenizer = QwenTokenizerWrapper("Qwen/Qwen2.5-3B")
dataset = CacheDataset(SFTDataset("data.json", tokenizer, max_seq_len=2048))
val_dataset = SFTDataset("val.json", tokenizer, max_seq_len=2048)

# 4. 학습 설정
config = TrainingConfig(
    learning_rate=1e-4,
    max_steps=5000,
    grad_accum_steps=4,
    save_every=500,
    eval_every=500,
    output_dir="checkpoints/my-run",
)

# 5. 온도 모니터링 설정
policy = ThermalPolicy(pause_temp=85.0, stop_temp=95.0)
cooling = CoolingScheduler(thermal_monitor, policy)

# 6. 학습
trainer = Trainer(model, config, dataset, val_dataset, cooling_scheduler=cooling)
result = trainer.train()

print(f"Step {result['step']}: loss={result['loss']:.4f}, grad_norm={result['grad_norm']:.4f}")
```

### ORPO 학습 예제

```python
from bit_axon.training.config import TrainingConfig
from bit_axon.training.data import ORPODataset
from bit_axon.training.orpo_trainer import ORPOTrainer

config = TrainingConfig(
    training_mode="orpo",
    beta=0.1,
    max_steps=2000,
    save_every=500,
)

dataset = ORPODataset("prefs.jsonl", tokenizer, max_seq_len=2048)
trainer = ORPOTrainer(model, config, dataset, cooling_scheduler=cooling)
result = trainer.train()

print(f"Reward margin: {result['reward_margin']:.4f}")
print(f"Reward accuracy: {result['reward_accuracy']:.2f}")
```

### 수동 어댑터 적용

```python
from bit_axon.training.lora import apply_lora_to_model, LoRALinear
from bit_axon.training.dora import DoRALinear

# 모든 대상 레이어에 적용 (래핑된 경로 목록 반환)
wrapped = apply_lora_to_model(model, rank=8, scale=20.0, use_dora=True)
print(f"Wrapped {len(wrapped)} layers: {wrapped[:3]}...")

# 개별 레이어 제어
dora_layer = DoRALinear.from_base(existing_linear, r=8, scale=20.0)
lora_layer = LoRALinear.from_base(existing_linear, r=8, scale=20.0)

# 어댑터를 베이스 가중치에 퓨즈
fused_linear = dora_layer.fuse()  # 크기 벡터로 재정규화
```

---

## 체크포인팅 및 재개

### 자동 체크포인트

체크포인트는 `save_every` 스텝마다 저장됩니다 (기본값 500). 각 체크포인트는 다음을 포함합니다.

| 파일 | 내용 |
|------|----------|
| `adapters.safetensors` | 모든 모델 파라미터 (어댑터 가중치는 `lora_a`, `lora_b`, `.m` 키로 식별 가능) |
| `optimizer_state.safetensors` | AdamW 모멘텀 및 분산 버퍼 |
| `training_state.json` | `{"step": int, "loss": float}` |

회전 정책은 최근 3개의 체크포인트를 유지하고 오래된 것은 삭제합니다.

```
checkpoints/
├── step_00000500/
│   ├── adapters.safetensors
│   ├── optimizer_state.safetensors
│   └── training_state.json
├── step_00001000/
│   ├── adapters.safetensors
│   ├── optimizer_state.safetensors
│   └── training_state.json
└── step_00001500/
    ├── adapters.safetensors
    ├── optimizer_state.safetensors
    └── training_state.json
```

### 학습 재개

Trainer는 `output_dir`에서 가장 높은 스텝의 체크포인트를 찾아 어댑터 가중치와 옵티마이저 상태를 복원하고, 해당 스텝에서 계속합니다.

```bash
bit-axon train data.json --model-weights ./model --tokenizer Qwen/Qwen2.5-3B \
    --output-dir ./checkpoints --resume
```

### 체크포인트 Python API

```python
from bit_axon.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    save_adapter_only,
)

# 수동으로 체크포인트 저장
ckpt_path = save_checkpoint(model, optimizer, step=1500, loss=1.23, output_dir="checkpoints")

# 최신 체크포인트 찾기
latest = get_latest_checkpoint("checkpoints")  # Path 또는 None 반환

# 체크포인트 로드 (모델 가중치 + 옵티마이저 상태 복원)
step, loss = load_checkpoint(model, optimizer, latest)

# 어댑터 가중치만 내보내기 (공유 또는 배포용)
save_adapter_only(model, "my_adapter.safetensors")
```

---

## 어댑터 병합

학습 후 어댑터 가중치를 베이스 모델에 퓨징하여 배포합니다.

### CLI

```bash
bit-axon merge ./model \
    --adapter ./checkpoints/final_adapter.safetensors \
    --output ./merged-model
```

기본적으로 병합된 모델은 4-bit로 재양자화됩니다. 전체 정밀도 가중치를 유지하려면 `--no-re-quantize`를 전달하세요.

### Python API

```python
from bit_axon.training.merging import merge_adapters, save_merged_model, load_and_merge

# 원스텝: 베이스 + 어댑터 로드, 병합, 재양자화, 저장
load_and_merge(
    base_model_path="./model",
    adapter_path="./checkpoints/final_adapter.safetensors",
    output_dir="./merged-model",
    quantize_after_merge=True,
    bits=4,
    group_size=64,
    lora_rank=8,
)

# 수동 단계별 실행
merge_adapters(model)
save_merged_model(model, "./merged-model", config=model_config)
```

---

## 캐싱

`CacheDataset`으로 데이터셋을 래핑하면 에포크 간 중복 토큰화를 피할 수 있습니다.

```python
from bit_axon.training.data import SFTDataset, CacheDataset

raw = SFTDataset("train.jsonl", tokenizer, max_seq_len=2048)
cached = CacheDataset(raw)

# 첫 접근 시 토큰화 및 캐싱; 이후 접근은 캐시에서 로드
for token_ids, loss_mask in cached:
    train_step(token_ids, loss_mask)
```

이는 배치 반복자에서 `loop=True`를 사용할 때 특히 유용합니다. 학습 중 데이터셋을 여러 번 순회할 때 캐싱이 효과를 발휘합니다.

---

## 학습 파이프라인 요약

`bit-axon train`이 실행하는 전체 SFT 파이프라인:

| 단계 | 동작 | 모듈 |
|------|--------|--------|
| 1 | `BitAxonConfig` 생성 | `config.py` |
| 2 | `TrainingConfig` 생성 | `training/config.py` |
| 3 | `BitAxonModel` 및 가중치 로드 | `model.py` |
| 4 | 4-bit (NF4) 양자화 | `quantization/nf4.py` |
| 5 | 모든 가중치 고정, LoRA/DoRA 어댑터 적용 | `training/lora.py`, `training/dora.py` |
| 6 | Tokenizer 및 데이터셋 로드 | `training/data.py` |
| 7 | `powermetrics`를 통해 `ThermalMonitor` 시작 | `profiling/thermal.py` |
| 8 | 기울기 누적으로 학습 루프 실행 | `training/trainer.py` |
| 9 | 최종 어댑터 가중치 저장 | `training/checkpoint.py` |
| 10 | 결과 출력 | CLI |

---

## 관련 페이지

- [CLI 레퍼런스](../cli/reference.ko.md): 전체 명령어 문서
- [양자화 가이드](quantization.ko.md): NF4 양자화 및 메모리 예산
- [아키텍처](../architecture/index.md): 모델 설계, 샌드위치 구조, 메모리 레이아웃
