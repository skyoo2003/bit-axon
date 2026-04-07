# 가중치 포팅 가이드: Qwen2.5-3B → Bit-Axon

Bit-Axon의 아키텍처는 표준 트랜스포머와 상당히 다르지만, 초기 가중치는 Qwen2.5-3B에서 부트스트랩합니다. 이 가이드에서는 전체 포팅 파이프라인을 다룹니다. 각 파라미터 패밀리가 어떻게 매핑, 변환, 검증되는지 설명합니다.

## 왜 Qwen2.5-3B인가?

Qwen2.5-3B는 세 가지 이유로 검증된 기반이 됩니다.

- **사전 학습된 표현**. 임베딩 행과 RMSNorm 가중치는 아키텍처 간 전이 가능한 분포 지식을 인코딩합니다. 이를 시작점으로 사용하는 것이 무작위 초기화보다 훨씬 낫습니다.
- **차원 정렬**. Qwen2.5-3B의 은닉 차원(2048)은 Bit-Axon의 2560 바로 아래에 있습니다. 이로 인해 RMSNorm에 대한 패딩 전략이 거의 손실이 없으며, MLP→MoE 프로젝션이 소스 모델의 feedforward 용량 대부분을 유지할 수 있습니다.
- **MLP 호환성**. Qwen2.5-3B의 밀집 SwiGLU MLP는 Bit-Axon의 공유 expert MoE에 깔끔하게 매핑됩니다. 게이트 구조(gate/up/down 프로젝션)가 보존되며, expert 간에 잘리고 복제됩니다.

## 아키텍처 불일치 한눈에 보기

| 측면 | Qwen2.5-3B | Bit-Axon |
|---|---|---|
| 어휘 크기 | 151,936 토큰 | 32,000 토큰 |
| 은닉 차원 | 2,048 | 2,560 |
| MLP 중간 차원 | 11,008 | 4,096 (expert당) |
| FFN 구조 | 밀집 SwiGLU | 8-expert 공유 MoE |
| 정규화 레이어 | RMSNorm (2048) | RMSNorm (2560) |
| 어텐션 | 전체 36 레이어 | SWA만, 레이어 9-16 |
| SSM 레이어 | 없음 | 레이어 1-8, 17-24 |

## 어휘 매핑

첫 번째 단계는 Qwen의 151K tokenizer를 Bit-Axon의 32K로 축소하는 것입니다. 두 가지 전략이 있습니다.

### First-N (기본값)

BPE 병합 순서대로 처음 32,000개 토큰을 가져옵니다. BPE 병합이 빈도 순서에 근사하므로 가장 흔한 토큰이 유지됩니다.

```python
from bit_axon.porting.vocab_map import build_vocab_mapping

# 기본값: BPE 순서대로 처음 32K 토큰
vocab_mapping = build_vocab_mapping(
    tokenizer_name="Qwen/Qwen2.5-3B",
    target_size=32000,
)
# vocab_mapping: {0: 0, 1: 1, 2: 2, ..., 31999: 31999}
```

### 빈도 기반 선택

대표적인 코퍼스 텍스트를 전달하여 가장 빈도가 높은 32K 토큰을 선택합니다.

```python
vocab_mapping = build_vocab_mapping(
    tokenizer_name="Qwen/Qwen2.5-3B",
    target_size=32000,
    corpus_text=open("corpus.txt").read(),
)
# vocab_mapping의 키는 Qwen ID, 값은 새 Bit-Axon ID
```

매핑은 간단한 dict입니다: `{old_qwen_id: new_bitaxon_id}`. 다운스트림에서 임베딩 추출 단계가 이를 읽어 행을 재정렬합니다.

### 축소된 tokenizer 로드

매핑을 구축한 후 32K 선택된 토큰만 아는 tokenizer를 만들 수 있습니다.

```python
from bit_axon.porting.vocab_map import load_truncated_tokenizer

tokenizer = load_truncated_tokenizer("Qwen/Qwen2.5-3B", vocab_mapping)
encoded = tokenizer.encode("Hello, world!")
print(encoded.ids)  # 모든 ID가 [0, 32000) 범위
```

## 가중치 매핑

`weight_map.py`는 BitAxonModel의 모든 파라미터를 5가지 변환 카테고리 중 하나로 분류합니다. 기본 설정(24 레이어, 8 expert)으로 517개 파라미터 키를 생성합니다.

```python
from bit_axon.config import BitAxonConfig
from bit_axon.porting.weight_map import build_key_mappings

config = BitAxonConfig()
mappings = build_key_mappings(config)

# 분류 결과 확인
from collections import Counter
counts = Counter(m.transform for m in mappings)
print(counts)
# Counter({'default': 401, 'pad_2048_2560': 72, 'moe_project': 24,
#          'copy_perturb': 18, 'vocab_extract': 2})
```

각 매핑은 세 개 필드를 가진 `KeyMapping`입니다.

- `target_key`: BitAxonModel에서의 파라미터 이름
- `source_key`: Qwen2.5-3B에서의 해당 파라미터 이름 (또는 동등한 것이 없으면 `None`)
- `transform`: 소스 가중치를 대상 가중치로 변환하는 방법

## 변환 유형

### `vocab_extract`: 임베딩 행 재정렬

`embed_tokens.weight`와 `lm_head.weight`에 적용됩니다. Bit-Axon은 가중치 결합(weight tying)을 사용하므로 두 항목이 동일한 행렬을 가리킵니다. 이 변환은 Qwen의 `(151936, 2048)` 임베딩 테이블에서 어휘 매핑으로 지정된 32K 행을 선택하고 `(32000, 2048)` 행렬로 재정렬합니다.

```python
from bit_axon.porting.mapper import extract_embeddings

embeddings = extract_embeddings(
    qwen_weights,
    vocab_mapping,
    target_vocab_size=32000,
    source_hidden_dim=2048,
)
# 형태: (32000, 2048)
```

### `pad_2048_2560`: 제로 패딩 RMSNorm

모든 `input_norm.weight`, `post_attention_norm.weight`, `post_ssm_norm.weight` 파라미터에 적용됩니다. RMSNorm은 1.0으로 초기화되므로, 2048에서 2560으로 1.0으로 패딩하는 것은 거의 손실이 없습니다.

```python
from bit_axon.porting.mapper import pad_rms_norm

padded = pad_rms_norm(qwen_weights["model.layers.5.input_layernorm.weight"], target_dim=2560)
# 형태: (2560,) — 처음 2048개 값은 Qwen에서, 나머지 512개는 1.0
```

### `moe_project`: 구조화된 잘라내기 + 제로 패딩

MoE 레이어(레이어 8-23)의 공유 expert의 gate/up/down 프로젝션에 적용됩니다. Qwen의 밀집 MLP는 중간 차원이 11,008이고, Bit-Axon의 expert는 4,096을 사용합니다. 변환은 처음 4,096개의 행/열을 잘라내고 은닉 차원을 2,048에서 2,560으로 제로 패딩합니다.

```
Qwen gate_proj (11008, 2048) → 열 0-2047 자르기, (4096, 2560)으로 패딩
Qwen up_proj   (11008, 2048) → 열 0-2047 자르기, (4096, 2560)으로 패딩
Qwen down_proj (2048, 11008) → 행 0-2047, 열 0-4095 자르기, (2560, 4096)으로 패딩
```

```python
from bit_axon.porting.mapper import project_mlp_to_shared_expert

gate, up, down = project_mlp_to_shared_expert(
    qwen_gate=qwen_weights["model.layers.10.mlp.gate_proj.weight"],
    qwen_up=qwen_weights["model.layers.10.mlp.up_proj.weight"],
    qwen_down=qwen_weights["model.layers.10.mlp.down_proj.weight"],
    target_intermediate=4096,
    target_hidden=2560,
    source_hidden=2048,
)
```

### `copy_perturb`: 라우팅 expert를 위해 공유 expert 복제

MoE 레이어의 `switch_mlp` 라우팅 expert에 적용됩니다. Expert 0은 공유 expert의 정확한 복사본입니다. Expert 1~7은 공유 expert의 가중치에 표준편차 0.02의 가우시안 노이즈를 더한 값을 갖습니다.

```python
from bit_axon.porting.mapper import init_routed_experts

routed_gate, routed_up, routed_down = init_routed_experts(
    shared_gate=gate,
    shared_up=up,
    shared_down=down,
    num_experts=8,
    perturbation_std=0.02,
)
# routed_gate 형태: (8, 4096, 2560)
# routed_up 형태:   (8, 4096, 2560)
# routed_down 형태: (8, 2560, 4096)
```

작은 섭동(perturbation)은 각 expert가 파인튜닝 전 고유한 시작점을 갖도록 하면서, 공유 expert의 검증된 표현에 충분히 가까워 빠르게 학습할 수 있도록 합니다.

### `default`: 무작위 초기화 유지

Qwen에 동등한 것이 없는 파라미터는 기본 초기화 상태를 유지합니다. 여기에는 다음이 포함됩니다.

- **SSM 파라미터** (`ssm.*`): A, B, C, D 행렬 및 합성곱 커널. Bit-Axon의 Mamba 스타일 SSM은 Qwen에 대응하는 것이 없습니다.
- **어텐션 파라미터** (`attention.*`): 슬라이딩 윈도우 어텐션 레이어의 Q, K, V, O 프로젝션.
- **라우터 파라미터** (`moe.gate.weight`): top-2 라우팅 게이트는 밀집 대응이 없습니다.
- **차원 브릿지** (`input_proj.weight`, `output_proj.weight`): 2048과 2560 차원 사이를 연결하는 선형 레이어.

## 전체 파이프라인

### CLI

CLI는 엔드투엔드로 모든 것을 처리합니다. Qwen 다운로드, 어휘 매핑 구축, 모든 변환 실행, 결과 저장.

```bash
# 실제 Qwen2.5-3B 가중치로 전체 파이프라인 실행
bit-axon port-weights ./output
# 출력: ./output/model.safetensors
```

전체 모델 다운로드 없이 빠르게 테스트하려면 소형 설정으로 목업 가중치를 사용하세요.

```bash
# 작은 모델로 테스트 (4 레이어, 256 은닉 차원, 목업 Qwen 가중치)
bit-axon port-weights ./output --config-small
```

소형 설정은 `hidden_dim=256`, `num_layers=4`, `d_source_model=128`, `vocab_size=1024`를 사용합니다. 목업 Qwen 가중치가 즉시 생성되므로 다운로드가 필요 없습니다. 전체 포팅에 앞서 파이프라인이 오류 없이 실행되는지 확인하는 데 유용합니다.

### Python API

더 많은 제어가 필요하면 파이프라인 함수를 직접 사용하세요.

```python
import mlx.core as mx
from bit_axon.config import BitAxonConfig
from bit_axon.porting.vocab_map import build_vocab_mapping
from bit_axon.porting.pipeline import initialize_from_qwen_weights, save_ported_model

# 1. Qwen 가중치 로드
weight_files = sorted(glob.glob("/path/to/qwen/*.safetensors"))
qwen_weights = {}
for f in weight_files:
    qwen_weights.update(mx.load(f))

# 2. 어휘 매핑 구축
vocab_mapping = build_vocab_mapping(
    tokenizer_name="Qwen/Qwen2.5-3B",
    target_size=32000,
)

# 3. 파이프라인 실행
config = BitAxonConfig()
model, vocab_mapping = initialize_from_qwen_weights(
    qwen_weights,
    vocab_mapping=vocab_mapping,
    config=config,
)

# 4. 저장
save_ported_model(model, "./output/model.safetensors", vocab_mapping)
```

어휘 매핑을 건너뛰고 기본 항등 매핑을 사용할 수도 있습니다.

```python
model, vocab_mapping = initialize_from_qwen_weights(qwen_weights, config=config)
```

## 검증

포팅 후 문제가 없는지 확인하기 위해 건전성 검사를 실행합니다.

### 가중치 통계

`visualization.py` 모듈은 분포 통계를 계산하고 이상을 감지합니다.

```python
from mlx.utils import tree_flatten
from bit_axon.porting.visualization import compute_weight_stats, detect_anomalies, format_stats_table

params = dict(tree_flatten(model.parameters()))
stats = compute_weight_stats(params)

# 가장 이상한 가중치 테이블 출력
print(format_stats_table(stats))

# 문제 확인
warnings = detect_anomalies(stats)
for w in warnings:
    print(w)
```

이상 감지기는 네 가지 조건을 감지합니다.

| 조건 | 임계값 | 가능한 원인 |
|---|---|---|
| 모두 0 | max == 0 and min == 0 | 변환 건너뛰거나 소스 키 누락 |
| NaN 값 | mean 또는 std가 NaN | 프로젝션 중 형태 불일치 |
| 높은 이상치 비율 | >10%의 값이 3σ 초과 | 잘못된 섭동 또는 패딩 |
| 극단적 희소성 | >99%가 0에 근접 | 차원 불일치, 빈 값으로 잘림 |

### 빠른 형태 확인

포팅 후 모든 파라미터가 예상 형태를 가지는지 확인합니다.

```python
from bit_axon.porting.weight_map import build_key_mappings

mappings = build_key_mappings(config)
params = dict(tree_flatten(model.parameters()))

for m in mappings:
    if m.target_key not in params:
        print(f"MISSING: {m.target_key}")
    elif m.transform == "vocab_extract":
        assert params[m.target_key].shape == (config.vocab_size, config.d_source_model)
    elif m.transform == "pad_2048_2560":
        assert params[m.target_key].shape == (config.hidden_dim,)
    elif m.transform == "moe_project":
        assert params[m.target_key].shape[0] in (config.moe_intermediate_dim, config.hidden_dim)
    elif m.transform == "copy_perturb":
        assert params[m.target_key].shape[0] == config.moe_num_experts

print("All shapes validated.")
```

## 포팅 후 다음 단계

포팅된 모델은 완성된 모델이 아니라 시작점입니다. SSM과 어텐션 파라미터는 무작위 초기화에서 시작합니다. 라우팅 expert는 작은 노이즈가 추가된 공유 expert의 복사본입니다. 다음을 위해 파인튜닝(`bit-axon train`을 통한 QLoRA)이 필요합니다.

- SSM 레이어가 순차적 컨텍스트를 흡수하도록 학습
- SWA 레이어의 어텐션 헤드 보정
- 라우팅 expert를 차별화하여 전문화
- 출력 헤드를 축소된 어휘에 정렬
