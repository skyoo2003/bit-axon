# 추론 가이드

CLI 또는 Python API로 Bit-Axon 모델을 실행합니다. 이 가이드에서는 빠른 추론, 대화형 채팅, 스트리밍, sampling 전략, 모델 로딩을 다룹니다.

## 빠른 시작

Bit-Axon을 설치하고 모델을 다운로드한 뒤 단일 명령으로 텍스트를 생성합니다.

```bash
pip install bit-axon
bit-axon download skyoo2003/bit-axon
bit-axon run "Explain quantum computing in simple terms."
```

CLI는 기본적으로 스트리밍으로 출력하며, 완료 시 토큰 수와 속도를 출력합니다.

```
Quantum computing uses qubits instead of classical bits...
── 128 tokens · 42.3 tok/s · TTFT 180ms ──
```

## CLI 추론

### 단일 프롬프트

프롬프트를 위치 인자로 전달합니다. 모든 생성 파라미터를 플래그로 사용할 수 있습니다.

```bash
bit-axon run "Write a haiku about debugging" \
  --model skyoo2003/bit-axon \
  --max-tokens 256 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --seed 42
```

| 플래그 | 단축 | 기본값 | 설명 |
|---|---|---|---|
| `--model` | `-m` | `skyoo2003/bit-axon` | 로컬 경로 또는 HuggingFace 저장소 ID |
| `--tokenizer` | `-t` | 모델과 동일 | Tokenizer 경로 또는 HF 저장소 ID |
| `--max-tokens` | | `512` | 생성할 최대 토큰 수 |
| `--temperature` | | `0.6` | Sampling temperature |
| `--top-k` | | `50` | Top-k 필터링 |
| `--top-p` | | `0.95` | Nucleus sampling 임계값 |
| `--seed` | | 없음 | 재현성을 위한 난수 시드 |
| `--no-stream` | | false | 마지막에 전체 응답 출력 |

위치 인자가 없으면 stdin에서 입력을 파이프로 받습니다.

```bash
echo "Summarize this article in three bullet points." | bit-axon run
```

### 대화형 채팅

`--chat`(또는 `-c`)으로 다중 턴 대화를 시작합니다.

```bash
bit-axon run --chat
```

채팅 루프는 턴 간 대화 기록을 유지하며, tokenizer의 채팅 템플릿을 자동으로 적용합니다. `exit`을 입력하거나 Ctrl+C를 눌러 종료합니다.

```
You: What are the main differences between Rust and Go?
Assistant: Rust prioritizes memory safety through ownership...
You: Which one would you pick for a web API?
Assistant: For a web API, Go is often the pragmatic choice...
```

### 소형 모델로 테스트

`--config-small`을 사용하면 가중치 다운로드 없이 즉시 작은 모델을 실행할 수 있습니다. 파이프라인 테스트에 유용합니다.

```bash
bit-axon run "Hello" --config-small
```

## Python API

### 모델 로딩

`load_model`로 로컬 디렉토리나 HuggingFace Hub 저장소에서 가중치를 로드합니다. 로드 시 NF4 양자화를 적용하려면 `quantize=True`를 전달하세요.

```python
from bit_axon.inference import load_model

# HuggingFace Hub에서 (자동 다운로드 및 캐싱)
model = load_model("skyoo2003/bit-axon", quantize=True)

# 로컬 디렉토리에서
model = load_model("./my-model", quantize=True)
```

`load_model`은 가중치 디렉토리에서 `config.json`을 찾습니다. 없으면 `BitAxonConfig()` 기본값으로 대체합니다. 디렉토리 내의 모든 `.safetensors` 파일이 로드됩니다.

사용자 정의 설정을 전달할 수도 있습니다.

```python
from bit_axon import BitAxonConfig
from bit_axon.inference import load_model

config = BitAxonConfig(hidden_dim=256, num_layers=4)
model = load_model("./tiny-model", config=config)
```

### 기본 생성

`generate` 함수는 전체 자기회귀 루프를 실행합니다. 프롬프트를 프리필한 다음 `max_tokens`에 도달하거나 EOS 토큰이 샘플링될 때까지 한 토큰씩 디코딩합니다.

```python
from bit_axon.inference import load_model, generate, GenerateConfig
from bit_axon.tokenizer import QwenTokenizerWrapper

model = load_model("skyoo2003/bit-axon", quantize=True)
tokenizer = QwenTokenizerWrapper("skyoo2003/bit-axon")

result = generate(
    model,
    tokenizer,
    "Explain async/await in Python.",
    config=GenerateConfig(max_tokens=256),
)

print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### GenerateConfig 옵션

`GenerateConfig`는 생성 동작을 제어합니다. 모든 필드에 합리적인 기본값이 있습니다.

```python
config = GenerateConfig(
    max_tokens=512,          # 생성할 최대 토큰 수
    temperature=0.6,         # Sampling temperature (0.0 = 탐욕적, 1.0 = 창의적)
    top_k=50,                # Sampling 중 상위 k logits 유지 (0 = 비활성화)
    top_p=0.95,              # Nucleus sampling 임계값 (1.0 = 비활성화)
    repetition_penalty=1.0,  # 반복 토큰 페널티 (1.0 = 비활성화)
    seed=42,                 # 재현성을 위한 난수 시드
)

result = generate(model, tokenizer, "prompt", config=config)
```

!!! tip
    일관성이 중요한 작업(코드 생성, 구조화된 출력 등)에는 `temperature=0.0`으로 설정하세요. 결정론적 탐욕 디코딩을 사용합니다.

### GenerateResult 필드

`generate`는 출력과 성능 메트릭이 포함된 `GenerateResult`를 반환합니다.

```python
result = generate(model, tokenizer, "Hello, world!")

result.text                  # 디코딩된 출력 문자열
result.token_ids             # 생성된 토큰 ID 목록 (프롬프트 제외)
result.prompt_tokens         # 입력 프롬프트의 토큰 수
result.completion_tokens     # 생성된 토큰 수
result.tokens_per_sec        # 생성 처리량
result.time_to_first_token_ms  # 프리필 시작부터 첫 토큰까지의 시간
```

### 메시지로 채팅

메시지 목록을 전달하여 tokenizer의 채팅 템플릿을 사용합니다.

```python
messages = [
    {"role": "system", "content": "You are a concise technical writer."},
    {"role": "user", "content": "Explain KV caching."},
]

result = generate(model, tokenizer, "", messages=messages)
print(result.text)
```

또는 `chat=True` 플래그를 사용하여 단일 프롬프트를 채팅 템플릿으로 감쌀 수 있습니다.

```python
result = generate(model, tokenizer, "What is attention?", chat=True)
```

## 스트리밍

`stream=True`로 설정하면 토큰이 생성될 때마다 부분 텍스트를 산출하는 제너레이터를 받습니다.

```python
from bit_axon.inference import load_model, generate, GenerateConfig
from bit_axon.tokenizer import QwenTokenizerWrapper

model = load_model("skyoo2003/bit-axon", quantize=True)
tokenizer = QwenTokenizerWrapper("skyoo2003/bit-axon")

config = GenerateConfig(max_tokens=256, temperature=0.7)

for text in generate(model, tokenizer, "Tell me a story.", config=config, stream=True):
    print(text, end="", flush=True)

# 제너레이터가 소진되면 GenerateResult 반환:
gen = generate(model, tokenizer, "prompt", config=config, stream=True)
for text in gen:
    print(text, end="", flush=True)

result = gen.return_value
print(f"\n\n{result.completion_tokens} tokens at {result.tokens_per_sec:.1f} tok/s")
```

스트리밍은 채팅 모드에서도 동작합니다.

```python
messages = [{"role": "user", "content": "Write a poem about the sea."}]

for text in generate(model, tokenizer, "", messages=messages, stream=True):
    print(text, end="", flush=True)
```

## Sampling 전략

Bit-Axon은 세 가지 sampling 필터를 순차적으로 적용합니다: temperature 스케일링, top-k 필터링, top-p (nucleus) sampling. 이들은 함께 조합되므로 세밀한 제어가 가능합니다.

### Temperature

Temperature는 sampling 전 logits를 스케일링하여 출력의 무작위성을 제어합니다.

```python
# 결정론적 출력: 항상 최고 확률 토큰 선택
greedy = GenerateConfig(temperature=0.0)

# 기본값: 약간의 무작위성, 대부분의 작업에 적합한 균형
balanced = GenerateConfig(temperature=0.6)

# 창의적: 더 높은 무작위성, 더 다양한 출력
creative = GenerateConfig(temperature=1.0)
```

- **0.0**: `argmax`를 통한 탐욕 디코딩. 무작위성 없음. 사실적이거나 코드 작성 작업에 최적.
- **0.6**: 기본값. 출력의 일관성을 유지하면서 통제된 변화를 추가.
- **1.0**: 스케일링 없음. 순수 확률 분포, 최대 다양성.
- **1.0 초과**: 분포를 더 평탄하게 만들어 무작위성은 높아지지만 일관성이 저하됨.

!!! warning
    매우 높은 temperature(1.5 이상)는 일관성 없는 텍스트를 생성하는 경향이 있습니다. 대부분의 실용적인 사용 사례는 0.0과 1.0 사이에 있습니다.

### Top-k 필터링

Top-k는 확률이 가장 높은 k개 토큰만 유지하고 나머지는 버립니다.

```python
# 적극적: 상위 10개 토큰만 고려
config = GenerateConfig(top_k=10)

# 기본값: 상위 50개 토큰
config = GenerateConfig(top_k=50)

# 비활성화: 모든 토큰 고려
config = GenerateConfig(top_k=0)
```

값이 작을수록 모델이 더 집중되지만 덜 창의적입니다. `top_k=0`으로 설정하면 필터링이 완전히 비활성화됩니다.

### Top-p (Nucleus) Sampling

Top-p는 누적 확률이 임계값을 초과하는 가장 작은 토큰 집합을 선택합니다.

```python
# 엄격: 가장 확률이 높은 토큰만
config = GenerateConfig(top_p=0.8)

# 기본값: 적절한 균형
config = GenerateConfig(top_p=0.95)

# 비활성화: 필터링 없음
config = GenerateConfig(top_p=1.0)
```

Top-p는 확률 분포에 동적으로 적응합니다. 모델이 확신이 있을 때(하나의 토큰이 지배적), 여전히 작은 집합에서 선택합니다. 불확실할 때는 더 많은 옵션을 고려합니다.

### 전략 조합

세 필터는 순서대로 적용됩니다: temperature, top-k, top-p. 함께 잘 동작합니다.

```python
# 집중적이고 사실적인 출력
config = GenerateConfig(temperature=0.2, top_k=20, top_p=0.85)

# 균형 잡힌 스토리텔링
config = GenerateConfig(temperature=0.7, top_k=50, top_p=0.95)

# 매우 창의적인 브레인스토밍
config = GenerateConfig(temperature=1.0, top_k=100, top_p=0.99)
```

### 재현 가능한 출력

시드를 설정하면 실행 간에 동일한 출력을 얻을 수 있습니다.

```python
config = GenerateConfig(temperature=0.8, seed=42)
result1 = generate(model, tokenizer, "What is life?", config=config)
result2 = generate(model, tokenizer, "What is life?", config=config)
assert result1.text == result2.text  # True
```

## KV Cache

Bit-Axon의 24레이어 아키텍처는 샌드위치 구조에 맞춘 하이브리드 캐싱 전략을 사용합니다.

- **레이어 1-8 (순수 SSM)**: 외부 cache 없음. SSM 레이어는 토큰당 O(1)로 증가하는 내부 상태 벡터를 유지하므로 KV cache가 필요 없습니다.
- **레이어 9-16 (SWA + MoE)**: 슬라이딩 윈도우 어텐션을 위해 `KVCache` 객체를 사용합니다. 이 cache는 4K 어텐션 윈도우의 key/value 쌍을 저장합니다.
- **레이어 17-24 (SSM + MoE)**: 외부 cache 없음. 레이어 1-8과 마찬가지로 SSM 상태가 모든 것을 내부적으로 처리합니다.

모델을 호출하면 길이 24의 `caches` 목록이 반환됩니다. 위치 0-8과 17-23은 `None`이며, 위치 8-16은 `KVCache` 인스턴스를 갖습니다.

```python
import mlx.core as mx

input_ids = mx.array([[1, 42, 100, 200, 500]], dtype=mx.uint32)
logits, caches = model(input_ids)

# caches[0:8]    -> None (SSM 레이어)
# caches[8:16]   -> KVCache 객체 (SWA 레이어)
# caches[16:24]  -> None (SSM + MoE 레이어)
```

자기회귀 생성 중 cache는 각 디코드 스텝마다 전달됩니다.

```python
logits, caches = model(input_ids)           # Prefill
logits, caches = model(next_token, cache=caches)  # 디코드 스텝 1
logits, caches = model(next_token, cache=caches)  # 디코드 스텝 2
```

`generate` 함수는 cache 관리를 자동으로 처리합니다. 사용자 정의 생성 루프를 작성하는 경우에만 cache를 직접 다루면 됩니다.

!!! info
    24개 레이어 중 8개만 KV cache를 사용하므로, Bit-Axon의 추론 중 메모리 사용량은 작게 유지됩니다. 이는 16GB Apple Silicon 기기에서 모델을 실행하기 위한 의도적인 설계 선택입니다.
