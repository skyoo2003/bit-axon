# 빠른 시작

이 페이지에서는 모델 다운로드, 추론 실행, Python API 사용법을 안내합니다. 아직 Bit-Axon을 설치하지 않았다면 [설치](installation.ko.md) 가이드부터 시작하세요.

## 1단계: 모델 다운로드

Bit-Axon의 가중치는 HuggingFace Hub에서 호스팅됩니다. 명령 하나로 다운로드할 수 있습니다:

```bash
bit-axon download skyoo2003/bit-axon
```

CLI가 양자화된 모델 파일을 가져와 로컬에 저장합니다. 4비트 양자화 가중치의 다운로드 크기는 약 1.8GB입니다.

!!! note
    이 단계에서 인터넷 연결이 필요합니다. 다운로드가 완료되면 이후 모든 추론은 Mac에서 완전히 오프라인으로 실행됩니다.

## 2단계: 추론 실행

프롬프트를 직접 CLI에 전달합니다:

```bash
bit-axon run "Hello, world!"
```

모델이 프롬프트를 처리하고 생성된 텍스트를 터미널에 출력합니다. 기본적으로 MLX를 통해 GPU에서 NF4 양자화 가중치로 추론이 실행됩니다.

생성 파라미터를 조절할 수도 있습니다:

```bash
bit-axon run "Explain quantum computing in one paragraph." --max-tokens 200 --temperature 0.7
```

`bit-axon run --help`로 사용 가능한 모든 옵션을 확인하세요.

## 3단계: 대화형 채팅 모드

주고받는 대화를 하려면 채팅 모드를 실행합니다:

```bash
bit-axon run --chat
```

대화형 REPL이 열리며, 프롬프트를 입력하면 스트리밍 응답을 받을 수 있습니다. `Ctrl+C`를 누르거나 `exit`을 입력하면 종료됩니다.

!!! tip
    채팅 모드는 대화 컨텍스트를 유지하므로, 세션에서 이전에 말한 내용을 모델이 기억합니다.

## 4단계: Python API

Bit-Axon을 직접 코드에 통합하려면 Python API를 사용합니다:

```python
import mlx.core as mx
from bit_axon import BitAxonConfig, BitAxonModel

config = BitAxonConfig()
model = BitAxonModel(config)

input_ids = mx.array([[1, 42, 100, 200, 500]])
logits, caches = model(input_ids)

print(f"Output shape: {logits.shape}")  # (1, 5, 32000)
```

`BitAxonConfig` 데이터클래스에서는 히든 차원, 레이어 수, SSM 상태 크기, MoE 전문가 수 등 모든 모델 파라미터를 설정할 수 있습니다. 기본 생성자는 표준 3.2B 설정을 로드합니다 (히든 차원 2,560, 24레이어, top-2 라우팅의 8 MoE 전문가, 32K 어휘).

반환된 `caches` 리스트는 SWA 어텐션 레이어(9~16레이어)의 KV 캐시 객체를 포함하며, 순수 SSM 레이어에 대해서는 `None`을 반환합니다. SSM 레이어는 외부 캐싱 없이 내부 상태를 유지하기 때문입니다.

!!! note
    모델 가중치는 1단계에서 다운로드한 경로에서 로드됩니다. 다운로드를 건너뛰었다면, 처음 사용 시 CLI에서 다운로드를 안내합니다.

## 다음 단계

추론이 실행되고 있다면, 더 심화된 작업을 위해 다음 가이드를 확인해 보세요:

- **[학습](../guides/training.md)**: 열 관리 QLoRA로 모델 파인튜닝
- **[양자화](../guides/quantization.md)**: 가중치 양자화 및 KV 캐시 압축
- **[추론](../guides/inference.md)**: 생성 속도 및 메모리 사용량 최적화
- **[벤치마킹](../guides/benchmarking.md)**: 초당 token 수 및 메모리 점유율 측정
