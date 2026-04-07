# FAQ

## 일반

### Bit-Axon이란 무엇인가요?

Bit-Axon은 Apple Silicon을 위해 설계된 32억 파라미터 하이브리드 소형 언어 모델 엔진입니다. Mamba 스타일 상태 공간 모델(Axon-SSM), 슬라이딩 윈도우 어텐션, 혼합 전문가 모델을 24층 샌드위치 아키텍처로 결합합니다. 학습부터 추론까지의 전체 파이프라인이 팬리스 MacBook Air M4 (16GB 통합 메모리)에서 실행됩니다. 별도의 GPU나 클라우드가 필요 없습니다.

### Bit-Axon은 어떤 프레임워크를 사용하나요?

Bit-Axon은 PyTorch가 아닌 **Apple MLX**로 구축되었습니다. 이를 통해 Apple Silicon의 Metal GPU 가속에 직접 접근할 수 있습니다. 추가 의존성으로 NumPy, HuggingFace `tokenizers`, `typer` (CLI), `rich` (터미널 UI)가 있습니다.

### macOS 외의 플랫폼에서도 Bit-Axon을 사용할 수 있나요?

아니요. Bit-Axon은 Apple Silicon (M1 이상)과 macOS 13+이 필요합니다. MLX 프레임워크와 열 모니터링(`powermetrics`)은 macOS 전용입니다.

---

## 설치

### MLX 설치가 실패합니다.

MLX는 `bit-axon`의 의존성으로 pip를 통해 자동 설치됩니다. 문제가 발생하면:

```bash
pip install --upgrade pip
pip install bit-axon
```

MLX가 소스에서 빌드되지 않으면 Xcode Command Line Tools가 설치되어 있는지 확인하세요:

```bash
xcode-select --install
```

### Python 버전 요구 사항

Bit-Axon은 **Python 3.10 이상**이 필요합니다. Python 3.10, 3.11, 3.12, 3.13에서 테스트되었습니다.

### "No module named mlx" 오류가 발생합니다.

MLX가 올바르게 설치되지 않은 것입니다. 재설치를 시도해 보세요:

```bash
pip uninstall mlx -y
pip install bit-axon
```

---

## 학습

### 학습 중 열 스로틀링이 발생합니다.

Bit-Axon은 macOS `powermetrics`를 통해 SoC 온도를 모니터링하는 열 인식 쿨링 스케줄러를 포함합니다. 기본 설정:

- **85°C**에서 학습 **일시 정지** (`--temp-pause`로 변경 가능)
- **95°C**에서 학습 **중지** (`--temp-stop`으로 변경 가능)

학습이 자주 일시 정지되면:

1. 다른 리소스 집약적인 애플리케이션을 종료하세요
2. 적절한 통풍을 확보하세요 (MacBook 공기 흡입구를 막지 마세요)
3. 배치 크기를 줄이세요: `--batch-size 1 --grad-accum-steps 8`
4. `--no-thermal`로 열 모니터링 비활성화 (장시간 세션에는 권장하지 않음)

!!! warning
    팬리스 MacBook에서 열 모니터링을 비활성화(`--no-thermal`)하면 지속적인 고온이 발생할 수 있습니다. 수동으로 모니터링하세요.

### 학습 중 메모리 부족이 발생합니다.

Bit-Axon은 16GB 통합 메모리용으로 설계되었습니다. OOM이 발생하면:

- `--max-seq-len`을 줄이세요 (기본값 2048)
- `--batch-size`를 1로 줄이세요
- Q4 양자화가 활성화되어 있는지 확인하세요 (학습 파이프라인 기본값)
- `--lora-rank`를 줄이세요 (기본값 8)

### 체크포인트에서 학습을 재개하려면 어떻게 하나요?

```bash
bit-axon train data.json --model-weights ./model --resume
```

`--resume` 플래그는 출력 디렉토리에서 최신 체크포인트를 로드합니다.

---

## 추론

### 모델을 어떻게 다운로드하나요?

```bash
bit-axon download skyoo2003/bit-axon
```

또는 로컬 디렉토리를 지정할 수 있습니다:

```bash
bit-axon download skyoo2003/bit-axon --local-dir ./models/bit-axon
```

### 채팅 모드를 어떻게 사용하나요?

```bash
bit-axon run --chat
```

메시지를 입력하고 Enter를 누르세요. `exit`를 입력하거나 Ctrl+C로 종료합니다.

### 추론이 느립니다.

- MLX 컴파일 캐싱으로 인해 첫 추론 호출이 느릴 수 있습니다
- 짧은 응답을 위해 `--max-tokens`를 줄이세요
- 약간 더 빠른 greedy 디코딩을 위해 `--temperature 0`을 시도해 보세요

---

## 양자화

### Bit-Axon은 어떤 양자화를 지원하나요?

Bit-Axon은 설정 가능한 그룹 크기(기본값 64)의 **NF4** (4비트 NormalFloat) 양자화를 지원합니다. 이를 통해 모델 가중치를 ~6.4GB (FP16)에서 ~1.76GB (Q4)로 줄입니다.

### 모델을 어떻게 양자화하나요?

```bash
bit-axon quantize ./model --output ./model-q4 --bits 4 --group-size 64
```

### LoRA 어댑터를 병합하고 재양자화하려면 어떻게 하나요?

```bash
bit-axon merge ./base-model --adapter ./adapter.safetensors --output ./merged
```

병합 명령은 자동으로 역양자화, 어댑터 병합, 재양자화를 수행합니다.

---

## 도움 받기

- **GitHub Issues**: [skyoo2003/bit-axon](https://github.com/skyoo2003/bit-axon/issues)
- **PyPI**: [bit-axon](https://pypi.org/project/bit-axon/)
- **HuggingFace**: [skyoo2003/bit-axon](https://huggingface.co/skyoo2003/bit-axon)
