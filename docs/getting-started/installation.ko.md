# 설치

## 사전 요구 사항

### 하드웨어

Bit-Axon은 Apple Silicon 전용으로 동작합니다. M1, M2, M3, M4 시리즈 칩이 탑재된 Mac이 필요합니다. Intel 기반 Mac은 지원하지 않습니다.

### 운영 체제

macOS 13 (Ventura) 이상이 필요합니다.

### Python

Python 3.10 이상. 아직 설치하지 않았다면 Homebrew를 사용하는 것이 가장 간단합니다:

```bash
brew install python@3.12
```

!!! tip
    여러 버전의 Python을 함께 사용한다면 `uv`나 `pyenv`로 환경을 관리해 보세요. Bit-Axon은 3.10부터 3.13까지 모든 Python 버전과 호환됩니다.

### MLX

MLX는 별도로 설치할 필요가 없습니다. `bit-axon` 패키지의 의존성으로 선언되어 있어 `pip` 설치 시 자동으로 함께 설치됩니다.

## 패키지 설치

```bash
pip install bit-axon
```

이 명령으로 `mlx`, `numpy`, `tokenizers`, `huggingface_hub`, `typer`, `rich` 등 모든 런타임 의존성이 설치됩니다.

### 설치 확인

```bash
bit-axon --version
```

```bash
bit-axon --help
```

두 명령 모두 에러 없이 출력되면 설치가 완료된 것입니다.

## SwiftUI 앱 (선택 사항)

Bit-Axon은 SwiftUI로 구축된 네이티브 macOS 채팅 애플리케이션을 제공합니다. 사용하려면 다음이 필요합니다:

- Xcode 15 이상
- macOS 14 (Sonoma) 이상

```bash
cd BitAxonApp
open BitAxonApp.xcodeproj
```

Xcode에서 빌드하고 실행합니다. 앱은 실시간 token 스트리밍, GPU 메모리 모니터링, 드래그 앤 드롭 파인튜닝을 지원합니다.

!!! warning
    SwiftUI 앱은 Xcode와 macOS 14 이상이 필요합니다. CLI와 Python API만 사용한다면 이 섹션을 건너뛰셔도 됩니다.

## 개발 환경 설정

기여하거나 코드베이스를 직접 다루려면 편집 가능한 모드로 패키지를 설치하세요:

```bash
git clone https://github.com/skyoo2003/bit-axon.git
cd bit-axon
pip install -e ".[dev]"
```

`[dev]` 추가 항목은 `pytest`, `pytest-xdist`, `ruff`, `pre-commit`을 설치합니다.

### Pre-commit 훅

푸시 전에 린트 문제를 잡아주는 pre-commit 훅을 설정합니다:

```bash
pre-commit install
```

### Ruff

Ruff는 린팅과 포매팅을 모두 처리합니다. 수동으로 실행하려면:

```bash
ruff check .
ruff format .
```

프로젝트 설정은 `pyproject.toml`의 `[tool.ruff]` 섹션에 있습니다.

## 문제 해결

**"No module named mlx"**: MLX는 Apple Silicon이 필요합니다. Intel Mac에서는 동작하지 않습니다. Apple Silicon Mac인데도 이 문제가 발생하면 `pip install --force-reinstall mlx`로 재설치해 보세요.

**"Python 3.9 or earlier"**: Bit-Axon은 Python 3.10 이상이 필요합니다. `python3 --version`으로 버전을 확인하고 업그레이드하세요.

**"Command not found: bit-axon"**: pip으로 설치한 스크립트가 PATH에 없을 수 있습니다. `python3 -m bit_axon`을 대신 사용해 보거나, pip bin 디렉터리가 PATH에 포함되어 있는지 확인하세요.
