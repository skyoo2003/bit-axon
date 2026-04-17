# Bit-Axon

**Minimal Bits, Maximal Impulse**

GPU 없이, 클라우드 없이. 팬리스 MacBook Air M4에서 학습, 추론, 배포를 모두 끝내는 32억 파라미터 하이브리드 소형 언어 모델 엔진. Apple Silicon을 위해 처음부터 설계되었습니다.

---

## 설치

```bash
pip install bit-axon
```

[![PyPI](https://img.shields.io/pypi/v/bit-axon?color=blue&label=pypi)](https://pypi.org/project/bit-axon/)
[![License](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/skyoo2003/bit-axon)

---

## 왜 Bit-Axon인가?

대부분의 LLM은 데이터센터가 있다고 가정합니다. Bit-Axon은 MacBook이 있다고 가정합니다.

**Python + Apple MLX** (PyTorch가 아닙니다)로 구축된 Bit-Axon은 24레이어 하이브리드 아키텍처를 Q4 양자화와 함께 약 1.76GB의 가중치로 구동합니다. 16GB RAM에 들어맞고, 열 스로틀링 없이 학습하며, 네이티브 macOS 앱으로 배포할 수 있습니다.

!!! tip "인프라 제로"
    CUDA 드라이버, 클라우드 과금, 렌탈 GPU가 전혀 필요 없습니다. 단一台의 Apple Silicon 맥으로 데이터 준비부터 추론까지 전체 수명 주기를 처리합니다.

---

## 주요 기능

=== "아키텍처"

    :material-brain: &nbsp; **하이브리드 샌드위치 설계**

    24레이어 아키텍처가 단일 순전파에서 Axon-SSM, 슬라이딩 윈도우 어텐션, 혼합 전문가(MoE)를 겹겹이 쌓습니다. 각 레이어 유형이 가장 잘하는 역할을 맡습니다.

=== "효율성"

    :material-memory: &nbsp; **Q4 양자화**

    4비트 정밀도로 약 1.76GB의 가중치. 컨텍스트 윈도우와 KV 캐시까지 포함해 16GB RAM에서 여유롭게 구동됩니다.

=== "열 관리"

    :material-thermometer: &nbsp; **Powermetrics 기반 학습**

    macOS의 `powermetrics`를 실시간으로 읽어 배치 크기와 학습률을 자동 조절합니다. 열 제한 내에서 수동 개입 없이 학습이 진행됩니다.

=== "도구"

    :material-console: &nbsp; **10개 CLI 명령어**

    학습, 평가, 양자화, 내보내기, 채팅 등 모델 수명 주기의 모든 단계에 전용 명령어가 있습니다.

=== "네이티브 앱"

    :material-apple: &nbsp; **SwiftUI macOS 애플리케이션**

    추론 전용 데스크톱 앱. 터미널이 필요 없습니다. 모델 선택, 프롬프트 기록, 생성 파라미터를 모두 네이티브 인터페이스에서 관리합니다.

=== "오픈 스택"

    :material-open-source-initiative: &nbsp; **완전 오픈소스**

    Apache 2.0 라이선스. PyPI 패키지로 `pip install` 가능. HuggingFace에서 가중치와 데이터셋 제공. 나머지는 모두 GitHub에서 관리합니다.

---

## 빠른 시작

패키지를 설치하고 첫 번째 생성을 실행해 보세요:

```bash title="Terminal"
pip install bit-axon
bit-axon run --model skyoo2003/bit-axon --prompt "Explain quantum entanglement in one sentence."
```

또는 대화형 채팅 세션을 실행합니다:

```bash title="Terminal"
bit-axon chat --model skyoo2003/bit-axon
```

!!! note "모델 가중치"
    처음 실행 시 가중치가 HuggingFace에서 자동으로 다운로드됩니다. 이후에는 네트워크 연결 없이 모든 작업이 로컬에서 실행됩니다.

---

## 기술 스택

| 구성 요소 | 선택 |
|-----------|------|
| 언어 | Python 3.10+ |
| ML 프레임워크 | Apple MLX |
| 아키텍처 | Axon-SSM + SWA + MoE |
| 파라미터 | 3.2B |
| 양자화 | Q4 (~1.76GB) |
| 데스크톱 앱 | SwiftUI (macOS) |
| 라이선스 | Apache 2.0 |

---

## 링크

[:material-github: GitHub](https://github.com/skyoo2003/bit-axon){ .md-button }
[:simple-huggingface: HuggingFace](https://huggingface.co/skyoo2003/bit-axon){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/bit-axon/){ .md-button }

---

## 문서

[:material-book-open-variant: 시작하기](getting-started/index.ko.md){ .md-button }
[:material-cog: CLI 레퍼런스](cli/reference.md){ .md-button }
[:material-sitemap: 아키텍처](architecture/index.md){ .md-button }
[:material-api: API 레퍼런스](api/index.md){ .md-button }
[:material-laptop: macOS 앱](macos-app/index.md){ .md-button }
[:material-school: 가이드](guides/training.md){ .md-button }
