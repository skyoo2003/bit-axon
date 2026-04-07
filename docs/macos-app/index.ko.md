# Bit-Axon macOS 앱

Apple Silicon에서 Bit-Axon 추론을 실행하는 네이티브 SwiftUI 애플리케이션입니다. 이 앱은 Python 패키지와 동일한 24층 하이브리드 아키텍처(Axon-SSM, 슬라이딩 윈도우 어텐션, shared-expert MoE)를 Swift와 MLX-Swift로 구현하며, 채팅 인터페이스, 실시간 스트리밍, GPU 모니터링, 드래그 앤 드롭 파인튜닝을 제공합니다.

## 요구 사항

| 요구 사항 | 버전 |
|---|---|
| macOS | 14 (Sonoma) 이상 |
| Xcode | 15 이상 |
| Swift Package Manager | Xcode에 번들 포함 |
| 하드웨어 | Apple Silicon (M1 이상) |

## 의존성

`Package.swift`에 선언되어 있으며 SPM이 자동으로 해결합니다.

| 패키지 | 버전 | 용도 |
|---|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | 0.29.1 to <0.30.0 | GPU 가속 텐서를 위한 핵심 MLX 바인딩 |
| [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) | 2.21.2+ | 공유 LLM 유틸리티 (MLXLMCommon) |

!!! note
    두 의존성 모두 Swift Package Manager를 통해 해결됩니다. 수동으로 설치할 것은 없습니다. Xcode가 첫 빌드 시 자동으로 가져오며, 명령어에서 `swift package resolve`를 사용할 수도 있습니다.

## 설치

### 명령어

```bash
cd BitAxonApp
swift build
```

### Xcode

```bash
open BitAxonApp.xcodeproj
# 또는 SPM 워크스페이스를 선호하는 경우:
open Package.swift
```

그런 다음 **Cmd+R**을 눌러 빌드하고 실행합니다. 스킴은 기본적으로 `BitAxonApp`입니다.

### 테스트 실행

```bash
cd BitAxonApp
swift test --filter EquivalenceTests
```

이 명령은 모든 레이어 유형에 대해 Python과 Swift의 수치적 일치성을 검증하는 교차 언어 동등성 테스트 스위트를 실행합니다. 자세한 내용은 [동등성 테스트](#equivalence-tests)를 참조하세요.

## 기능

### 실시간 token 스트리밍

모델이 생성하는 동안 토큰이 하나씩 도착합니다. 어시스턴트 메시지가 실시간으로 업데이트되며, 툴바에는 초당 토큰 처리량과 첫 토큰까지의 지연 시간(TTFT)이 표시됩니다.

### GPU 메모리 모니터링

`DeviceStat` 서비스가 2초마다 `GPU.snapshot()`을 폴링하여 툴바에 네 가지 지표를 표시합니다: 활성 메모리, 캐시 메모리, 최대 메모리, 구성된 메모리 제한입니다. 또한 상승된 권한으로 실행 시 `powermetrics`를 통해 SoC 다이 온도를 읽습니다.

### 드래그 앤 드롭 파인튜닝

Fine-Tune 뷰에서 드래그 앤 드롭(또는 파일 선택기)으로 JSONL 데이터 파일을 받은 후, 내부적으로 `bit-axon train` CLI를 실행합니다. 앱은 학습 로그를 실시간으로 스트리밍하며, 스텝 수와 손실값을 파싱하여 진행률 표시줄을 표시합니다. 학습은 서브프로세스로 실행되므로 GUI가 응답하지 않는 현상이 발생하지 않습니다.

### 대화 기록이 있는 채팅 인터페이스

`NavigationSplitView` 레이아웃은 메인 채팅 영역 옆에 사이드바(모델 컨트롤, 파인튜닝 링크, 메트릭 토글)를 배치합니다. 메시지는 사용자/어시스턴트 역할과 함께 스크롤 가능한 목록에 누적됩니다. 입력 필드는 전송 시 자동으로 지워지며, "Clear Chat" 버튼으로 기록을 삭제할 수 있습니다.

## 아키텍처

### 디렉토리 구조

```
BitAxonApp/
├── BitAxonApp.swift          # @main 진입점, GPU 캐시 제한 설정
├── ContentView.swift         # NavigationSplitView: 사이드바 + 채팅 상세
├── Models/
│   ├── BitAxonConfig.swift   # Codable 설정 구조체 (Python BitAxonConfig와 대응)
│   ├── BitAxonModel.swift    # 24층 모델, 블록 변형에 디스패치
│   ├── BitAxonKVCache.swift  # SWA 레이어용 KV 캐시
│   └── Layers/
│       ├── AxonSSM.swift     # Mamba 스타일 상태 공간 모델
│       ├── AxonSWA.swift     # 슬라이딩 윈도우 어텐션 (4K 윈도우)
│       ├── AxonMoE.swift     # Shared-expert 혼합 전문가 모델
│       ├── AxonRMSNorm.swift # RMS 정규화
│       └── AxonBlocks.swift  # 세 가지 블록 유형: SSM, SWA+MoE, SSM+MoE
├── ViewModels/
│   ├── ChatViewModel.swift   # 생성 루프, 메시지 상태, 모의 토크나이저
│   └── DeviceStat.swift      # GPU 메모리 폴링, 온도 읽기
├── Views/
│   ├── ChatView.swift        # 메시지 목록 + 입력 영역
│   ├── MessageRow.swift      # 단일 메시지 버블
│   ├── PromptInputView.swift # 전송 버튼이 있는 텍스트 필드
│   ├── MetricsView.swift     # 토큰 속도, TTFT, GPU 통계 툴바
│   └── FineTuneView.swift    # 학습 설정 + 로그 스트리밍
└── Services/
    ├── ModelService.swift    # 모델 로딩 (디렉토리 또는 기본 설정에서)
    ├── FineTuneBridge.swift  # bit-axon CLI로의 서브프로세스 브릿지
    └── BitAxonRegistry.swift # 기본 설정으로 모델 생성 팩토리
```

### Swift 모델 포팅

각 Python 모델 클래스에는 직접 대응하는 Swift 클래스가 있습니다. 설정은 `CodingKeys`를 사용하여 Swift camelCase와 Python `config.json`의 snake_case 키 사이를 매핑하므로, 동일한 설정 파일을 두 언어 모두에서 로드할 수 있습니다.

| Python 클래스 | Swift 클래스 | 참고 |
|---|---|---|
| `BitAxonConfig` | `BitAxonConfig` | 동일한 기본값, JSON용 `Codable` |
| `BitAxonModel` | `BitAxonModel` | `Module` 서브클래스, 레이어별 캐시용 `LayerCache` 열거형 |
| `BitAxonKVCache` | `BitAxonKVCache` | SWA 레이어(9-16)에서만 사용 |

### 레이어 포팅

| Python 모듈 | Swift 파일 | 주요 타입 |
|---|---|---|
| `axon_ssm.py` | `AxonSSM.swift` | `AxonSSM` (conv1d, 상태 벡터가 있는 SSM 레이어) |
| `swa.py` | `AxonSWA.swift` | `AxonSWA` (슬라이딩 윈도우 어텐션) |
| `moe.py` | `AxonMoE.swift` | `AxonSharedExpertMoE` (게이트, 8개 전문가, 공유 전문가) |
| `rms_norm.py` | `AxonRMSNorm.swift` | `AxonRMSNorm` |
| `block.py` | `AxonBlocks.swift` | `AxonSSMBlock`, `AxonSWAMoEBlock`, `AxonSSMMoEBlock` |

모델의 `getLayerType` 함수는 Python 샌드위치 레이아웃을 그대로 반영합니다: 레이어 0-7은 순수 SSM, 8-15는 SWA+MoE, 16-23은 SSM+MoE입니다.

### ViewModel 레이어

**`ChatViewModel`**은 생성 루프를 관리합니다. 메시지 배열을 보유하고, 분리된 `Task`에서 자기회귀 디코딩을 구동하며, 부분 텍스트를 메인 액터로 스트리밍합니다. 현재는 임시 `MockTokenizer`가 인코딩/디코딩을 처리하며, 나중에 실제 토크나이저(Qwen2.5 호환)로 교체될 예정입니다.

**`DeviceStat`**은 `GPU.snapshot()`과 선택적으로 `powermetrics`를 읽는 반복 타이머를 시작합니다. 모든 속성은 `@MainActor`이므로 SwiftUI가 직접 바인딩할 수 있습니다.

### Service 레이어

**`ModelService`**는 모델 수명 주기를 관리합니다. 디렉토리에서 로드하거나(`config.json`을 읽고 모델을 구성), `BitAxonModelRegistry`를 통해 기본 설정으로 대체합니다. 상태는 `idle`, `loading(progress:)`, `ready`, `failed(String)`을 거쳐 전환됩니다.

**`FineTuneBridge`**는 `PATH`(또는 일반적인 대체 경로)에서 `bit-axon` CLI를 찾고, 학습 인수와 함께 `Process`로 실행하며, stdout/stderr를 UI로 스트리밍합니다. 로그 행에서 스텝과 손실값을 파싱하여 진행률 표시줄을 구동합니다.

**`BitAxonRegistry`**는 설정에서 `BitAxonModel` 인스턴스를 생성하는 간단한 팩토리입니다.

## 동등성 테스트

테스트 타겟 `EquivalenceTests`는 `BitAxonApp/Tests/EquivalenceTests/`에 있으며, 모든 Swift 레이어가 Python 대응 레이어와 동일한 수치적 출력을 생성하는지 검증합니다.

### 작동 방식

1. `export_reference.py` (빌드에서 제외됨)이 각 Python 레이어를 결정론적 가중치와 입력으로 실행한 다음, 입력, 가중치, 출력을 `EquivalenceTestSupport/reference/`의 JSON 파일로 직렬화합니다.
2. 각 Swift 테스트는 참조 JSON을 로드하고, 동일한 가중치로 레이어를 재구성한 다음, 순전파를 실행하고 출력 텐서가 허용 오차 내에서 일치하는지 검증합니다.

### 테스트 항목

| 테스트 | 레이어 | 허용 오차 |
|---|---|---|
| `testRMSNorm` | `AxonRMSNorm` | 1e-3 |
| `testAxonSSM` | `AxonSSM` (conv 캐시 및 SSM 상태 포함) | 1e-2 |
| `testAxonSWA` | `AxonSWA` | 1e-3 |
| `testAxonMoE` | `AxonSharedExpertMoE` | 5e-2 |

MoE 허용 오차가 더 널그 이유는 전문가 라우팅과 게이팅이 PyTorch와 MLX 백엔드 간에 추가적인 부동소수점 차이를 도입하기 때문입니다.

### 테스트 실행

```bash
# 모든 동등성 테스트
cd BitAxonApp && swift test --filter EquivalenceTests

# 단일 테스트
cd BitAxonApp && swift test --filter EquivalenceTests/testAxonSSM
```

### 참조 데이터 재생성

Python 레이어를 변경한 후 참조 텐서를 재생성합니다:

```bash
cd BitAxonApp/Tests/EquivalenceTests/EquivalenceTestSupport
python export_reference.py
```

## 빌드 및 실행

### 명령어로 빠른 시작

```bash
# 클론하고 앱 디렉토리로 이동
cd bit-axon/BitAxonApp

# 의존성 해결
swift package resolve

# 빌드
swift build

# 실행 (SwiftUI 창이 열립니다)
.build/debug/BitAxonApp
```

### Xcode에서

1. `BitAxonApp.xcodeproj`를 엽니다 (또는 SPM 워크스페이스의 경우 `Package.swift`).
2. **My Mac** 대상을 선택합니다.
3. **Cmd+R**을 누릅니다.

### 모델 로딩

실행 시 앱은 "Load Model" 버튼이 있는 사이드바를 표시합니다. 탭하면 기본 설정으로 모델을 인스턴스화합니다. 특정 체크포인트를 로드하려면 모델 디렉토리를 앱에 드롭합니다 (이 기능은 디렉토리 루트에 `config.json`이 있어야 하는 `ModelService.loadFromDirectory`를 사용합니다).

### 성능 모니터링

사이드바에서 "Show Metrics"를 토글합니다. 툴바에 다음이 표시됩니다:

- **Tokens/sec**: 실시간 생성 처리량
- **TTFT**: 첫 토큰까지의 지연 시간 (밀리초)
- **GPU Used / Cache / Peak / Limit**: 메모리 사용량 (MB)
- **SoC Temp**: 다이 온도 (`powermetrics` 접근을 위해 `sudo` 필요)

### 파인튜닝

1. 사이드바에서 "Fine-Tune"을 클릭합니다.
2. JSONL 학습 파일을 드롭 존에 드래그하거나, 클릭하여 찾아봅니다.
3. 하이퍼파라미터를 조정합니다 (학습률, LoRA rank, 배치 크기 등).
4. "Start Training"을 클릭합니다.

앱은 `bit-axon train`을 실행하고 로그를 스트리밍합니다. `bit-axon` CLI가 설치되어 있어야 하며 (`pip install bit-axon`), `PATH`에서 찾을 수 있어야 합니다.
