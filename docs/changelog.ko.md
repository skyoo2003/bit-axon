# Changelog

Bit-Axon 프로젝트의 모든 주요 변경 사항은 이 파일에 기록됩니다.

## [Unreleased]

### 추가
- CLI: WikiText-103 perplexity 벤치마크용 `evaluate` 명령
- CLI: Qwen2.5-3B에서 Bit-Axon 포맷 변환용 `port-weights` 명령
- CLI: 엔드투엔드 ML 워크플로용 `pipeline` 모듈 (SFT → merge → quantize → evaluate → inference → ORPO)
- CLI: 데이터셋 포맷 변환용 `prepare` 명령 (alpaca, messages, orpo)
- 평가: `WikiTextDataset`에서 커스텀 토크나이저 지원
- 테스트: 공개 API 스모크 테스트 및 CLI 명령어 테스트

## [0.1.0] - 2026-04-07

### 추가

#### 아키텍처 및 모델
- `BitAxonModel` — 세 가지 블록 변형이 있는 24층 샌드위치 아키텍처
- `BitAxonConfig` — 모델 설정 데이터클래스 (32억 파라미터, 32K 어휘, 65K 최대 컨텍스트)
- Axon-SSM (Mamba 스타일 상태 공간 모델), 토큰당 O(1) 메모리
- Shared-Expert MoE (8개 전문가, top-2 라우팅, 토큰당 ~14억 활성 파라미터)
- Sliding Window Attention (4K 윈도우)
- RMSNorm 레이어 및 KV 캐시 유틸리티

#### 학습
- QLoRA 지원이 있는 LoRA 및 DoRA 어댑터 레이어
- SFT, Alpaca, ORPO 데이터셋 클래스
- 열 인식 쿨링 스케줄러
- 웜업이 포함된 코사인 LR 스케줄러
- 어댑터 병합 및 safetensors 내보내기

#### 추론
- 스트리밍이 포함된 자기회귀 텍스트 생성
- Temperature, top-k, top-p 샘플링
- 인터랙티브 채팅 모드
- 로컬 경로 및 HuggingFace Hub에서 모델 로딩

#### 양자화
- NF4 (4비트 NormalFloat) 양자화

#### CLI
- `run`, `train`, `quantize`, `merge`, `benchmark`, `download` 명령

#### macOS 앱
- MLX-Swift 백엔드가 있는 SwiftUI 네이티브 채팅 애플리케이션

#### 인프라
- GitHub Actions CI, PyPI 배포, pre-commit 후크
