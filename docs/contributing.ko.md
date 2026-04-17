# 기여 가이드

Bit-Axon에 기여해 주셔서 감사합니다. 이 프로젝트는 Apple Silicon을 위해 설계된 오픈소스 소형 언어 모델 엔진이며, 커뮤니티의 패치, 버그 수정, 새로운 기능을 환영합니다.

참여하기 전에 [행동 강령](https://github.com/skyoo2003/bit-axon/blob/main/CODE_OF_CONDUCT.md)을 읽어주세요.

## 사전 요구 사항

- Apple Silicon (M1 이상) 기반 **macOS**
- **Python 3.10+**
- **Apple MLX 0.31.0+** (의존성으로 자동 설치됨)
- **Git**

최신 버전의 Xcode Command Line Tools도 권장됩니다 (`xcode-select --install`).

## 개발 환경 설정

```bash
# 리포지토리 클론
git clone https://github.com/skyoo2003/bit-axon.git
cd bit-axon

# 가상 환경 생성
python -m venv .venv
source .venv/bin/activate

# 편집 가능한 모드로 개발 의존성과 함께 설치
pip install -e ".[dev]"

# pre-commit 후크 설치
pre-commit install

# 설치 확인
python -c "import bit_axon; print(bit_axon.__version__)"
```

`[dev]` extras는 `pytest`, `pytest-xdist`, `ruff`, `pre-commit`을 설치합니다. 프로젝트 구성은 모두 `pyproject.toml`에 있습니다.

!!! tip
    pre-commit 후크는 모든 커밋 시 자동으로 실행됩니다. 후행 공백, 파일 끝 빈 줄 누락, YAML/TOML 구문 오류, Ruff 린트 문제를 CI에 도달하기 전에 잡아냅니다.

## 개발 명령어

| 명령어 | 설명 |
| --------------------------------- | ----------------------------------- |
| `pytest tests/` | 전체 테스트 스위트 실행 |
| `pytest -n auto tests/` | 병렬 테스트 실행 |
| `pytest tests/test_model.py` | 단일 테스트 파일 실행 |
| `pytest -k TestAxonSSM` | 이름 패턴과 일치하는 테스트 실행 |
| `ruff check src/ tests/` | 오류 및 스타일 문제 린트 |
| `ruff check src/ tests/ --fix` | 가능한 경우 린트 문제 자동 수정 |
| `ruff format src/ tests/` | 코드 포맷팅 |
| `ruff format --check src/ tests/` | 쓰기 없이 포맷팅 확인 |

## 프로젝트 구조

```
bit-axon/
├── src/bit_axon/           # 패키지 소스
│   ├── config.py           # 모델 설정
│   ├── model.py            # 최상위 모델 정의
│   ├── layers/             # 모델 레이어 (SSM, block, MoE, norms, attention)
│   ├── quantization/       # 양자화 방식 (NF4, ternary, TurboQuant)
│   ├── training/           # 학습 어댑터 (LoRA, DoRA)
│   └── utils/              # 유틸리티 (KV 캐시, 헬퍼)
├── tests/                  # 테스트 스위트, src 구조와 대응
│   ├── conftest.py         # 공유 픽스처
│   ├── test_config.py
│   ├── test_model.py
│   └── ...
└── docs/                   # MkDocs Material 문서
```

## MLX 관련 규칙

Bit-Axon은 PyTorch가 아닌 Apple의 MLX 프레임워크로 구축되었습니다. 이는 오버헤드를 최소화하면서 Apple Silicon에서 네이티브로 실행하기 위한 의도적인 선택입니다.

!!! warning
    코드베이스 어디에서도 PyTorch를 가져오거나 의존하지 마세요. 모든 텐서 연산은 MLX의 `mx.array`를 사용합니다.

### PyTorch와의 주요 차이점

**순전파에는 `__call__`을 사용하세요, `forward()`가 아닙니다.** MLX 모듈은 `__call__`을 기본 진입점으로 사용합니다. 별도의 `forward` 메서드 정의를 피하세요.

**`mx.eval()`을 명시적으로 호출하세요.** MLX는 지연 평가(lazy evaluation)를 사용합니다. 구체적인 값이 필요할 때(예: 테스트나 Python으로 반환할 때) 결과에 `mx.eval()`을 호출하세요.

**`nn.Module`을 상속하세요.** 모든 모델 컴포넌트는 `mlx.nn.Module`을 상속합니다. 파라미터와 버퍼는 `mx.array` 속성을 직접 할당하여 등록됩니다.

```python
class MyLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.zeros((dim, dim))  # 파라미터로 자동 등록

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight
```

!!! info
    새로운 레이어나 모델 컴포넌트를 작성할 때는 항상 모든 텐서 타입에 `mx.array`를 사용하세요. 여기에는 파라미터, 활성화, 중간 계산이 모두 포함됩니다. `torch.Tensor`나 `jax.Array`를 참조하지 마세요.

## 코드 스타일

린팅과 포맷팅에 **Ruff**를 사용합니다. 다른 린터나 포매터는 필요하지 않습니다.

### 포맷팅

Ruff는 다음 규칙을 적용합니다 (`pyproject.toml`에 설정됨):

- 줄 길이: **160자**
- 모든 문자열에 **이중 따옴표**
- 들여쓰기에 공백 사용 (탭 불가)
- 대상 Python 버전: 3.10

### 린팅

활성화된 규칙 세트: E, W, F, I, N, UP, B, SIM, C4, DTZ, RUF. 일부 특정 규칙은 의도적으로 무시됩니다 (E501, B008, N802, N803). 1순위 임포트(`bit_axon`)가 맨 위로 정렬됩니다.

### 타입 힌트

모든 공개 함수 시그니처와 클래스 속성에 타입 힌트를 사용하세요. 전방 참조에는 `from __future__ import annotations`을 권장합니다.

### 독스트링

모든 공개 클래스와 함수에 **Google 스타일 독스트링**을 사용하세요:

```python
def compute_attention_scores(query: mx.array, key: mx.array, scale: float) -> mx.array:
    """Compute scaled dot-product attention scores.

    Args:
        query: Query tensor of shape (batch, heads, seq_len, dim).
        key: Key tensor of shape (batch, heads, seq_len, dim).
        scale: Scaling factor applied before softmax.

    Returns:
        Attention weights of shape (batch, heads, seq_len, seq_len).
    """
```

### 임포트 순서

Ruff의 `I` 규칙이 이를 자동으로 처리합니다. 1순위(`bit_axon`) 임포트가 3순위 임포트보다 앞에 옵니다.

!!! tip
    커밋 전에 `ruff check src/ tests/ --fix`와 `ruff format src/ tests/`를 실행하세요. pre-commit 후크도 이 작업을 하지만, 로컬에서 실행하면 문제를 더 빠르게 수정할 수 있습니다.

## 테스트

테스트는 `tests/`에 있으며 **`src/bit_axon/` 디렉토리 구조를 미러링합니다**. `src/bit_axon/layers/my_layer.py`에 새 모듈을 추가했다면, 테스트는 `tests/layers/test_my_layer.py`에 작성하세요. 공유 픽스처는 `tests/conftest.py`에 넣습니다.

```bash
# 모두 실행
pytest tests/

# 병렬 실행 (모든 코어 사용)
pytest -n auto tests/

# 특정 테스트 파일 실행
pytest tests/test_model.py

# 이름 패턴과 일치하는 테스트 실행
pytest -k TestAxonSSM
```

!!! tip
    새 테스트를 작성할 때는 기존 테스트 스위트의 패턴을 따르세요. 공유 픽스처를 추가한다면 `tests/conftest.py`에 넣어 모든 테스트 모듈에서 사용할 수 있게 하세요.

!!! note
    모든 테스트 파일은 pytest가 발견할 수 있어야 합니다. 파일 이름이 `test_*.py`이고, 테스트 함수 이름이 `test_*`이거나 `Test*`로 명명된 클래스 내의 메서드인지 확인하세요.

## CLI 개발

CLI는 [Typer](https://typer.tiangolo.com/)와 [Rich](https://rich.readthedocs.io/)로 구축되었습니다.

### 새 CLI 명령어 추가

1. `src/bit_axon/cli/<command>.py`에 명령어 로직을 구현하는 함수를 생성합니다.
2. `src/bit_axon/cli/main.py`에서 `@app.command()`를 사용하여 등록합니다:

```python
@app.command()
def mycommand(...):
    from bit_axon.cli.mycommand import mycommand_impl
    mycommand_impl(...)
```

3. `typer.testing.CliRunner`를 사용하여 `tests/cli/test_<command>.py`에 테스트를 추가합니다.
4. `--help`를 위해 MLX를 로드하지 않도록 모든 임포트는 지연(lazy)되어야 합니다 (함수 내부).

### CLI 규칙

- 실제 모델 없이 테스트하려면 `--config-small` 플래그를 사용합니다.
- 출력에는 Rich 콘솔을 사용합니다 (스피너, 진행률 표시줄, 테이블).
- 명령어 함수 내부에서 모든 `bit_axon` 모듈을 지연 임포트합니다.

!!! warning
    CLI 파일의 모듈 레벨에서 MLX나 `bit_axon` 모델 코드를 임포트하지 마세요. MLX 종속 코드의 모든 임포트는 지연되어야 합니다 (함수 본문 내부). 그래야 MLX가 설치되지 않은 환경에서도 `bit-axon --help`가 작동합니다.

## 문서 기여

문서는 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)로 구축되며 `docs/` 디렉토리에 있습니다.

### 로컬 미리보기

```bash
mkdocs serve
```

`http://localhost:8000`에서 로컬 개발 서버가 시작됩니다. markdown 파일 변경은 새로고침 시 즉시 반영됩니다.

### 편집 가이드라인

- 영어로 작성하세요. 소스 markdown 파일에 한국어를 작성하지 마세요.
- Material 경고문 (`!!! tip`, `!!! warning`, `!!! note`, `!!! info`)을 콜아웃에 사용하세요.
- 적절한 제목 계층을 사용하세요. 레벨을 건너뛰지 마세요 (예: `##`에서 `####`로 바로 넘어가지 마세요).
- 언어 태그가 있는 코드 블록을 추가하세요 (`` ```python ``, `` ```bash `` 등).

### 국제화 (한국어 번역)

Bit-Axon은 영어와 함께 한국어 문서를 지원합니다. i18n 워크플로우는 **파일 접미사 규칙**을 사용합니다:

- 영어 소스: `docs/some-page/index.md`
- 한국어 번역: `docs/some-page/index.ko.md`

한국어 번역을 추가하거나 업데이트하려면 영어 소스와 함께 `.ko.md` 파일을 생성하거나 편집하세요. 빌드 시스템이 두 버전을 자동으로 인식합니다.

!!! tip
    영어 소스 파일만 직접 편집하세요. 한국어 번역은 대응하는 `.ko.md` 파일에 작성합니다. 단일 파일 내에서 언어를 혼합하지 마세요.

## 커밋 메시지

[Conventional Commits](https://www.conventionalcommits.org/)를 따릅니다. 형식:

```
<type>(<scope>): <description>
```

**유형**: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`

**사용 중인 범위**: `layers`, `model`, `training`, `quantization`, `utils`, `ci`

프로젝트 히스토리의 예시:

```
feat(layers): add selective scan wrapper for AxonSSM
feat(model): implement sparse MoE router with top-k gating
feat(quantization): add TurboQuant mixed-precision quantization
fix(training): correct LoRA gradient accumulation for batched inputs
docs: update README with new benchmark results
chore(ci): add parallel test execution to GitHub Actions
```

!!! info
    pre-commit 후크가 커밋 메시지 형식을 자동으로 검증합니다. 커밋 메시지가 conventional commits 패턴과 일치하지 않으면, 후크가 제안과 함께 거부합니다.

## Pull Request 프로세스

1. 리포지토리를 **포크**하고 `main`에서 기능 브랜치를 생성합니다.
2. 위 스타일 가이드에 따라 **변경 사항을 적용**합니다.
3. 새로운 기능에 대한 **테스트를 작성**합니다. 테스트 파일은 `tests/`에, `src/bit_axon/` 구조를 미러링하여 배치합니다. 공유 픽스처는 `tests/conftest.py`에 추가합니다.
4. 푸시 전에 **로컬에서 린팅과 테스트를 실행**합니다:

    ```bash
    ruff check src/ tests/ && ruff format --check src/ tests/
    pytest tests/
    ```

5. 변경 사항과 동기에 대한 명확한 설명과 함께 **PR을 엽니다**.
6. **AI 지원 기여**: PR의 일부가 AI 도구로 생성되었거나 AI 도구의 도움을 받았다면, PR 설명에 이를 표기해 주세요. 어떤 도구인지나 프롬프트를 공개할 필요는 없으며, 리뷰어가 알 수 있도록 표시만 하면 됩니다.
7. **리뷰 피드백을 반영**하고 동일한 브랜치에 업데이트를 푸시합니다. 승인되면 PR이 병합됩니다.

## 버그 보고

버그를 발견하셨나요? 명확한 제목과 함께 이슈를 열고 다음을 포함해 주세요:

- 최소 재현 가능 예제
- Python 버전, macOS 버전, MLX 버전
- 기대한 동작과 실제 동작
- 관련 로그나 스택 트레이스

버그인지 확신이 서지 않아도 이슈를 열어 주세요. 놓치는 것보다 분류하는 것을 선호합니다.

## 라이선스

Bit-Axon은 [Apache 2.0 라이선스](https://github.com/skyoo2003/bit-axon/blob/main/LICENSE)로 배포됩니다. 기여함으로써 기여물이 동일한 조건으로 라이선스됨에 동의하는 것입니다.
