# CLI 레퍼런스

Bit-Axon은 [Typer](https://typer.tiangolo.com/) 기반의 단일 `bit-axon` 진입점을 제공합니다. 모든 하위 명령은 `typer.Argument` 및 `typer.Option` 어노테이션으로 타입이 지정되므로, `bit-axon --help`와 `bit-axon <command> --help`는 항상 최신 시그니처를 반영합니다.

---

## 추론

### `bit-axon run`

프롬프트(또는 stdin)에서 텍스트를 생성합니다.

```bash
bit-axon run "Explain entropy in one sentence"
bit-axon run --chat
echo "What is 2+2?" | bit-axon run
```

**사용법**

```
bit-axon run [PROMPT] [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--model` / `-m` | `str` | `skyoo2003/bit-axon` | 모델 식별자 (Hugging Face 저장소 또는 로컬 경로) |
| `--tokenizer` / `-t` | `str` | `None` | tokenizer 재정의 (기본값은 모델 자체의 tokenizer) |
| `--max-tokens` | `int` | `512` | 생성할 최대 토큰 수 |
| `--temperature` | `float` | `0.6` | Sampling temperature |
| `--top-k` | `int` | `50` | Top-K 필터링 |
| `--top-p` | `float` | `0.95` | Nucleus (top-p) 필터링 임계값 |
| `--seed` | `int` | `None` | 재현 가능한 출력을 위한 난수 시드 |
| `--chat` / `-c` | `bool` | `False` | 대화형 채팅 세션 시작 |
| `--no-stream` | `bool` | `False` | 토큰을 스트리밍하지 않고 전체 응답을 한 번에 출력 |
| `--config-small` | `bool` | `False` | 소형 모델 설정 사용 |

---

## 학습

### `bit-axon train`

JSONL 데이터셋으로 LoRA를 사용해 모델을 파인튜닝합니다.

```bash
bit-axon train data/train.jsonl -w skyoo2003/bit-axon -o checkpoints/my-run
```

**사용법**

```
bit-axon train DATA [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--model-weights` / `-w` | `str` | *필수* | 파인튜닝할 베이스 모델 가중치 |
| `--val-data` | `str` | `None` | 검증 데이터셋 경로 |
| `--tokenizer` / `-t` | `str` | `Qwen/Qwen2.5-3B` | Tokenizer 식별자 |
| `--lora-rank` | `int` | `8` | LoRA 랭크 |
| `--lora-dropout` | `float` | `0.0` | LoRA dropout 확률 |
| `--lora-scale` | `float` | `20.0` | LoRA 스케일링 인자 |
| `--no-dora` | `bool` | `False` | DoRA 비활성화 (대신 표준 LoRA 사용) |
| `--learning-rate` / `-lr` | `float` | `1e-4` | 최대 학습률 |
| `--max-steps` | `int` | `10000` | 최대 학습 스텝 수 |
| `--batch-size` | `int` | `1` | 디바이스당 배치 크기 |
| `--grad-accum-steps` | `int` | `4` | 기울기 누적 스텝 수 |
| `--max-seq-len` | `int` | `2048` | 최대 시퀀스 길이 |
| `--warmup-steps` | `int` | `100` | 선형 웜업 스텝 수 |
| `--max-grad-norm` | `float` | `1.0` | 기울기 클리핑 노름 |
| `--seed` | `int` | `42` | 난수 시드 |
| `--no-thermal` | `bool` | `False` | 온도 관리 비활성화 |
| `--temp-pause` | `float` | `85.0` | 학습이 일시 정지되는 온도 (°C) |
| `--temp-stop` | `float` | `95.0` | 학습이 중지되는 온도 (°C) |
| `--output-dir` / `-o` | `str` | `checkpoints` | 체크포인트 저장 디렉토리 |
| `--save-every` | `int` | `500` | N 스텝마다 체크포인트 저장 |
| `--eval-every` | `int` | `500` | N 스텝마다 평가 실행 |
| `--resume` | `bool` | `False` | 최신 체크포인트에서 재개 |
| `--config-small` | `bool` | `False` | 소형 모델 설정 사용 |

**학습 파이프라인 (10단계)**

1. 베이스 모델 가중치와 tokenizer를 로드합니다.
2. 대상 모듈에 LoRA (또는 DoRA) 어댑터를 적용합니다.
3. 학습 데이터셋을 로드하고 토큰화합니다.
4. 구성된 학습률과 웜업 스케줄로 옵티마이저를 설정합니다.
5. 더 큰 유효 배치 크기를 시뮬레이션하도록 기울기 누적을 구성합니다.
6. GPU가 구성된 임계값을 초과하면 학습을 일시 정지하거나 중지하도록 온도 모니터링을 선택적으로 활성화합니다.
7. 학습 루프를 실행하고, 구성된 간격으로 평가 및 체크포인트 저장을 수행합니다.
8. 중단되거나 완료되면 최종 체크포인트를 저장합니다.
9. 매 스텝마다 학습 메트릭 (loss, 학습률, 처리량)을 로깅합니다.
10. 종료하고 최적 또는 최신 체크포인트의 경로를 보고합니다.

---

## 모델 관리

### `bit-axon quantize`

모델을 더 낮은 정밀도(예: 4-bit 정수)로 양자화합니다.

```bash
bit-axon quantize skyoo2003/bit-axon -o models/bit-axon-q4 -b 4 -g 64
```

**사용법**

```
bit-axon quantize MODEL_PATH [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--output` / `-o` | `str` | `""` | 양자화된 모델의 출력 디렉토리 |
| `--bits` / `-b` | `int` | `4` | 양자화 비트 폭 |
| `--group-size` / `-g` | `int` | `64` | 그룹 양자화의 그룹 크기 |
| `--config-small` | `bool` | `False` | 소형 모델 설정 사용 |

### `bit-axon merge`

LoRA 어댑터를 베이스 모델에 병합하고, 선택적으로 결과를 재양자화합니다.

```bash
bit-axon merge skyoo2003/bit-axon -a checkpoints/my-run/adapter -o models/merged
```

**사용법**

```
bit-axon merge BASE_MODEL [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--adapter` / `-a` | `str` | *필수* | 병합할 LoRA 어댑터 경로 |
| `--output` / `-o` | `str` | `""` | 병합된 모델의 출력 디렉토리 |
| `--no-re-quantize` | `bool` | `False` | 병합 후 재양자화 생략 |
| `--bits` / `-b` | `int` | `4` | 재양자화 시 비트 폭 |
| `--group-size` / `-g` | `int` | `64` | 재양자화 시 그룹 크기 |
| `--lora-rank` / `-r` | `int` | `8` | 어댑터의 LoRA 랭크 |

### `bit-axon download`

Hugging Face에서 모델(또는 데이터셋)을 다운로드합니다.

```bash
bit-axon download skyoo2003/bit-axon -d models/bit-axon
```

**사용법**

```
bit-axon download [REPO_ID] [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--local-dir` / `-d` | `str` | `None` | 파일을 저장할 로컬 디렉토리 |
| `--include` | `list[str]` | `None` | 포함할 파일의 glob 패턴 |

### `bit-axon upload`

모델을 Hugging Face Hub에 업로드합니다.

```bash
bit-axon upload models/merged -r skyoo2003/bit-axon -t Qwen/Qwen2.5-3B
```

**사용법**

```
bit-axon upload MODEL_PATH [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--repo-id` / `-r` | `str` | `skyoo2003/bit-axon` | Hugging Face 저장소 ID |
| `--tokenizer` / `-t` | `str` | `Qwen/Qwen2.5-3B` | Tokenizer 이름 또는 경로 |
| `--private` | `bool` | `False` | 비공개 저장소로 생성 |
| `--commit-message` / `-m` | `str` | `Upload Bit-Axon 3.2B model` | 업로드 커밋 메시지 |
| `--benchmark-results` | `str` | `None` | 쉼표로 구분된 벤치마크 결과, 예: `mmlu=0.45,gsm8k=0.32` |

### `bit-axon port-weights`

모델 가중치를 Bit-Axon 형식으로 변환합니다.

```bash
bit-axon port-weights models/bit-axon-ported
```

**사용법**

```
bit-axon port-weights OUTPUT [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--config-small` | `bool` | `False` | 소형 모델 설정 사용 |

---

## 평가

### `bit-axon benchmark`

여러 시퀀스 길이에서 생성 처리량을 측정합니다.

```bash
bit-axon benchmark -s "128,512,1024,2048" -i 10
```

**사용법**

```
bit-axon benchmark [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--seq-lengths` / `-s` | `str` | `128,512,1024,2048` | 벤치마크할 시퀀스 길이 (쉼표로 구분) |
| `--batch-size` | `int` | `1` | 각 벤치마크 실행의 배치 크기 |
| `--warmup` / `-w` | `int` | `2` | 웜업 반복 횟수 (측정에서 제외) |
| `--iterations` / `-i` | `int` | `5` | 시퀀스 길이당 측정 반복 횟수 |
| `--config-small` | `bool` | `False` | 소형 모델 설정 사용 |

### `bit-axon evaluate`

모델에서 평가를 실행하고 집계 메트릭을 출력합니다.

```bash
bit-axon evaluate models/bit-axon-q4 -t Qwen/Qwen2.5-3B
```

**사용법**

```
bit-axon evaluate MODEL_PATH [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--max-tokens` | `int` | `100000` | 전체 평가 실행의 토큰 예산 |
| `--seq-length` | `int` | `2048` | 최대 시퀀스 길이 |
| `--tokenizer` / `-t` | `str` | `None` | Tokenizer 재정의 |
| `--batch-size` | `int` | `4` | 평가 배치 크기 |
| `--config-small` | `bool` | `False` | 소형 모델 설정 사용 |

### `bit-axon pipeline`

내장 데이터셋에서 엔드투엔드 학습 및 정렬 파이프라인을 실행합니다.

```bash
bit-axon pipeline -o pipeline_output --max-steps 200
```

**사용법**

```
bit-axon pipeline [OPTIONS]
```

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--output-dir` / `-o` | `str` | `pipeline_output` | 최상위 출력 디렉토리 |
| `--max-steps` | `int` | `100` | 최대 SFT 학습 스텝 수 |
| `--orpo-steps` | `int` | `50` | 최대 ORPO 정렬 스텝 수 |
| `--max-seq-len` | `int` | `32` | 최대 시퀀스 길이 |
| `--lora-rank` | `int` | `8` | SFT 및 ORPO 단계 모두에 적용할 LoRA 랭크 |
| `--batch-size` | `int` | `1` | 디바이스당 배치 크기 |

**파이프라인 단계 (7단계)**

1. 내장 학습 데이터셋을 다운로드(또는 검증)합니다.
2. 지도 파인튜닝용 데이터를 전처리하고 토큰화합니다.
3. LoRA를 적용해 구성된 스텝 수만큼 SFT(지도 파인튜닝)를 실행합니다.
4. SFT 체크포인트에서 선호도 쌍을 생성합니다.
5. 선호도 쌍에 대해 ORPO(승산비 선호도 최적화)를 실행합니다.
6. 최종 어댑터 가중치를 베이스 모델에 병합합니다.
7. 완성된 모델과 요약 보고서를 출력 디렉토리에 저장합니다.
