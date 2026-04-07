# 프로파일링 및 벤치마킹

Bit-Axon은 Apple Silicon에서 메모리, 속도, 온도 동작을 측정하는 내장 프로파일링 툴킷을 제공합니다. 빠른 결과를 위한 CLI 명령과 사용자 정의 워크플로우를 위한 Python API 두 가지 인터페이스로 동작합니다.

## CLI 벤치마크

성능 스냅샷을 얻는 가장 빠른 방법:

```bash
bit-axon benchmark --seq-lengths 128,512,1024,2048
```

이 명령은 네 가지 시퀀스 길이에서 전체 `BenchmarkSuite`를 실행하고, tok/sec, 지연 시간, 메모리 사용량, SoC 온도를 포함한 Rich 테이블을 출력합니다.

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|--------|---------|-------------|
| `--seq-lengths, -s` | `128,512,1024,2048` | 쉼표로 구분된 시퀀스 길이 |
| `--batch-size` | `1` | 순전파당 배치 크기 |
| `--warmup, -w` | `2` | 측정 제외 웜업 반복 횟수 |
| `--iterations, -i` | `5` | 시퀀스 길이당 측정 반복 횟수 |
| `--config-small` | `false` | 빠른 테스트를 위한 작은 설정 (4 레이어, 256 차원) 사용 |

### 예제

소형 모델로 빠른 스모크 테스트:

```bash
bit-axon benchmark --config-small --seq-lengths 32,64,128
```

안정적인 수치를 위해 더 많은 반복으로 전체 벤치마크:

```bash
bit-axon benchmark -s 128,256,512,1024,2048,4096 -i 10 -w 3
```

배치 추론 프로파일링:

```bash
bit-axon benchmark --batch-size 4 -s 256,1024
```

## Python API

네 가지 프로파일러 클래스는 모두 `bit_axon.profiling`에 있습니다.

### MemoryProfiler

MLX Metal API를 통해 GPU 메모리를 추적합니다. 세 가지 메트릭을 사용할 수 있습니다: active memory(현재 할당됨), peak memory(마지막 리셋 이후 최고점), cache memory(회수되었지만 OS에 아직 반환되지 않음).

```python
from bit_axon.profiling.memory import MemoryProfiler

profiler = MemoryProfiler()

# 모델 가중치 메모리
info = profiler.profile_model(model)
print(f"Weights: {info['weight_memory_gb']:.2f} GB, {info['param_count']:,} params")

# 시퀀스 길이 2048에서 순전파 중 메모리
profiler.reset_peak()
result = profiler.profile_forward(model, seq_len=2048)
print(f"Peak: {result['peak_memory_gb']:.2f} GB")
print(f"Activations: {result['activation_memory_gb']:.2f} GB")
```

`profile_forward` 메서드는 무작위 입력으로 단일 순전파를 실행하고 최대 및 활성 메모리를 보고합니다. 활성화 메모리는 최대 판독값에서 모델 가중치 메모리를 뺀 값으로 추정됩니다.

디바이스 정보도 확인할 수 있습니다.

```python
info = profiler.device_info()
print(info)  # {'architecture': 'apple-m4', ...}
```

### SpeedProfiler

두 가지 모드로 추론 처리량을 측정합니다: prefill(전체 프롬프트의 배치 처리)과 autoregressive(한 번에 하나의 토큰, 디코딩 시뮬레이션).

```python
from bit_axon.profiling.speed import SpeedProfiler

profiler = SpeedProfiler()

# Prefill 속도: 모델이 프롬프트를 얼마나 빨리 처리하는가?
result = profiler.benchmark_tokens_per_sec(
    model,
    seq_len=1024,
    num_warmup=2,
    num_iterations=5,
)
print(f"Prefill: {result['tokens_per_sec']:.0f} tok/s")
print(f"Latency: {result['latency_ms']:.1f} ms (±{result['std_latency_ms']:.1f})")

# Autoregressive 속도: 얼마나 빨리 생성하는가?
result = profiler.benchmark_autoregressive(
    model,
    total_tokens=64,
)
print(f"Decode: {result['tokens_per_sec']:.0f} tok/s")
print(f"Per-token: {result['mean_per_token_ms']:.1f} ms")
```

Prefill 벤치마크는 반복 간 평균 지연 시간과 표준 편차를 보고합니다. Autoregressive 벤치마크는 실제 생성을 시뮬레이션합니다. 모델이 KV cache를 사용하여 한 번에 하나씩 토큰을 생성하며, 결과는 지속적인 디코딩 처리량을 반영합니다.

### ThermalMonitor

`sudo powermetrics`를 통해 Apple Silicon SoC 다이 온도를 읽습니다. 일회성 읽기와 트렌드 감지가 포함된 연속 백그라운드 폴링을 모두 지원합니다.

```python
from bit_axon.profiling.thermal import ThermalMonitor

monitor = ThermalMonitor(poll_interval=1.0, history_size=60)

# 일회성 읽기
temp = monitor.get_soc_temperature()
print(f"SoC: {temp}°C" if temp else "Temperature unavailable")

# 연속 백그라운드 폴링 (학습 루프용)
monitor.start()
# ... 학습 실행 ...
print(f"Current: {monitor.temperature}°C")
print(f"History: {monitor.get_history()}")
print(f"Rising? {monitor.is_rising(window=5)}")
print(f"Above 95°C? {monitor.is_above(95.0)}")
monitor.stop()
```

!!! warning "sudo 필요"
    `ThermalMonitor`는 내부적으로 `sudo powermetrics`를 호출합니다. 터미널 세션에 sudo 권한이 있는지 확인하거나, 스크립트를 `sudo`로 실행하세요. sudo 없이는 온도 읽기가 `None`을 반환합니다.

`is_rising` 메서드는 마지막 N개 판독값에 대해 선형 회귀를 사용해 상승 트렌드를 감지합니다. 이는 온도 인식 학습에 유용합니다. 온도가 쓰로틀링 영역으로 향하고 있다면 배치 크기를 줄이거나 일시 정지를 삽입할 수 있습니다.

### BenchmarkSuite

세 프로파일러를 여러 시퀀스 길이에 걸쳐 단일 조율된 실행으로 결합합니다.

```python
from bit_axon.profiling.benchmark import BenchmarkSuite, BenchmarkResult

suite = BenchmarkSuite()  # 기본 BitAxonConfig 사용

results: BenchmarkResult = suite.benchmark_sequence_lengths(
    seq_lengths=[128, 512, 1024, 2048],
    batch_size=1,
    num_warmup=2,
    num_iterations=5,
)

# ASCII 테이블 출력
print(results.to_table())

# 개별 결과 접근
for name, metrics in results.results.items():
    print(f"{name}: {metrics['tokens_per_sec']} tok/s, "
          f"{metrics['peak_memory_gb']:.2f} GB peak")
```

`results.results`의 각 항목은 `seq_len`, `tokens_per_sec`, `latency_ms`, `peak_memory_gb`, `active_memory_gb`, `soc_temp_c`를 포함합니다.

## 결과 해석

### 속도

통합 메모리 16GB의 MacBook Air M4에서 대략 다음을 예상할 수 있습니다.

| 메트릭 | 일반적 범위 | 참고 |
|--------|---------------|-------|
| Prefill (128 토큰) | 800 ~ 2,000 tok/s | 짧은 시퀀스는 메모리 대역폭 제한 |
| Prefill (1024 토큰) | 300 ~ 800 tok/s | 긴 시퀀스는 오버헤드를 더 잘 분산 |
| Prefill (2048 토큰) | 200 ~ 600 tok/s | 연산 제한 영역에 근접 |
| Autoregressive 디코딩 | 30 ~ 80 tok/s | 단일 토큰 순전파는 토큰당 오버헤드가 더 큼 |

실제 수치는 Mac 모델(M2 vs M4, Air vs Pro), 다른 앱의 메모리 압력, 쓰로틀링 발생 여부에 따라 다릅니다. Pro 및 Max 칩은 냉각 시스템 덕분에 더 높은 처리량을 더 오래 유지합니다.

### 메모리

전체 Bit-Axon 설정(32억 파라미터, Q4 양자화)의 경우:

| 메트릭 | 예상 범위 |
|--------|---------------|
| 가중치 메모리 | ~1.7 ~ 1.8 GB |
| 4K 컨텍스트 전체 | ~2.4 ~ 2.6 GB |
| 64K 컨텍스트 전체 | ~2.8 ~ 3.2 GB |

활성 메모리가 최대 메모리보다 현저히 높으면 MLX 할당자가 캐시된 버퍼를 유지하고 있는 것입니다. 정확한 측정이 필요하면 실행 간에 `mx.clear_cache()`를 호출하세요.

### 온도

| 범위 | 의미 |
|-------|---------|
| 40 ~ 60 °C | 유휴 ~ 가벼운 부하. 짧은 벤치마크에 정상. |
| 60 ~ 80 °C | 중간 부하. 지속적인 추론 또는 소형 배치 학습. |
| 80 ~ 95 °C | 무거운 부하. 팬리스 Mac에서 온도 한계에 근접. |
| 95+ °C | 쓰로틀링 가능성. 클럭 속도 저하 및 처리량 감소 예상. |

팬리스 머신(MacBook Air)은 활성 냉각 모델(MacBook Pro, Mac Studio)보다 빠르게 온도 한계에 도달합니다. 학습 중 온도가 90°C 이상으로 올라가면 배치 크기를 줄이거나 스텝 간 냉각을 허용하도록 폴링 간격을 늘리세요.

## 팁

- **항상 웜업하세요.** MLX는 커널을 지연 컴파일합니다. 처음 몇 번의 순전파는 Metal 셰이더 컴파일 때문에 느립니다. 프로파일러는 `num_warmup`으로 이를 처리하지만, 사용자 정의 측정 코드를 작성할 때는 이를 고려하세요.
- **측정 전 동기화하세요.** MLX 연산은 비동기적입니다. 타이머를 시작하고 중지하기 전에 항상 `mx.synchronize()`를 호출하세요. 프로파일러는 내부적으로 이렇게 합니다.
- **벤치마크 전 다른 앱을 닫으세요.** GPU 메모리와 온도 여유는 시스템 전체에서 공유됩니다. 탭이 50개인 브라우저가 수치에 영향을 줍니다.
- **여러 반복을 실행하세요.** 단일 측정은 OS 스케줄링과 온도 상태 변화로 인해 노이즈가 많을 수 있습니다. 5회 이상의 반복이 안정적인 평균을 제공합니다.
- **반복 속도를 위해 `--config-small`을 사용하세요.** 실제 성능이 아닌 벤치마크 파이프라인 자체를 디버깅할 때, 작은 설정은 거의 즉시 로드되고 반복 시간을 1초 미만으로 유지합니다.
