# Thermal-Aware Training on Fanless Hardware

**상태**: :fontawesome-solid-circle-check:{ .green } 구현됨
**소스**: [`src/bit_axon/profiling/thermal.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/profiling/thermal.py), [`src/bit_axon/training/cooling.py`](https://github.com/skyoo2003/bit-axon/blob/main/src/bit_axon/training/cooling.py)

## 요약

팬 없는 Apple Silicon MacBook에서 언어 모델을 학습할 때 고유한 제약이 발생합니다. 지속적인 GPU 연산은 열적 스로틀링을 유발하여 처리량을 40–60% 저하시키거나 하드웨어 강제 종료를 초래할 수 있습니다. Bit-Axon은 백그라운드 열 모니터(`ThermalMonitor`)와 냉각 게이트 학습 스케줄러(`CoolingScheduler`)로 구성된 열 인식 학습 시스템으로 이 문제를 해결합니다. 모니터는 macOS `powermetrics`를 통해 SoC 다이 온도를 폴링하고, 스케줄러는 온도가 구성 가능한 임계값을 초과하면 학습 스텝을 일시 정지하여, MacBook Air M4에서 열적 스로틀링 없이 지속적인 다중 시간 미세조정을 가능하게 합니다.

## 핵심 기여

1. **백그라운드 열 폴링** — 데몬 스레드가 구성 가능한 간격으로 SoC 온도를 지속적으로 읽고, 롤링 이력 버퍼를 유지합니다.
2. **3단계 열 정책** — 배치 크기 감소, 학습 일시 정지, 긴급 종료를 위한 개별 임계값을 제공합니다.
3. **추세 감지** — 온도 이력 윈도우에 대한 선형 회귀로 SoC가 가열되고 있는지 판단하여 사전 예방적 스로틀링을 가능하게 합니다.
4. **비 macOS 환경에서 제로 오버헤드** — `powermetrics`를 사용할 수 없는 경우 자연스럽게 no-op으로 성능 저하됩니다.

## 수학적 기반

### 온도 샘플링 모델

샘플 $k$에서의 SoC 다이 온도를 $T_k$라고 합시다. 모니터는 롤링 윈도우를 유지합니다:

$$
\mathcal{H} = [T_{k - n + 1}, T_{k - n + 2}, \ldots, T_k]
$$

여기서 $n = |\mathcal{H}|$는 이력 버퍼 크기입니다 (기본값: 60개 샘플).

### 선형 회귀를 통한 추세 감지

SoC가 가열되고 있는지 판단하기 위해, 모니터는 마지막 $w$개 측정값에 대해 단순 선형 모델을 적합합니다:

$$
T_i = \alpha + \beta \cdot i + \varepsilon_i, \quad i \in \{k - w + 1, \ldots, k\}
$$

기울기 추정값은 다음과 같습니다:

$$
\hat{\beta} = \frac{\sum_{i=0}^{w-1}(i - \bar{i})(T_{k-w+1+i} - \bar{T})}{\sum_{i=0}^{w-1}(i - \bar{i})^2}
$$

여기서 $\bar{i} = (w-1)/2$이고 $\bar{T}$는 윈도우의 평균입니다. $\hat{\beta} > 0$일 때 SoC가 **상승** 추세에 있다고 판단합니다.

### 3단계 열 정책

`ThermalPolicy`는 세 개의 온도 임계값을 정의합니다:

$$
T_{\text{speed}} < T_{\text{pause}} < T_{\text{stop}}
$$

| 임계값 | 기본값 | 동작 |
|--------|--------|------|
| $T_{\text{speed}} = 75°C$ | `max_speed_temp` | 배치 크기 감소 신호 |
| $T_{\text{pause}} = 85°C$ | `pause_temp` | $\delta t$초 동안 학습 일시 정지 |
| $T_{\text{stop}} = 95°C$ | `stop_temp` | `ThermalShutdownError` 발생 |

### 일시 정지 루프

$T \geq T_{\text{pause}}$일 때, 스케줄러는 슬립 루프에 진입합니다:

$$
\text{while } T \geq T_{\text{pause}}: \quad \text{sleep}(\delta t), \quad t_{\text{total}} \mathrel{+}= \delta t
$$

여기서 $\delta t = 0.5$초입니다 (구성 가능). 온도가 $T_{\text{pause}}$ 미만으로 떨어지거나 측정값을 사용할 수 없게 되면 루프가 종료됩니다.

### 배치 크기 감소 신호

$T_{\text{speed}} \leq T < T_{\text{pause}}$일 때, 스케줄러는 배치 크기를 줄여야 함을 신호합니다:

$$
\text{reduce\_batch} = \begin{cases}
\text{True} & \text{if } T_{\text{speed}} \leq T < T_{\text{pause}} \\
\text{False} & \text{otherwise}
\end{cases}
$$

## Bit-Axon에서의 구현

### ThermalMonitor

| 구성 요소 | 설명 |
|-----------|------|
| `get_soc_temperature()` | `sudo powermetrics --samplers smc -i 1 -n 1`을 통해 다이 온도 읽기 |
| `start()` / `stop()` | 백그라운드 폴링을 위한 데몬 스레드 수명 주기 |
| `temperature` | 최신 측정값을 반환하는 스레드 안전 프로퍼티 |
| `get_history()` | 온도 버퍼의 스냅샷 반환 |
| `is_rising(window=5)` | 마지막 $w$개 측정값에 대한 선형 회귀 기울기 검사 |
| `is_above(threshold)` | 단순 임계값 비교 |

### CoolingScheduler

| 구성 요소 | 설명 |
|-----------|------|
| `check_before_step(step)` | 스텝 전 온도 게이트; 일시 정지 또는 종료 에러 발생 |
| `should_reduce_batch()` | 적응형 배치 크기 조정을 위한 신호 |
| `total_pause_time` | 학습 로그를 위한 누적 일시 정지 시간 |

### ThermalPolicy

```python
@dataclass
class ThermalPolicy:
    max_speed_temp: float = 75.0   # 배치 감소 신호
    pause_temp: float = 85.0       # 학습 일시 정지
    stop_temp: float = 95.0        # 긴급 종료
    pause_duration: float = 0.5    # 일시 정지 반복당 초
```

초기화 시 다음 불변 조건이 강제됩니다: $T_{\text{speed}} < T_{\text{pause}} < T_{\text{stop}}$.

### 학습 루프 통합

```python
from bit_axon.profiling.thermal import ThermalMonitor
from bit_axon.training.cooling import CoolingScheduler, ThermalPolicy

monitor = ThermalMonitor(poll_interval=1.0, history_size=60)
monitor.start()

scheduler = CoolingScheduler(monitor, ThermalPolicy())

for step in range(num_steps):
    scheduler.check_before_step(step)  # 온도가 너무 높으면 일시 정지 또는 에러 발생
    # ... 학습 스텝 ...
```

## 참고 문헌

- Apple Developer Documentation. *powermetrics — System Power Metrics*.
- Apple Machine Learning Research. *MLX: An array framework for machine learning on Apple silicon*.
- You, Y., et al. (2017). *Large Batch Training of Convolutional Networks*. arXiv:1708.03888. (관련: 하드웨어 제약 하의 학습률 스케일링.)
