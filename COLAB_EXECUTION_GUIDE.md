# 코랩 실행 가이드 (Colab Execution Guide)

이 가이드는 리팩터링된 NHiTS 파이프라인을 Google Colab에서 실행하는 방법을 설명합니다.

## 📋 목차

1. [초기 설정](#초기-설정)
2. [전체 파이프라인 실행](#전체-파이프라인-실행)
3. [개별 기능 실행](#개별-기능-실행)
   - [학습 (Training)](#1-학습-training)
   - [평가 (Evaluation)](#2-평가-evaluation)
   - [예측 (Forecasting)](#3-예측-forecasting)
4. [결과 확인](#결과-확인)

---

## 초기 설정

### 1. 데이터 업로드 및 환경 설정

```python
# 코랩 셀에서 실행
from google.colab import files
import zipfile
from pathlib import Path

# 프로젝트 파일 업로드 (또는 Git clone)
# 데이터 파일들을 data/ 폴더에 업로드
# - 8개 타겟 CSV 파일
# - 6개 외생변수 CSV 파일

# 필요한 디렉토리 생성
Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("config").mkdir(exist_ok=True)
```

### 2. tqdm 설정 (진행바 표시 개선)

```python
# tqdm을 notebook 모드로 설정
from tqdm.notebook import tqdm
import tqdm as tqdm_module
tqdm_module.tqdm = tqdm
```

---

## 전체 파이프라인 실행

**가장 간단한 방법**: 모든 단계를 한 번에 실행

```python
# 방법 1: import 방식 (권장 - tqdm 출력이 깔끔함)
from src.pipeline.run_all import main
main()
```

```python
# 방법 2: 모듈 실행 방식
!python -m src.pipeline.run_all --config_dir config
```

**실행되는 단계:**
1. **학습 (Training)**: 모든 타겟에 대해 모델 학습
   - 결정론적 예측으로 validation metrics 계산 (RMSE, MAE, MAPE)
   - R²는 계산하지 않음
2. **평가 (Evaluation)**: 
   - Standard validation: RMSE, MAE, MAPE
   - Historical forecast: Rolling backtest + R² 계산
3. **예측 (Forecasting)**: 미래 예측 생성 (MC Dropout으로 신뢰구간 포함)

---

## 개별 기능 실행

### 1. 학습 (Training)

**목적**: 모델 학습 및 기본 validation metrics 계산

```python
# 방법 1: import 방식 (권장)
from src.pipeline.train import main
main()
```

```python
# 방법 2: 모듈 실행
!python -m src.pipeline.train --config_dir config
```

**결과:**
- `models/{target}/`: 학습된 모델 체크포인트
- `results/{target}_val_metrics.json`: Validation metrics (RMSE, MAE, MAPE만)
- **R²는 계산하지 않음**

**설정 변경:**
```python
# validation 기간 변경
from src.pipeline.train import train_target
from src.config.yaml_loader import (
    load_train_config, load_validation_config, 
    load_exogenous_config, load_paths_config
)
from pathlib import Path

config_dir = Path("config")
train_config = load_train_config(config_dir)
validation_config = load_validation_config(config_dir)
validation_config.mode = "tail"
validation_config.tail_months = 24  # 마지막 24개월을 validation으로
exog_config = load_exogenous_config(config_dir)
paths_config = load_paths_config(config_dir)

# 단일 타겟 학습
result = train_target("Korea_Import", train_config, validation_config, exog_config, paths_config)
```

---

### 2. 평가 (Evaluation)

**목적**: 모델 성능 평가 + Historical Forecast (Rolling Backtest)

```python
# 방법 1: import 방식 (권장)
from src.pipeline.evaluate import main
main()
```

```python
# 방법 2: 모듈 실행
!python -m src.pipeline.evaluate --config_dir config
```

**실행되는 작업:**

#### 2-1. Standard Validation
- 결정론적 예측으로 validation metrics 계산
- **MC Dropout 사용 안 함** (빠른 평가)
- Metrics: RMSE, MAE, MAPE만

#### 2-2. Historical Forecast (Rolling Backtest)
- Rolling-origin forecast 수행
- **MC Dropout 사용** (불확실성 추정)
- Metrics: RMSE, MAE, MAPE, **R²** (여기서만 R² 계산!)

**결과 파일:**
- `results/{target}_validation.csv`: Standard validation 결과
- `results/{target}_forecast.png`: Standard validation 플롯
- `results/{target}_val_metrics.json`: Standard validation metrics
- `results/{target}_historical_forecast.csv`: Historical forecast 결과 (신뢰구간 포함)
- `results/{target}_historical_forecast.png`: Historical forecast 플롯 (R² 포함)
- `results/{target}_historical_metrics.json`: Historical forecast metrics (R² 포함)

**설정 변경:**
```python
# validation 기간 변경
from src.pipeline.evaluate import evaluate_target
from src.config.yaml_loader import (
    load_train_config, load_validation_config, 
    load_exogenous_config, load_paths_config, load_uncertainty_config
)
from pathlib import Path

config_dir = Path("config")
train_config = load_train_config(config_dir)
validation_config = load_validation_config(config_dir)
validation_config.mode = "range"
validation_config.start = "2023-01-01"
validation_config.end = "2024-12-01"
exog_config = load_exogenous_config(config_dir)
paths_config = load_paths_config(config_dir)
uncertainty_config = load_uncertainty_config(config_dir)

# 단일 타겟 평가
result = evaluate_target("Korea_Import", train_config, validation_config, exog_config, paths_config)
```

**MC Dropout 설정 변경:**
```yaml
# config/uncertainty.yaml 수정
method: "mc_dropout"
enabled: true          # false로 설정하면 deterministic만 사용
n_samples: 100        # 샘플 수 (더 많을수록 정확하지만 느림)
ci_level: 0.95        # 신뢰구간 레벨
```

---

### 3. 예측 (Forecasting)

**목적**: 미래 예측 생성 (MC Dropout으로 신뢰구간 포함)

```python
# 방법 1: import 방식 (권장)
from src.pipeline.forecast import main
main()
```

```python
# 방법 2: 모듈 실행
!python -m src.pipeline.forecast --config_dir config
```

**결과:**
- `results/{target}_forecast.csv`: 미래 예측값 (신뢰구간 포함)
- `results/{target}_future_forecast.png`: 예측 플롯

**MC Dropout 사용 여부:**
- `config/uncertainty.yaml`의 `enabled: true/false`로 제어
- `enabled: true` → MC Dropout으로 신뢰구간 생성
- `enabled: false` → 결정론적 예측만

---

## 결과 확인

### 1. Metrics 확인

```python
import json
from pathlib import Path

# Standard validation metrics
with open("results/Korea_Import_val_metrics.json", "r") as f:
    val_metrics = json.load(f)
print("Validation Metrics:", val_metrics)
# 출력: {"RMSE": ..., "MAE": ..., "MAPE": ...}  (R² 없음)

# Historical forecast metrics (R² 포함!)
with open("results/Korea_Import_historical_metrics.json", "r") as f:
    hist_metrics = json.load(f)
print("Historical Forecast Metrics:", hist_metrics)
# 출력: {"target": "Korea_Import", "rmse": ..., "mae": ..., "mape": ..., "r2": ...}
```

### 2. 플롯 확인

```python
from IPython.display import Image, display

# Standard validation plot
display(Image("results/Korea_Import_forecast.png"))

# Historical forecast plot (R² 포함)
display(Image("results/Korea_Import_historical_forecast.png"))
```

### 3. CSV 결과 확인

```python
import pandas as pd

# Standard validation
val_df = pd.read_csv("results/Korea_Import_validation.csv")
print(val_df.head())

# Historical forecast (신뢰구간 포함)
hist_df = pd.read_csv("results/Korea_Import_historical_forecast.csv")
print(hist_df.head())
# 컬럼: ds, y, y_hat, y_hat_lower, y_hat_upper, error, abs_error
```

---

## 주요 변경사항 요약

### 이전 vs 현재

| 기능 | 이전 | 현재 |
|------|------|------|
| **학습 시 MC Dropout** | 사용함 (느림) | 사용 안 함 (빠름) |
| **검증 시 MC Dropout** | 사용함 (느림) | 사용 안 함 (빠름) |
| **예측 시 MC Dropout** | 사용함 | 설정으로 제어 가능 |
| **R² 계산 위치** | 모든 곳 | Historical Forecast만 |
| **Historical Forecast** | 없음 | 있음 (Rolling Backtest) |

### 실행 흐름

```
전체 파이프라인 (run_all.py)
├── 1. Training
│   ├── 모델 학습 (결정론적)
│   └── Validation metrics (RMSE, MAE, MAPE만)
│
├── 2. Evaluation
│   ├── Standard Validation (결정론적, 빠름)
│   │   └── Metrics: RMSE, MAE, MAPE
│   └── Historical Forecast (MC Dropout, 느림)
│       └── Metrics: RMSE, MAE, MAPE, R²
│
└── 3. Forecasting
    └── 미래 예측 (MC Dropout, 신뢰구간)
```

---

## 문제 해결

### MC Dropout이 너무 느린 경우

```yaml
# config/uncertainty.yaml
n_samples: 50  # 100 → 50으로 줄이기
```

또는

```yaml
enabled: false  # MC Dropout 완전히 비활성화
```

### R²를 보고 싶은 경우

**Historical Forecast를 실행해야 합니다:**

```python
from src.pipeline.evaluate import main
main()  # 이 함수가 자동으로 Historical Forecast도 실행함
```

### 특정 타겟만 실행

```python
from src.pipeline.train import train_target
from src.config.yaml_loader import (
    load_train_config, load_validation_config,
    load_exogenous_config, load_paths_config
)
from pathlib import Path

config_dir = Path("config")
train_config = load_train_config(config_dir)
validation_config = load_validation_config(config_dir)
exog_config = load_exogenous_config(config_dir)
paths_config = load_paths_config(config_dir)

# 단일 타겟만
result = train_target("Korea_Import", train_config, validation_config, exog_config, paths_config)
```

---

## 빠른 시작 예제

```python
# 1. tqdm 설정
from tqdm.notebook import tqdm
import tqdm as tqdm_module
tqdm_module.tqdm = tqdm

# 2. 전체 파이프라인 실행
from src.pipeline.run_all import main
main()

# 3. 결과 확인
import json
with open("results/Korea_Import_historical_metrics.json", "r") as f:
    metrics = json.load(f)
    print(f"R²: {metrics['r2']:.4f}")
```

---

## 참고

- **Training/Validation**: 빠른 결정론적 예측 사용
- **Historical Forecast**: MC Dropout 사용 (불확실성 추정)
- **Forecasting**: MC Dropout 사용 (신뢰구간 생성)
- **R²**: Historical Forecast에서만 계산

