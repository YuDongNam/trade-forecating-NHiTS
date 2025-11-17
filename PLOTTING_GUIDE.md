# 플롯 생성 가이드

NHiTS 예측 결과를 시각화하는 여러 방법을 설명합니다.

## 방법 1: 평가 스크립트 실행 시 자동 생성 (가장 간단)

평가 스크립트를 실행하면 자동으로 플롯이 생성됩니다:

```bash
python -m src.pipeline.evaluate
```

또는 import 방식:
```python
from src.pipeline.evaluate import main
main()
```

**생성되는 파일:**
- `results/{target}_forecast.png`: 검증 기간 플롯
- `results/{target}_validation.csv`: 상세 검증 결과 (플롯 데이터 포함)

## 방법 2: plot_results.py 스크립트 사용

### 특정 타겟 플롯

```bash
# 검증 기간만
python plot_results.py --target Korea_Import

# 전체 기간 (학습 + 검증)
python plot_results.py --target Korea_Import --full_period
```

### 모든 타겟 플롯

```bash
# 모든 타겟의 검증 기간 플롯
python plot_results.py

# 모든 타겟의 전체 기간 플롯
python plot_results.py --full_period
```

### Python 코드에서 사용

```python
from plot_results import plot_validation_results, plot_full_period

# 검증 기간만
plot_validation_results("Korea_Import")

# 전체 기간
plot_full_period("Korea_Import")
```

## 방법 3: plotting 모듈 직접 사용

### 기본 플롯 (검증 기간)

```python
import pandas as pd
from pathlib import Path
from src.pipeline.plotting import plot_forecast

# 결과 로드
result_dir = Path("results")
val_df = pd.read_csv(result_dir / "Korea_Import_validation.csv")
val_df["ds"] = pd.to_datetime(val_df["ds"])

# 플롯 생성
plot_forecast(
    dates=pd.Series(val_df["ds"]),
    y_actual=pd.Series(val_df["y_actual"]),
    y_pred=pd.Series(val_df["y_pred"]),
    y_lower=pd.Series(val_df["y_pred_lower_95"]) if "y_pred_lower_95" in val_df.columns else None,
    y_upper=pd.Series(val_df["y_pred_upper_95"]) if "y_pred_upper_95" in val_df.columns else None,
    title="Korea_Import",
    metrics_text="RMSE: 123.45, MAE: 98.76, MAPE: 5.2%",
    show=True
)
```

### 전체 기간 플롯

```python
import pandas as pd
from pathlib import Path
from src.pipeline.plotting import plot_full_period_with_validation
from src.data import load_target_df
from src.config.yaml_loader import load_paths_config, load_exogenous_config

# 설정 로드
config_dir = Path("config")
paths_config = load_paths_config(config_dir)
exog_config = load_exogenous_config(config_dir)

# 데이터 로드
target = "Korea_Import"
full_df = load_target_df(target, Path(paths_config.raw_data_dir), exog_config)
full_df["ds"] = pd.to_datetime(full_df["ds"])

# 검증 결과 로드
val_df = pd.read_csv(Path(paths_config.result_dir) / f"{target}_validation.csv")
val_df["ds"] = pd.to_datetime(val_df["ds"])

# 학습/검증 분할
val_start = val_df["ds"].min()
train_df = full_df[full_df["ds"] < val_start]
val_actual_df = full_df[full_df["ds"] >= val_start]

# 플롯 생성
plot_full_period_with_validation(
    train_dates=pd.Series(train_df["ds"]),
    train_actual=pd.Series(train_df["y"]),
    val_dates=pd.Series(val_df["ds"]),
    val_actual=pd.Series(val_actual_df["y"]),
    val_pred=pd.Series(val_df["y_pred"]),
    val_lower=pd.Series(val_df["y_pred_lower_95"]) if "y_pred_lower_95" in val_df.columns else None,
    val_upper=pd.Series(val_df["y_pred_upper_95"]) if "y_pred_upper_95" in val_df.columns else None,
    target_name=target,
    metrics_text="RMSE: 123.45, MAE: 98.76, MAPE: 5.2%",
    show=True
)
```

## 방법 4: Colab에서 사용

### 간단한 방법

```python
# 평가 실행 (자동으로 플롯 생성)
from src.pipeline.evaluate import main
main()

# 또는 저장된 결과 플롯
from plot_results import plot_all_targets
plot_all_targets()
```

### 전체 기간 플롯 (Colab 노트북)

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config.yaml_loader import load_paths_config, load_targets_config
from src.data import load_target_df
from src.config.yaml_loader import load_exogenous_config

# 설정
config_dir = Path("config")
paths_config = load_paths_config(config_dir)
targets_config = load_targets_config(config_dir)
exog_config = load_exogenous_config(config_dir)
result_dir = Path(paths_config.result_dir)

# 각 타겟 플롯
for target in targets_config.targets:
    validation_file = result_dir / f"{target}_validation.csv"
    if not validation_file.exists():
        continue
    
    val_df = pd.read_csv(validation_file)
    val_df["ds"] = pd.to_datetime(val_df["ds"])
    
    # 전체 데이터 로드
    full_df = load_target_df(target, Path(paths_config.raw_data_dir), exog_config)
    full_df["ds"] = pd.to_datetime(full_df["ds"])
    
    # 학습/검증 분할
    val_start = val_df["ds"].min()
    train_df = full_df[full_df["ds"] < val_start]
    val_actual_df = full_df[full_df["ds"] >= val_start]
    
    # 플롯
    plt.figure(figsize=(14, 6))
    
    # 학습 데이터
    plt.plot(train_df["ds"], train_df["y"], 
             color="gray", alpha=0.6, linewidth=1.5,
             label="Training Data", linestyle="-")
    
    # 검증 실제값
    plt.plot(val_actual_df["ds"], val_actual_df["y"],
             color="black", linewidth=2.5, marker="o", markersize=6,
             label="Validation Actual", linestyle="-")
    
    # 신뢰구간
    if "y_pred_lower_95" in val_df.columns:
        plt.fill_between(
            val_df["ds"], val_df["y_pred_lower_95"], val_df["y_pred_upper_95"],
            color="gold", alpha=0.3, label="95% Confidence Interval"
        )
    
    # 예측값
    plt.plot(val_df["ds"], val_df["y_pred"],
             color="blue", linewidth=2, marker="s", markersize=6,
             label="Forecast", linestyle="--")
    
    # 구분선
    plt.axvline(x=val_start, color="red", linestyle=":", 
                linewidth=2, alpha=0.7, label="Train/Val Split")
    
    plt.title(f"{target} - Full Period", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Trade Balance", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

## 플롯 종류

### 1. 검증 기간 플롯 (`plot_forecast`)
- 검증 기간의 실제값 vs 예측값
- 95% 신뢰구간 (있는 경우)
- 메트릭 정보 (RMSE, MAE, MAPE, R²)

### 2. 전체 기간 플롯 (`plot_full_period_with_validation`)
- 학습 기간 실제값 (회색)
- 검증 기간 실제값 (검은색)
- 검증 기간 예측값 (파란색)
- 95% 신뢰구간 (노란색)
- 학습/검증 구분선 (빨간 점선)

## 저장 위치

- **자동 생성**: `results/{target}_forecast.png`
- **전체 기간**: `results/{target}_full_period.png` (plot_results.py 사용 시)
- **커스텀**: `save_path` 파라미터로 지정 가능

## 주의사항

1. **평가 먼저 실행**: 플롯을 그리기 전에 `evaluate.py`를 실행하여 결과 파일을 생성해야 합니다.
2. **파일 존재 확인**: `results/{target}_validation.csv` 파일이 있어야 합니다.
3. **Colab 환경**: `plt.show()` 대신 자동으로 표시되므로 `show=False`로 설정할 수도 있습니다.

