# 2024년 검증 가이드

이 가이드는 2023년까지의 데이터로 학습하고 2024년 1년치 예측값을 생성하여 검증하는 방법을 설명합니다.

## 실행 방법

### 방법 1: 명령줄에서 실행

```bash
# 2024년 검증 실행
python -m src.pipeline.evaluate 2024
```

### 방법 2: Python 스크립트에서 실행

```python
from src.pipeline.evaluate import main_2024

# 2024년 검증 실행
metrics = main_2024()
```

### 방법 3: 개별 타겟 검증

```python
from src.pipeline.evaluate import evaluate_2024
from src.config import TrainConfig, ExogenousConfig

train_config = TrainConfig()
exog_config = ExogenousConfig()

# 특정 타겟만 검증
metrics = evaluate_2024("Korea_Import", train_config, exog_config)
print(f"RMSE: {metrics['RMSE']:.3f}")
print(f"MAE: {metrics['MAE']:.3f}")
print(f"MAPE: {metrics['MAPE']:.3f}%")
```

## 출력 파일

검증 실행 후 다음 파일들이 생성됩니다:

### 1. 메트릭 파일
- `results/{target}_2024_metrics.json`: 각 타겟별 RMSE, MAE, MAPE 지표
- `results/summary_2024_metrics.json`: 모든 타겟의 요약 지표

### 2. 시각화
- `results/{target}_2024_validation.png`: 2024년 실제값 vs 예측값 비교 그래프

### 3. 상세 결과
- `results/{target}_2024_validation.csv`: 월별 실제값, 예측값, 오차 등 상세 데이터

## 검증 프로세스

1. **데이터 분할**
   - 학습 데이터: 2010-01-01 ~ 2023-12-01
   - 검증 데이터: 2024-01-01 ~ 2024-12-01

2. **모델 학습**
   - 2023년까지의 데이터로 NHITS 모델 학습

3. **예측 생성**
   - 2024년 12개월 예측값 생성

4. **평가 지표 계산**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - MAPE (Mean Absolute Percentage Error)

## 예상 출력 예시

```
============================================================
2024 VALIDATION - Training on 2023 data, Predicting 2024
============================================================

============================================================
Evaluating Korea_Import - Training on 2023 data, Predicting 2024
============================================================
Loading data...
Splitting data: train <= 2023-12-01, validation = 2024...
Train period: 2010-01-01 to 2023-12-01
Train shape: (168, 9)
Validation period: 2024-01-01 to 2024-12-01
Validation shape: (12, 9)
Creating NHITS model with 7 exogenous variables...
Training model on 2023 data...
Generating predictions for 2024...

============================================================
Validation Metrics for Korea_Import (2024):
  RMSE: 12345.678
  MAE:  9876.543
  MAPE: 5.432%
============================================================
```

## 주의사항

- 2024년 데이터가 없는 경우 해당 타겟은 건너뜁니다
- 검증 데이터가 12개월 미만인 경우 사용 가능한 월수만큼만 검증합니다
- 외생변수(관세, 환율, WTI, 구리)는 2024년 데이터도 필요합니다

