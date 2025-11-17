# 파이프라인 실행 요약

## `run_all.py` 실행 시 자동으로 수행되는 작업

### ✅ Step 1: 학습 (Training)
- 모든 타겟에 대해 NHiTS 모델 학습
- 검증 메트릭 계산 (RMSE, MAE, MAPE)
- 전체 기간 R² 계산
- 모델 저장: `models/{target}/`
- 메트릭 저장: `results/{target}_val_metrics.json`, `results/{target}_full_r2.json`

### ✅ Step 2: 평가 (Evaluation)
- 학습된 모델로 검증 데이터 예측
- 검증 메트릭 계산 및 저장
- **자동으로 플롯 생성**: `results/{target}_forecast.png`
  - 검증 기간의 실제값 vs 예측값
  - 95% 신뢰구간 포함
  - 메트릭 정보 표시
- 상세 결과 저장: `results/{target}_validation.csv`

### ✅ Step 3: 예측 (Forecasting)
- 미래 기간 예측 생성 (horizon 개월)
- 예측 결과 저장: `results/{target}_forecast.csv`
- **자동으로 플롯 생성**: `results/{target}_future_forecast.png`
  - 전체 기간 (과거 + 미래)
  - 과거 실제값 (검은색)
  - 미래 예측값 (파란색)
  - 95% 신뢰구간 포함

### ✅ 최종 요약
- 모든 타겟의 메트릭 요약 테이블 출력
- 결과 저장 위치 안내

## 실행 방법

### 방법 1: 모듈 실행
```bash
python -m src.pipeline.run_all
```

### 방법 2: Import 실행 (Colab 권장)
```python
from src.pipeline.run_all import main
main()
```

### 방법 3: 스크립트 사용
```bash
python run_pipeline.py
```

## 생성되는 파일

### 모델 파일
- `models/{target}/scalers.pkl` - 스케일러 저장

### 결과 파일
- `results/{target}_val_metrics.json` - 검증 메트릭
- `results/{target}_full_r2.json` - 전체 기간 R²
- `results/{target}_validation.csv` - 검증 상세 결과
- `results/{target}_forecast.csv` - 미래 예측 결과
- `results/summary_metrics.json` - 전체 요약

### 플롯 파일 (자동 생성)
- `results/{target}_forecast.png` - 검증 기간 플롯 (평가 단계에서 생성)
- `results/{target}_future_forecast.png` - 미래 예측 플롯 (예측 단계에서 생성)

## 단계별 스킵 옵션

```bash
# 학습 스킵 (기존 모델 사용)
python -m src.pipeline.run_all --skip_training

# 평가 스킵
python -m src.pipeline.run_all --skip_evaluation

# 예측 스킵
python -m src.pipeline.run_all --skip_forecast
```

## 요약

**네, 맞습니다!** `run_all.py`를 실행하면:

1. ✅ **학습** 자동 실행
2. ✅ **평가** 자동 실행 (플롯 자동 생성)
3. ✅ **예측** 자동 실행 (플롯 자동 생성)
4. ✅ **최종 요약** 출력

모든 작업이 순차적으로 자동으로 수행되며, 플롯도 자동으로 생성됩니다!

