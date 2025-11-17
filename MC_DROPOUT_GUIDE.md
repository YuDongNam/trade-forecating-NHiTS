# Monte Carlo Dropout 가이드

## 개요

NHITS 모델은 기본적으로 신뢰구간을 제공하지 않습니다. 따라서 **Monte Carlo Dropout** 방식을 사용하여 불확실성을 근사하고 95% 신뢰구간을 생성합니다.

## Monte Carlo Dropout 원리

1. **학습 단계**: 모델에 dropout 레이어가 포함되어 학습됩니다
2. **예측 단계**: 
   - 모델을 training 모드로 설정하여 dropout을 활성화
   - 동일한 입력에 대해 여러 번 예측 수행 (각 예측마다 다른 dropout 마스크 적용)
   - 예측 결과들의 분포에서 quantile을 계산하여 신뢰구간 생성

## 설정

### config/train.yaml

```yaml
dropout_rate: 0.1      # dropout rate (0.0 ~ 1.0)
mc_samples: 100        # Monte Carlo 샘플 수
```

- **dropout_rate**: 모델의 dropout 비율 (기본값: 0.1)
  - 높을수록 더 넓은 신뢰구간 (더 보수적)
  - 낮을수록 더 좁은 신뢰구간 (더 확신)
  
- **mc_samples**: 신뢰구간 계산을 위한 샘플 수 (기본값: 100)
  - 많을수록 더 정확하지만 느림
  - 적을수록 빠르지만 부정확할 수 있음

## 사용 방법

### 자동 사용

파이프라인을 실행하면 자동으로 Monte Carlo Dropout이 사용됩니다:

```python
from src.pipeline.run_all import main
main()
```

### 수동 사용

```python
from src.pipeline.mc_dropout import predict_with_mc_dropout
from neuralforecast import NeuralForecast

# 모델 학습 후
predictions = predict_with_mc_dropout(
    nf=nf,
    df=df,
    n_samples=100,
    level=95,
    model_name="NHITS"
)

# 결과에는 다음 컬럼이 포함됩니다:
# - NHITS: 평균 예측값
# - NHITS-lo-95: 95% 신뢰구간 하한
# - NHITS-hi-95: 95% 신뢰구간 상한
# - NHITS-std: 예측 표준편차
```

## 성능 고려사항

- **MC 샘플 수**: 100개 샘플은 일반적으로 충분하지만, 더 정확한 신뢰구간을 원하면 200-500으로 증가 가능
- **계산 시간**: 샘플 수에 비례하여 증가 (100 샘플 ≈ 예측 시간 × 100)
- **메모리**: 각 샘플의 예측 결과를 메모리에 저장하므로 horizon이 길면 메모리 사용량 증가

## 결과 해석

- **신뢰구간이 넓음**: 모델이 해당 시점의 예측에 대해 불확실성이 높음
- **신뢰구간이 좁음**: 모델이 해당 시점의 예측에 대해 확신이 높음
- **실제값이 신뢰구간 밖**: 모델이 예상하지 못한 패턴이나 외부 충격 가능성

## 주의사항

1. **Dropout이 없는 모델**: dropout_rate=0이면 Monte Carlo Dropout이 작동하지 않습니다
2. **일관성**: 같은 입력에 대해 여러 번 예측하면 약간씩 다른 결과가 나옵니다 (정상 동작)
3. **계산 비용**: MC 샘플 수가 많을수록 예측 시간이 길어집니다

