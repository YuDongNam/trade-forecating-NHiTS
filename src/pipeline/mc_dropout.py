"""
Monte Carlo Dropout for uncertainty quantification and confidence intervals.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from neuralforecast import NeuralForecast
from typing import Optional, Tuple


def enable_dropout_in_model(model):
    """
    Enable dropout layers in the model for Monte Carlo Dropout.
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()  # Keep dropout active during inference


def predict_with_mc_dropout(
    nf: NeuralForecast,
    df: pd.DataFrame,
    n_samples: int = 100,
    level: int = 95,
    model_name: str = "NHITS"
) -> pd.DataFrame:
    """
    Generate predictions using Monte Carlo Dropout for uncertainty quantification.
    
    Monte Carlo Dropout 방식:
    1. 모델을 train 모드로 설정하여 dropout 활성화
    2. 여러 번 예측 수행 (각 예측마다 다른 dropout 마스크 적용)
    3. 예측 결과들의 분포에서 quantile 계산하여 신뢰구간 생성
    
    Args:
        nf: Trained NeuralForecast object
        df: Input dataframe for prediction
        n_samples: Number of Monte Carlo samples (default: 100)
        level: Confidence level (default: 95)
        model_name: Name of the model in NeuralForecast
    
    Returns:
        DataFrame with predictions and confidence intervals
    """
    # Get the model from NeuralForecast
    model = nf.models[0] if len(nf.models) > 0 else None
    if model is None:
        raise ValueError("No model found in NeuralForecast object")
    
    # Get base prediction structure (for index/columns) - use eval mode first
    base_pred = nf.predict(df=df)
    
    # Find and enable dropout in the model's underlying PyTorch model
    # neuralforecast wraps the PyTorch model, so we need to access it through the wrapper
    pytorch_model = None
    
    # Try different ways to access the underlying PyTorch model
    if hasattr(model, 'model'):
        pytorch_model = model.model
    elif hasattr(model, 'net'):
        pytorch_model = model.net
    elif hasattr(model, 'network'):
        pytorch_model = model.network
    elif hasattr(model, '_model'):
        pytorch_model = model._model
    else:
        # Try to find PyTorch model by checking attributes
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                attr = getattr(model, attr_name)
                if isinstance(attr, torch.nn.Module):
                    pytorch_model = attr
                    break
    
    if pytorch_model is None:
        # Last resort: use the model wrapper itself
        pytorch_model = model
    
    # Store original training state
    original_training = pytorch_model.training if hasattr(pytorch_model, 'training') else False
    
    # Set model to training mode to enable dropout during inference
    if hasattr(pytorch_model, 'train'):
        pytorch_model.train()
    
    # Explicitly enable all dropout layers
    enable_dropout_in_model(pytorch_model)
    
    # Generate multiple predictions with dropout active
    all_predictions = []
    print(f"Generating {n_samples} Monte Carlo samples with dropout...")
    
    try:
        for i in range(n_samples):
            if (i + 1) % 20 == 0:
                print(f"  Sample {i+1}/{n_samples}...")
            
            # Ensure dropout is still enabled (neuralforecast might reset it)
            if hasattr(pytorch_model, 'train'):
                pytorch_model.train()
            enable_dropout_in_model(pytorch_model)
            
            # Each prediction will have different dropout masks
            pred = nf.predict(df=df)
            if model_name in pred.columns:
                all_predictions.append(pred[model_name].values)
            else:
                # Fallback: use first column
                all_predictions.append(pred.iloc[:, 0].values)
    finally:
        # Restore original training mode
        if hasattr(pytorch_model, 'train'):
            if original_training:
                pytorch_model.train()
            else:
                pytorch_model.eval()
    
    # Stack predictions: shape (n_samples, horizon)
    predictions_array = np.array(all_predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions_array, axis=0)
    std_pred = np.std(predictions_array, axis=0)
    
    # Calculate quantiles for confidence interval
    alpha = (100 - level) / 100
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    
    lower_bound = np.quantile(predictions_array, lower_quantile, axis=0)
    upper_bound = np.quantile(predictions_array, upper_quantile, axis=0)
    
    # Create result dataframe
    result = base_pred.copy()
    result[f"{model_name}"] = mean_pred
    result[f"{model_name}-lo-{level}"] = lower_bound
    result[f"{model_name}-hi-{level}"] = upper_bound
    result[f"{model_name}-std"] = std_pred
    
    print(f"Monte Carlo Dropout completed. Mean std: {np.mean(std_pred):.4f}")
    
    return result

