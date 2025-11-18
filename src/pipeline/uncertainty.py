"""
Uncertainty quantification module for prediction-time uncertainty estimation.

This module provides Monte Carlo Dropout for generating prediction intervals.
MC Dropout is used ONLY at prediction time, NOT during training or validation.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Optional
from neuralforecast import NeuralForecast


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """
    Enable dropout layers for Monte Carlo Dropout inference.
    
    This function:
    1. Sets the model to eval() mode FIRST (to disable batchnorm noise, etc.)
    2. Then iterates through all submodules and sets ONLY dropout layers
       to train() mode
    3. This ensures only dropout remains stochastic at inference
    
    Args:
        model: PyTorch model (the underlying model from neuralforecast)
    """
    # First, set model to eval mode (disables batchnorm, etc.)
    model.eval()
    
    # Then, enable ONLY dropout layers
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()  # Enable dropout during inference


def find_pytorch_model(nf: NeuralForecast) -> torch.nn.Module:
    """
    Find the underlying PyTorch model from a NeuralForecast object.
    
    Args:
        nf: NeuralForecast object
    
    Returns:
        The underlying PyTorch model
    """
    if len(nf.models) == 0:
        raise ValueError("No model found in NeuralForecast object")
    
    model_wrapper = nf.models[0]
    
    # Try different ways to access the underlying PyTorch model
    if hasattr(model_wrapper, 'model'):
        return model_wrapper.model
    elif hasattr(model_wrapper, 'net'):
        return model_wrapper.net
    elif hasattr(model_wrapper, 'network'):
        return model_wrapper.network
    elif hasattr(model_wrapper, '_model'):
        return model_wrapper._model
    else:
        # Try to find PyTorch model by checking attributes
        for attr_name in dir(model_wrapper):
            if not attr_name.startswith('_'):
                attr = getattr(model_wrapper, attr_name)
                if isinstance(attr, torch.nn.Module):
                    return attr
        
        # Last resort: use the wrapper itself
        return model_wrapper


def check_dropout_exists(model: torch.nn.Module) -> bool:
    """
    Check if the model contains any dropout layers.
    
    Args:
        model: PyTorch model
    
    Returns:
        True if dropout layers exist, False otherwise
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            return True
    return False


def mc_dropout_predict(
    nf: NeuralForecast,
    df: pd.DataFrame,
    n_samples: int = 100,
    ci_level: float = 0.95,
    model_name: str = "NHITS",
    target_scaler=None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate predictions with uncertainty using Monte Carlo Dropout.
    
    This function:
    1. Enables dropout in the model (only dropout, rest stays in eval mode)
    2. Runs multiple forward passes (n_samples times) on the same input
    3. Computes mean, std, and confidence intervals from the samples
    4. Returns predictions in the ORIGINAL SCALE (after inverse-scaling)
    
    Args:
        nf: Trained NeuralForecast object
        df: Input dataframe for prediction
        n_samples: Number of Monte Carlo samples
        ci_level: Confidence interval level (e.g., 0.95 for 95%)
        model_name: Name of the model in NeuralForecast
        target_scaler: Scaler for inverse transformation (if None, assumes already in original scale)
    
    Returns:
        Tuple of (y_hat, y_hat_lower, y_hat_upper) in original scale
        If uncertainty cannot be computed, y_hat_lower and y_hat_upper are None
    """
    # Find the underlying PyTorch model
    pytorch_model = find_pytorch_model(nf)
    
    # Check if dropout exists
    if not check_dropout_exists(pytorch_model):
        print("Warning: Model has no dropout layers. MC Dropout will produce degenerate intervals.")
        print("Returning deterministic prediction only.")
        # Return deterministic prediction
        pred = nf.predict(df=df)
        y_hat = pred[model_name].values if model_name in pred.columns else pred.iloc[:, 0].values
        if target_scaler is not None:
            y_hat = target_scaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()
        return y_hat, None, None
    
    # Store original training state
    original_training = pytorch_model.training
    
    # Enable MC Dropout (eval mode + dropout layers in train mode)
    enable_mc_dropout(pytorch_model)
    
    # Verify dropout is enabled
    dropout_enabled = False
    for module in pytorch_model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            if module.training:
                dropout_enabled = True
                break
    
    if not dropout_enabled:
        print("Warning: Failed to enable dropout. Returning deterministic prediction.")
        pytorch_model.train(original_training)
        pred = nf.predict(df=df)
        y_hat = pred[model_name].values if model_name in pred.columns else pred.iloc[:, 0].values
        if target_scaler is not None:
            y_hat = target_scaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()
        return y_hat, None, None
    
    # Generate multiple predictions with dropout active
    all_predictions_scaled = []
    print(f"Generating {n_samples} Monte Carlo samples with dropout...")
    
    try:
        for i in range(n_samples):
            if (i + 1) % 20 == 0:
                print(f"  Sample {i+1}/{n_samples}...")
            
            # Ensure dropout is still enabled (neuralforecast might reset it)
            enable_mc_dropout(pytorch_model)
            
            # Each prediction will have different dropout masks
            pred = nf.predict(df=df)
            if model_name in pred.columns:
                all_predictions_scaled.append(pred[model_name].values)
            else:
                # Fallback: use first column
                all_predictions_scaled.append(pred.iloc[:, 0].values)
    finally:
        # Restore original training mode
        pytorch_model.train(original_training)
    
    # Stack predictions: shape (n_samples, horizon)
    predictions_array = np.array(all_predictions_scaled)
    
    # Compute statistics in SCALED space
    mean_scaled = np.mean(predictions_array, axis=0)
    std_scaled = np.std(predictions_array, axis=0)
    
    # Check for degenerate intervals (variance too small)
    mean_std = np.mean(std_scaled)
    if mean_std < 1e-6:
        print(f"Warning: Very small variance (mean std: {mean_std:.2e}). Intervals may be degenerate.")
        print("This suggests dropout is not effectively active or dropout rate is too low.")
    
    # Compute confidence intervals using normal approximation
    z_score = 1.96 if ci_level == 0.95 else 2.576 if ci_level == 0.99 else 1.645  # Approximate
    # More accurate: use scipy.stats.norm.ppf if available
    try:
        from scipy.stats import norm
        z_score = norm.ppf((1 + ci_level) / 2)
    except ImportError:
        pass
    
    lower_scaled = mean_scaled - z_score * std_scaled
    upper_scaled = mean_scaled + z_score * std_scaled
    
    # Inverse transform to original scale
    if target_scaler is not None:
        # Transform mean and bounds together to preserve relationships
        mean_original = target_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        lower_original = target_scaler.inverse_transform(lower_scaled.reshape(-1, 1)).flatten()
        upper_original = target_scaler.inverse_transform(upper_scaled.reshape(-1, 1)).flatten()
    else:
        mean_original = mean_scaled
        lower_original = lower_scaled
        upper_original = upper_scaled
    
    print(f"MC Dropout completed. Mean std (scaled): {mean_std:.4f}")
    
    return mean_original, lower_original, upper_original

