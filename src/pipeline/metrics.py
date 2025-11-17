"""
Metrics computation module for evaluation.
"""
import numpy as np
import pandas as pd
from typing import Union


def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def mape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute Mean Absolute Percentage Error (as percentage).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE value (as percentage, e.g., 5.0 means 5%)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Avoid division by zero
    mask = (y_true != 0) & ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute coefficient of determination (R²).
    
    R² = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R² value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    ss_res = np.sum((y_true_masked - y_pred_masked) ** 2)
    ss_tot = np.sum((y_true_masked - np.mean(y_true_masked)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    return float(1 - (ss_res / ss_tot))


def compute_all_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> dict:
    """
    Compute all metrics: RMSE, MAE, MAPE, R².
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with all metrics
    """
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred)
    }

