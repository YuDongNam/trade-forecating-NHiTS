"""
Historical forecast / rolling backtest pipeline.

Implements darts-style historical_forecast behavior for NHiTS models.
Performs rolling-origin forecasts and computes metrics including R².
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from neuralforecast import NeuralForecast

from ..config.yaml_loader import (
    TrainConfig,
    ValidationConfig,
    UncertaintyConfig,
    PathsConfig,
)
from .metrics import compute_all_metrics, rmse, mae, mape, r2
from .uncertainty import mc_dropout_predict
from .plotting import plot_forecast


def historical_forecast(
    target: str,
    train_config: TrainConfig,
    validation_config: ValidationConfig,
    uncertainty_config: UncertaintyConfig,
    paths_config: PathsConfig,
    nf: NeuralForecast,
    train_nf: pd.DataFrame,
    full_df: pd.DataFrame,
    target_scaler,
    exog_cols: list
) -> dict:
    """
    Perform rolling-origin historical forecast (backtest).
    
    This function:
    1. Uses the validation window from validation_config
    2. For each time step in the window, uses preceding input_size months
       to predict one step ahead
    3. Optionally uses MC Dropout for uncertainty estimation
    4. Computes RMSE, MAE, MAPE, and R² on the historical forecast window
    
    Args:
        target: Target name
        train_config: Training configuration
        validation_config: Validation configuration (defines evaluation window)
        uncertainty_config: Uncertainty configuration (MC Dropout settings)
        paths_config: Paths configuration
        nf: Trained NeuralForecast object
        train_nf: Training data in NeuralForecast format (scaled)
        full_df: Full dataset (unscaled, for getting actual values)
        target_scaler: Scaler for inverse transformation
        exog_cols: List of exogenous column names
    
    Returns:
        Dictionary with metrics and forecast results
    """
    print(f"\n{'='*60}")
    print(f"Historical Forecast (Rolling Backtest) for: {target}")
    print(f"{'='*60}")
    
    # 1) Determine evaluation window
    full_df = full_df.sort_values("ds").reset_index(drop=True)
    full_df["ds"] = pd.to_datetime(full_df["ds"])
    
    if validation_config.mode == "tail":
        if validation_config.tail_months is None:
            raise ValueError("tail_months must be specified when mode is 'tail'")
        split_idx = len(full_df) - validation_config.tail_months
        eval_start_idx = split_idx
        eval_end_idx = len(full_df)
    elif validation_config.mode == "range":
        if validation_config.start is None or validation_config.end is None:
            raise ValueError("start and end must be specified when mode is 'range'")
        start_date = pd.to_datetime(validation_config.start)
        end_date = pd.to_datetime(validation_config.end)
        eval_mask = (full_df["ds"] >= start_date) & (full_df["ds"] <= end_date)
        eval_start_idx = eval_mask.idxmax() if eval_mask.any() else len(full_df)
        eval_end_idx = len(full_df)
    else:
        raise ValueError(f"Invalid validation mode for historical_forecast: {validation_config.mode}")
    
    if eval_start_idx < train_config.input_size:
        raise ValueError(f"Evaluation window starts too early. Need at least {train_config.input_size} months of history.")
    
    print(f"Evaluation window: {full_df.iloc[eval_start_idx]['ds']} to {full_df.iloc[eval_end_idx-1]['ds']}")
    print(f"Number of forecast points: {eval_end_idx - eval_start_idx}")
    
    # 2) Perform rolling-origin forecast
    all_forecasts = []
    all_actuals = []
    all_dates = []
    all_lower = []
    all_upper = []
    
    # Prepare scaled full dataset for rolling predictions
    full_df_scaled = full_df.copy()
    # We'll use the scaler that was fitted on training data
    full_df_scaled["y"] = target_scaler.transform(full_df[["y"]]).flatten()
    if exog_cols:
        # Note: exog_scaler should be passed separately, but for simplicity
        # we assume exog_cols are already scaled in train_nf format
        pass
    
    # Create full NeuralForecast format dataframe
    full_nf = full_df_scaled.copy()
    full_nf["unique_id"] = target
    full_nf = full_nf[["unique_id", "ds", "y"] + exog_cols]
    
    print("Performing rolling-origin forecasts...")
    for t_idx in range(eval_start_idx, eval_end_idx):
        # Get input window: [t_idx - input_size, t_idx)
        input_start = max(0, t_idx - train_config.input_size)
        input_end = t_idx
        
        # Extract input window
        input_window = full_nf.iloc[input_start:input_end].copy()
        
        if len(input_window) < train_config.input_size:
            # Not enough history, skip
            continue
        
        # Get actual value at time t
        actual_t = full_df.iloc[t_idx]["y"]
        date_t = full_df.iloc[t_idx]["ds"]
        
        # Predict one step ahead
        if uncertainty_config.enabled and uncertainty_config.method == "mc_dropout":
            # Use MC Dropout for uncertainty
            try:
                y_hat, y_lower, y_upper = mc_dropout_predict(
                    nf=nf,
                    df=input_window,
                    n_samples=uncertainty_config.n_samples,
                    ci_level=uncertainty_config.ci_level,
                    model_name="NHITS",
                    target_scaler=target_scaler
                )
                # Take first value (one-step ahead)
                y_hat_val = y_hat[0] if len(y_hat) > 0 else np.nan
                y_lower_val = y_lower[0] if y_lower is not None and len(y_lower) > 0 else np.nan
                y_upper_val = y_upper[0] if y_upper is not None and len(y_upper) > 0 else np.nan
            except Exception as e:
                print(f"Warning: MC Dropout failed at {date_t}: {e}. Using deterministic prediction.")
                pred = nf.predict(df=input_window)
                y_hat_scaled = pred["NHITS"].values[0] if "NHITS" in pred.columns else pred.iloc[0, 0]
                y_hat_val = target_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).flatten()[0]
                y_lower_val = np.nan
                y_upper_val = np.nan
        else:
            # Deterministic prediction
            pred = nf.predict(df=input_window)
            y_hat_scaled = pred["NHITS"].values[0] if "NHITS" in pred.columns else pred.iloc[0, 0]
            y_hat_val = target_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).flatten()[0]
            y_lower_val = np.nan
            y_upper_val = np.nan
        
        all_forecasts.append(y_hat_val)
        all_actuals.append(actual_t)
        all_dates.append(date_t)
        all_lower.append(y_lower_val)
        all_upper.append(y_upper_val)
        
        if (t_idx - eval_start_idx + 1) % 10 == 0:
            print(f"  Forecasted {t_idx - eval_start_idx + 1}/{eval_end_idx - eval_start_idx} points...")
    
    # 3) Convert to arrays
    y_hat_array = np.array(all_forecasts)
    y_actual_array = np.array(all_actuals)
    y_lower_array = np.array(all_lower) if all_lower else None
    y_upper_array = np.array(all_upper) if all_upper else None
    
    # Remove NaN values for metric computation
    valid_mask = ~(np.isnan(y_hat_array) | np.isnan(y_actual_array))
    if valid_mask.sum() == 0:
        print("Warning: No valid forecasts generated.")
        return None
    
    y_hat_valid = y_hat_array[valid_mask]
    y_actual_valid = y_actual_array[valid_mask]
    
    # 4) Compute metrics (including R²)
    metrics = compute_all_metrics(y_actual_valid, y_hat_valid, include_r2=True)
    print(f"\nHistorical Forecast Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.3f}")
    print(f"  MAE: {metrics['MAE']:.3f}")
    print(f"  MAPE: {metrics['MAPE']:.3f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    
    # 5) Save results
    result_dir = Path(paths_config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    results_df = pd.DataFrame({
        "ds": all_dates,
        "y": y_actual_array,
        "y_hat": y_hat_array,
        "y_hat_lower": y_lower_array if y_lower_array is not None else [np.nan] * len(all_dates),
        "y_hat_upper": y_upper_array if y_upper_array is not None else [np.nan] * len(all_dates),
        "error": y_actual_array - y_hat_array,
        "abs_error": np.abs(y_actual_array - y_hat_array)
    })
    
    csv_path = result_dir / f"{target}_historical_forecast.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Historical forecast results saved to: {csv_path}")
    
    # Save metrics JSON
    metrics_dict = {
        "target": target,
        "rmse": float(metrics['RMSE']),
        "mae": float(metrics['MAE']),
        "mape": float(metrics['MAPE']),
        "r2": float(metrics['R2'])
    }
    
    metrics_path = result_dir / f"{target}_historical_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Historical forecast metrics saved to: {metrics_path}")
    
    # 6) Generate plot
    print("Generating historical forecast plot...")
    metrics_text = f"RMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}, MAPE: {metrics['MAPE']:.3f}%, R²(historical): {metrics['R2']:.4f}"
    
    plot_path = result_dir / f"{target}_historical_forecast.png"
    plot_forecast(
        dates=pd.Series(all_dates),
        y_actual=pd.Series(y_actual_array),
        y_pred=pd.Series(y_hat_array),
        y_lower=pd.Series(y_lower_array) if y_lower_array is not None else None,
        y_upper=pd.Series(y_upper_array) if y_upper_array is not None else None,
        title=f"{target} - Historical Forecast (Rolling Backtest)",
        metrics_text=metrics_text,
        save_path=plot_path,
        show=False
    )
    print(f"Plot saved to: {plot_path}")
    
    return {
        "target": target,
        "metrics": metrics_dict,
        "forecast_df": results_df
    }

