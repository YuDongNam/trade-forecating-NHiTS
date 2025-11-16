import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from neuralforecast import NeuralForecast
from ..config import TrainConfig, ExogenousConfig, RAW_DATA_DIR, RESULT_DIR, MODEL_DIR, TARGETS
from ..data import load_target_df, train_val_split, train_val_split_by_date, scale_columns
from ..data.feature_engineering import drop_unused_fx_columns
from ..model import create_nhits


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with RMSE, MAE, MAPE
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE: avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape)
    }


def evaluate_target(
    target: str,
    train_config: TrainConfig,
    exog_config: ExogenousConfig,
    val_length: int = 24
):
    """
    Evaluate NHITS model for a single target.
    
    Args:
        target: Target name
        train_config: Training configuration
        exog_config: Exogenous configuration
        val_length: Validation set length in months
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model for: {target}")
    print(f"{'='*60}")
    
    # 1) Load full data
    print("Loading data...")
    df = load_target_df(target, RAW_DATA_DIR, exog_config)
    df = drop_unused_fx_columns(df, target)
    
    # 2) Split into train/val
    train_df, val_df = train_val_split(df, train_config.horizon, val_length)
    
    # 3) Scale data (same as training)
    exog_cols = [col for col in train_df.columns if col not in ["ds", "y"]]
    train_df_scaled, val_df_scaled, target_scaler, exog_scaler = scale_columns(
        train_df, val_df, target_col="y", exog_cols=exog_cols
    )
    
    # 4) Prepare data for NeuralForecast
    train_nf = train_df_scaled.copy()
    train_nf["unique_id"] = target
    train_nf = train_nf[["unique_id", "ds", "y"] + exog_cols]
    
    val_nf = val_df_scaled.copy()
    val_nf["unique_id"] = target
    val_nf = val_nf[["unique_id", "ds", "y"] + exog_cols]
    
    # Combine train and val for full history
    full_nf = pd.concat([train_nf, val_nf], ignore_index=True)
    
    # 5) Create and load model
    exog_dim = len(exog_cols)
    nhits_model = create_nhits(train_config, exog_dim)
    nf = NeuralForecast(models=[nhits_model], freq="M")
    
    # 6) Fit on full training data
    print("Fitting model on training data...")
    nf.fit(df=train_nf, val_size=0)  # No validation split, we'll evaluate manually
    
    # 7) Generate predictions on validation set
    print("Generating predictions...")
    # Predict using only training data up to validation start
    predictions = nf.predict(df=train_nf)
    
    # Get predictions for the forecast horizon
    # NeuralForecast returns predictions for the next 'h' steps
    val_pred_scaled = predictions["NHITS"].values[:len(val_df)]
    
    # Get actual validation values (already scaled)
    val_actual_scaled = val_nf["y"].values[-len(val_df):]
    
    # 8) Inverse transform predictions and actuals
    val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    val_actual = target_scaler.inverse_transform(val_actual_scaled.reshape(-1, 1)).flatten()
    
    # 9) Compute metrics
    metrics = compute_metrics(val_actual, val_pred)
    print(f"Metrics: {metrics}")
    
    # 10) Save metrics
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULT_DIR / f"{target}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # 11) Generate plot
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual (validation)
    ax.plot(val_df["ds"], val_actual, "k-", label="Actual", linewidth=2)
    
    # Plot forecast
    ax.plot(val_df["ds"], val_pred, "b-", label="Forecast", linewidth=2)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Trade Balance", fontsize=12)
    ax.set_title(
        f"{target}\nRMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}, MAPE: {metrics['MAPE']:.3f}%",
        fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    plot_path = RESULT_DIR / f"{target}_forecast.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    return metrics


def evaluate_2024(
    target: str,
    train_config: TrainConfig,
    exog_config: ExogenousConfig
):
    """
    Evaluate model by training on data up to 2023-12-01 and predicting 2024.
    
    Args:
        target: Target name
        train_config: Training configuration
        exog_config: Exogenous configuration
    
    Returns:
        Dictionary with RMSE, MAE, MAPE metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {target} - Training on 2023 data, Predicting 2024")
    print(f"{'='*60}")
    
    # 1) Load full data
    print("Loading data...")
    df = load_target_df(target, RAW_DATA_DIR, exog_config)
    df = drop_unused_fx_columns(df, target)
    
    # 2) Split by date: train up to 2023-12-01, validate 2024
    print("Splitting data: train <= 2023-12-01, validation = 2024...")
    train_df, val_df = train_val_split_by_date(df, train_end_date="2023-12-01")
    
    print(f"Train period: {train_df['ds'].min()} to {train_df['ds'].max()}")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation period: {val_df['ds'].min()} to {val_df['ds'].max()}")
    print(f"Validation shape: {val_df.shape}")
    
    if len(val_df) == 0:
        print(f"Warning: No validation data found for 2024. Skipping {target}.")
        return None
    
    if len(val_df) < train_config.horizon:
        print(f"Warning: Validation data ({len(val_df)} months) is less than horizon ({train_config.horizon} months).")
        print(f"Using available {len(val_df)} months for validation.")
    
    # 3) Scale data (fit on train, transform both)
    exog_cols = [col for col in train_df.columns if col not in ["ds", "y"]]
    train_df_scaled, val_df_scaled, target_scaler, exog_scaler = scale_columns(
        train_df, val_df, target_col="y", exog_cols=exog_cols
    )
    
    # 4) Prepare data for NeuralForecast
    train_nf = train_df_scaled.copy()
    train_nf["unique_id"] = target
    train_nf = train_nf[["unique_id", "ds", "y"] + exog_cols]
    
    # For validation, we need exogenous variables for 2024
    # Use the scaled validation data
    val_nf = val_df_scaled.copy()
    val_nf["unique_id"] = target
    val_nf = val_nf[["unique_id", "ds", "y"] + exog_cols]
    
    # 5) Create and fit model
    exog_dim = len(exog_cols)
    print(f"Creating NHITS model with {exog_dim} exogenous variables...")
    nhits_model = create_nhits(train_config, exog_dim)
    nf = NeuralForecast(models=[nhits_model], freq="M")
    
    print("Training model on 2023 data...")
    nf.fit(df=train_nf, val_size=0)
    
    # 6) Generate predictions for 2024
    print("Generating predictions for 2024...")
    # Predict using training data up to 2023-12-01
    predictions = nf.predict(df=train_nf)
    
    # Get predictions for 2024 (first 12 months or available months)
    num_val_months = min(len(val_df), train_config.horizon)
    val_pred_scaled = predictions["NHITS"].values[:num_val_months]
    
    # Get actual 2024 values (first num_val_months)
    val_actual_scaled = val_nf["y"].values[:num_val_months]
    
    # 7) Inverse transform
    val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    val_actual = target_scaler.inverse_transform(val_actual_scaled.reshape(-1, 1)).flatten()
    
    # Align dates
    val_dates = val_df["ds"].values[:num_val_months]
    
    # 8) Compute metrics
    metrics = compute_metrics(val_actual, val_pred)
    print(f"\n{'='*60}")
    print(f"Validation Metrics for {target} (2024):")
    print(f"  RMSE: {metrics['RMSE']:.3f}")
    print(f"  MAE:  {metrics['MAE']:.3f}")
    print(f"  MAPE: {metrics['MAPE']:.3f}%")
    print(f"{'='*60}")
    
    # 9) Save metrics
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULT_DIR / f"{target}_2024_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # 10) Generate plot
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual 2024
    ax.plot(val_dates, val_actual, "ko-", label="Actual 2024", linewidth=2, markersize=8)
    
    # Plot forecast 2024
    ax.plot(val_dates, val_pred, "bs-", label="Forecast 2024", linewidth=2, markersize=8, alpha=0.7)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Trade Balance", fontsize=12)
    ax.set_title(
        f"{target} - 2024 Validation\n"
        f"RMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}, MAPE: {metrics['MAPE']:.3f}%",
        fontsize=14,
        fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    plot_path = RESULT_DIR / f"{target}_2024_validation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    # 11) Save detailed results
    results_df = pd.DataFrame({
        "ds": val_dates,
        "y_actual": val_actual,
        "y_pred": val_pred,
        "error": val_actual - val_pred,
        "abs_error": np.abs(val_actual - val_pred),
        "pct_error": ((val_actual - val_pred) / val_actual * 100) if np.any(val_actual != 0) else np.nan
    })
    
    results_path = RESULT_DIR / f"{target}_2024_validation.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to: {results_path}")
    
    return metrics


def main():
    """Evaluate models for all targets."""
    train_config = TrainConfig()
    exog_config = ExogenousConfig()
    
    all_metrics = {}
    
    for target in TARGETS:
        try:
            metrics = evaluate_target(target, train_config, exog_config)
            all_metrics[target] = metrics
        except Exception as e:
            print(f"Error evaluating {target}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY OF METRICS")
    print("="*60)
    print(f"{'Target':<20} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
    print("-"*60)
    for target, metrics in all_metrics.items():
        print(f"{target:<20} {metrics['RMSE']:<12.3f} {metrics['MAE']:<12.3f} {metrics['MAPE']:<12.3f}")
    
    # Save summary
    summary_path = RESULT_DIR / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


def main_2024():
    """Evaluate models for all targets using 2023 training and 2024 validation."""
    train_config = TrainConfig()
    exog_config = ExogenousConfig()
    
    all_metrics = {}
    
    print("\n" + "="*80)
    print("2024 VALIDATION - Training on 2023 data, Predicting 2024")
    print("="*80)
    
    for target in TARGETS:
        try:
            metrics = evaluate_2024(target, train_config, exog_config)
            if metrics is not None:
                all_metrics[target] = metrics
        except Exception as e:
            print(f"Error evaluating {target}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF 2024 VALIDATION METRICS")
    print("="*80)
    print(f"{'Target':<20} {'RMSE':<15} {'MAE':<15} {'MAPE':<15}")
    print("-"*80)
    for target, metrics in all_metrics.items():
        print(f"{target:<20} {metrics['RMSE']:<15.3f} {metrics['MAE']:<15.3f} {metrics['MAPE']:<15.3f}%")
    
    # Save summary
    summary_path = RESULT_DIR / "summary_2024_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return all_metrics


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "2024":
        # Run 2024 validation
        main_2024()
    else:
        # Run default validation (last 24 months)
        main()

