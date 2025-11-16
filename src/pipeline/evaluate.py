import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from neuralforecast import NeuralForecast
from ..config import TrainConfig, ExogenousConfig, RAW_DATA_DIR, RESULT_DIR, MODEL_DIR, TARGETS
from ..data import load_target_df, train_val_split, scale_columns
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


if __name__ == "__main__":
    main()

