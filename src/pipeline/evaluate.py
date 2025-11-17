"""
Evaluation pipeline for NHITS models.
"""
import argparse
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from neuralforecast import NeuralForecast

from ..config.yaml_loader import (
    load_paths_config,
    load_train_config,
    load_validation_config,
    load_exogenous_config,
    load_targets_config,
    TrainConfig,
    ExogenousConfig,
    ValidationConfig,
    PathsConfig,
)
from ..data import load_target_df, split_train_val, scale_columns
from ..data.feature_engineering import drop_unused_fx_columns
from ..model import create_nhits
from .metrics import compute_all_metrics, r2
from .plotting import plot_forecast, plot_full_period_with_validation


def evaluate_target(
    target: str,
    train_config: TrainConfig,
    validation_config: ValidationConfig,
    exog_config: ExogenousConfig,
    paths_config: PathsConfig,
    load_trained_model: bool = False
):
    """
    Evaluate NHITS model for a single target.
    
    Args:
        target: Target name
        train_config: Training configuration
        validation_config: Validation configuration
        exog_config: Exogenous configuration
        paths_config: Paths configuration
        load_trained_model: Whether to load a pre-trained model (not implemented yet)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model for: {target}")
    print(f"{'='*60}")
    
    # 1) Load full data
    print("Loading data...")
    raw_data_dir = Path(paths_config.raw_data_dir)
    df = load_target_df(target, raw_data_dir, exog_config)
    df = drop_unused_fx_columns(df, target)
    
    # 2) Split into train/val using config
    print(f"Splitting data (mode: {validation_config.mode})...")
    train_df, val_df = split_train_val(df, validation_config)
    
    if len(val_df) == 0:
        print(f"Warning: No validation data for {target}. Skipping evaluation.")
        return None
    
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
    
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
    
    # 5) Create and fit model (or load if available)
    exog_dim = len(exog_cols)
    print(f"Creating NHITS model with {exog_dim} exogenous variables...")
    nhits_model = create_nhits(train_config, exog_dim)
    nf = NeuralForecast(models=[nhits_model], freq="M")
    
    print("Fitting model on training data...")
    nf.fit(df=train_nf, val_size=0)
    
    # 6) Generate predictions on validation set
    print("Generating predictions...")
    predictions = nf.predict(df=train_nf, level=[95])
    
    # Get predictions for validation period
    num_val = min(len(val_df), train_config.horizon)
    val_pred_scaled = predictions["NHITS"].values[:num_val]
    val_lower_scaled = predictions["NHITS-lo-95"].values[:num_val] if "NHITS-lo-95" in predictions.columns else None
    val_upper_scaled = predictions["NHITS-hi-95"].values[:num_val] if "NHITS-hi-95" in predictions.columns else None
    
    val_actual_scaled = val_nf["y"].values[:num_val]
    
    # 7) Inverse transform
    val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    val_actual = target_scaler.inverse_transform(val_actual_scaled.reshape(-1, 1)).flatten()
    
    val_lower = None
    val_upper = None
    if val_lower_scaled is not None and val_upper_scaled is not None:
        val_lower = target_scaler.inverse_transform(val_lower_scaled.reshape(-1, 1)).flatten()
        val_upper = target_scaler.inverse_transform(val_upper_scaled.reshape(-1, 1)).flatten()
    
    val_dates = val_df["ds"].values[:num_val]
    
    # 8) Compute validation metrics
    val_metrics = compute_all_metrics(val_actual, val_pred)
    print(f"Validation Metrics: RMSE={val_metrics['RMSE']:.3f}, MAE={val_metrics['MAE']:.3f}, MAPE={val_metrics['MAPE']:.3f}%")
    
    # 9) Compute full-period R²
    print("Computing full-period R²...")
    full_predictions = nf.predict(df=train_nf, level=[95])
    num_pred = min(len(train_df), len(full_predictions))
    if num_pred > 0:
        train_pred_scaled = full_predictions["NHITS"].values[:num_pred]
        # Align with actual values (take last num_pred values from training set)
        train_actual_scaled = train_nf["y"].values[-num_pred:]
        train_pred = target_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
        train_actual = target_scaler.inverse_transform(train_actual_scaled.reshape(-1, 1)).flatten()
        r2_full = r2(train_actual, train_pred)
        print(f"Full-period R²: {r2_full:.4f}")
    else:
        r2_full = None
        print("Warning: Could not compute full-period R² (no predictions available)")
    
    # 10) Save metrics
    result_dir = Path(paths_config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = result_dir / f"{target}_val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # 11) Save full-period R²
    if r2_full is not None:
        r2_path = result_dir / f"{target}_full_r2.json"
        with open(r2_path, "w") as f:
            json.dump({"target": target, "r2_full": r2_full}, f, indent=2)
        print(f"Full-period R² saved to: {r2_path}")
    
    # 12) Generate plot
    print("Generating plot...")
    metrics_text = f"RMSE: {val_metrics['RMSE']:.3f}, MAE: {val_metrics['MAE']:.3f}, MAPE: {val_metrics['MAPE']:.3f}%"
    if r2_full is not None:
        metrics_text += f", R²(full): {r2_full:.4f}"
    
    plot_path = result_dir / f"{target}_forecast.png"
    plot_forecast(
        dates=pd.Series(val_dates),
        y_actual=pd.Series(val_actual),
        y_pred=pd.Series(val_pred),
        y_lower=pd.Series(val_lower) if val_lower is not None else None,
        y_upper=pd.Series(val_upper) if val_upper is not None else None,
        title=target,
        metrics_text=metrics_text,
        save_path=plot_path,
        show=False
    )
    print(f"Plot saved to: {plot_path}")
    
    # 13) Save detailed results
    results_df = pd.DataFrame({
        "ds": val_dates,
        "y_actual": val_actual,
        "y_pred": val_pred,
    })
    
    if val_lower is not None and val_upper is not None:
        results_df["y_pred_lower_95"] = val_lower
        results_df["y_pred_upper_95"] = val_upper
    
    results_df["error"] = val_actual - val_pred
    results_df["abs_error"] = np.abs(val_actual - val_pred)
    if np.any(val_actual != 0):
        results_df["pct_error"] = ((val_actual - val_pred) / val_actual * 100)
    else:
        results_df["pct_error"] = np.nan
    
    results_path = result_dir / f"{target}_validation.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to: {results_path}")
    
    return {
        "target": target,
        "val_metrics": val_metrics,
        "r2_full": r2_full
    }


def main():
    """Evaluate models for all targets."""
    parser = argparse.ArgumentParser(description="Evaluate NHITS models")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="config",
        help="Directory containing YAML config files"
    )
    parser.add_argument(
        "--val_mode",
        type=str,
        default=None,
        help="Override validation mode (tail/range/none)"
    )
    parser.add_argument(
        "--val_tail_months",
        type=int,
        default=None,
        help="Override validation tail_months"
    )
    parser.add_argument(
        "--val_start",
        type=str,
        default=None,
        help="Override validation start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--val_end",
        type=str,
        default=None,
        help="Override validation end date (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    
    # Load configurations
    paths_config = load_paths_config(config_dir)
    train_config = load_train_config(config_dir)
    validation_config = load_validation_config(config_dir)
    exog_config = load_exogenous_config(config_dir)
    targets_config = load_targets_config(config_dir)
    
    # Override validation config with CLI arguments if provided
    if args.val_mode:
        validation_config.mode = args.val_mode
    if args.val_tail_months:
        validation_config.tail_months = args.val_tail_months
    if args.val_start:
        validation_config.start = args.val_start
    if args.val_end:
        validation_config.end = args.val_end
    
    print("="*60)
    print("NHiTS Evaluation Pipeline")
    print("="*60)
    print(f"Config directory: {config_dir}")
    print(f"Validation mode: {validation_config.mode}")
    print("="*60)
    
    all_metrics = {}
    
    for target in targets_config.targets:
        try:
            result = evaluate_target(
                target,
                train_config,
                validation_config,
                exog_config,
                paths_config
            )
            if result:
                all_metrics[target] = result
        except Exception as e:
            print(f"Error evaluating {target}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF METRICS")
    print("="*80)
    print(f"{'Target':<20} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'R² Full':<12}")
    print("-"*80)
    for target, result in all_metrics.items():
        val_metrics = result.get("val_metrics", {})
        r2_full = result.get("r2_full")
        rmse = val_metrics.get("RMSE", "N/A")
        mae = val_metrics.get("MAE", "N/A")
        mape = val_metrics.get("MAPE", "N/A")
        r2_str = f"{r2_full:.4f}" if r2_full is not None else "N/A"
        
        if isinstance(rmse, (int, float)):
            print(f"{target:<20} {rmse:<12.3f} {mae:<12.3f} {mape:<12.3f} {r2_str:<12}")
        else:
            print(f"{target:<20} {rmse:<12} {mae:<12} {mape:<12} {r2_str:<12}")
    
    # Save summary
    summary_path = Path(paths_config.result_dir) / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
