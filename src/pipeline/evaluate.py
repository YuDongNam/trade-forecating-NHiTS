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
    load_uncertainty_config,
    TrainConfig,
    ExogenousConfig,
    ValidationConfig,
    PathsConfig,
    UncertaintyConfig,
)
from ..data import load_target_df, split_train_val, scale_columns
from ..data.feature_engineering import drop_unused_fx_columns
from ..model import create_nhits
from .metrics import compute_all_metrics
from .plotting import plot_forecast, plot_full_period_with_validation
from .historical_forecast import historical_forecast


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
    
    # Store exog_scaler for historical_forecast
    exog_scaler_stored = exog_scaler
    
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
    print(f"Dropout rate: {train_config.dropout_rate} (for Monte Carlo Dropout)")
    nhits_model = create_nhits(train_config, exog_dim, dropout_rate=train_config.dropout_rate)
    nf = NeuralForecast(models=[nhits_model], freq="ME")  # Use 'ME' instead of deprecated 'M'
    
    print("Fitting model on training data...")
    nf.fit(df=train_nf, val_size=0)
    
    # 6) Generate deterministic predictions on validation set
    # (No MC Dropout during standard validation - MC Dropout is only for prediction-time uncertainty)
    print("Generating deterministic predictions...")
    predictions = nf.predict(df=train_nf)
    
    # Get predictions for validation period
    num_val = min(len(val_df), train_config.horizon)
    val_pred_scaled = predictions["NHITS"].values[:num_val]
    val_actual_scaled = val_nf["y"].values[:num_val]
    
    # 7) Inverse transform
    val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    val_actual = target_scaler.inverse_transform(val_actual_scaled.reshape(-1, 1)).flatten()
    
    val_dates = val_df["ds"].values[:num_val]
    
    # 8) Compute validation metrics (RMSE, MAE, MAPE only - no R²)
    val_metrics = compute_all_metrics(val_actual, val_pred, include_r2=False)
    print(f"Validation Metrics: RMSE={val_metrics['RMSE']:.3f}, MAE={val_metrics['MAE']:.3f}, MAPE={val_metrics['MAPE']:.3f}%")
    
    # 9) Save metrics
    result_dir = Path(paths_config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = result_dir / f"{target}_val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # 10) Generate plot with full period (2010-2024) and validation period highlighted
    print("Generating plot...")
    metrics_text = f"RMSE: {val_metrics['RMSE']:.3f}, MAE: {val_metrics['MAE']:.3f}, MAPE: {val_metrics['MAPE']:.3f}%"
    
    # Prepare full period data
    full_dates = pd.Series(df["ds"].values)
    full_actual = pd.Series(df["y"].values)
    
    # Determine validation period dates
    val_start_date = pd.to_datetime(val_dates[0]) if len(val_dates) > 0 else None
    val_end_date = pd.to_datetime(val_dates[-1]) if len(val_dates) > 0 else None
    
    plot_path = result_dir / f"{target}_forecast.png"
    plot_full_period_with_validation(
        full_dates=full_dates,
        full_actual=full_actual,
        val_dates=pd.Series(val_dates),
        val_pred=pd.Series(val_pred),
        val_lower=None,  # No CI for standard validation
        val_upper=None,
        target_name=target,
        metrics_text=metrics_text,
        val_start_date=val_start_date,
        val_end_date=val_end_date,
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
        "nf": nf,
        "train_nf": train_nf,
        "full_df": df,
        "target_scaler": target_scaler,
        "exog_scaler": exog_scaler_stored,
        "exog_cols": exog_cols
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
    
    # Use parse_known_args to ignore Jupyter/Colab kernel arguments
    args, unknown = parser.parse_known_args()
    config_dir = Path(args.config_dir)
    
    # Load configurations
    paths_config = load_paths_config(config_dir)
    train_config = load_train_config(config_dir)
    validation_config = load_validation_config(config_dir)
    exog_config = load_exogenous_config(config_dir)
    targets_config = load_targets_config(config_dir)
    uncertainty_config = load_uncertainty_config(config_dir)
    
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
    all_historical_results = {}
    
    # Step 1: Standard validation evaluation
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
    
    # Step 2: Historical forecast (rolling backtest) with R²
    print("\n" + "="*80)
    print("HISTORICAL FORECAST (ROLLING BACKTEST)")
    print("="*80)
    
    for target in targets_config.targets:
        try:
            # Get the model and data from standard evaluation
            eval_result = all_metrics.get(target)
            if not eval_result:
                print(f"Skipping historical forecast for {target} (standard evaluation failed)")
                continue
            
            # Run historical forecast
            hist_result = historical_forecast(
                target=target,
                train_config=train_config,
                validation_config=validation_config,
                uncertainty_config=uncertainty_config,
                paths_config=paths_config,
                nf=eval_result["nf"],
                train_nf=eval_result["train_nf"],
                full_df=eval_result["full_df"],
                target_scaler=eval_result["target_scaler"],
                exog_scaler=eval_result.get("exog_scaler"),
                exog_cols=eval_result["exog_cols"]
            )
            
            if hist_result:
                all_historical_results[target] = hist_result
        except Exception as e:
            print(f"Error in historical forecast for {target}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF METRICS")
    print("="*80)
    print(f"{'Target':<20} {'Val_RMSE':<12} {'Val_MAE':<12} {'Val_MAPE':<12} {'Hist_RMSE':<12} {'Hist_MAE':<12} {'Hist_MAPE':<12} {'Hist_R2':<12}")
    print("-"*80)
    
    for target in targets_config.targets:
        val_result = all_metrics.get(target, {})
        hist_result = all_historical_results.get(target, {})
        
        val_metrics = val_result.get("val_metrics", {})
        hist_metrics = hist_result.get("metrics", {})
        
        val_rmse = val_metrics.get("RMSE", "N/A")
        val_mae = val_metrics.get("MAE", "N/A")
        val_mape = val_metrics.get("MAPE", "N/A")
        
        hist_rmse = hist_metrics.get("rmse", "N/A")
        hist_mae = hist_metrics.get("mae", "N/A")
        hist_mape = hist_metrics.get("mape", "N/A")
        hist_r2 = hist_metrics.get("r2", "N/A")
        
        if isinstance(val_rmse, (int, float)) and isinstance(hist_rmse, (int, float)):
            print(f"{target:<20} {val_rmse:<12.3f} {val_mae:<12.3f} {val_mape:<12.3f} {hist_rmse:<12.3f} {hist_mae:<12.3f} {hist_mape:<12.3f} {hist_r2:<12.4f}")
        else:
            val_rmse_str = f"{val_rmse:.3f}" if isinstance(val_rmse, (int, float)) else str(val_rmse)
            val_mae_str = f"{val_mae:.3f}" if isinstance(val_mae, (int, float)) else str(val_mae)
            val_mape_str = f"{val_mape:.3f}" if isinstance(val_mape, (int, float)) else str(val_mape)
            hist_rmse_str = f"{hist_rmse:.3f}" if isinstance(hist_rmse, (int, float)) else str(hist_rmse)
            hist_mae_str = f"{hist_mae:.3f}" if isinstance(hist_mae, (int, float)) else str(hist_mae)
            hist_mape_str = f"{hist_mape:.3f}" if isinstance(hist_mape, (int, float)) else str(hist_mape)
            hist_r2_str = f"{hist_r2:.4f}" if isinstance(hist_r2, (int, float)) else str(hist_r2)
            print(f"{target:<20} {val_rmse_str:<12} {val_mae_str:<12} {val_mape_str:<12} {hist_rmse_str:<12} {hist_mae_str:<12} {hist_mape_str:<12} {hist_r2_str:<12}")
    
    # Save summary
    summary_path = Path(paths_config.result_dir) / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
