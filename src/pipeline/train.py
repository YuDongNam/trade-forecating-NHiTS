"""
Training pipeline for NHITS models.
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


def train_target(
    target: str,
    train_config: TrainConfig,
    validation_config: ValidationConfig,
    exog_config: ExogenousConfig,
    paths_config: PathsConfig,
    compute_full_r2: bool = True
):
    """
    Train NHITS model for a single target.
    
    Args:
        target: Target name (e.g., "Korea_Import")
        train_config: Training configuration
        validation_config: Validation configuration
        exog_config: Exogenous configuration
        paths_config: Paths configuration
        compute_full_r2: Whether to compute full-period R²
    
    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training model for: {target}")
    print(f"{'='*60}")
    
    # 1) Load and merge data
    print("Loading data...")
    raw_data_dir = Path(paths_config.raw_data_dir)
    df = load_target_df(target, raw_data_dir, exog_config)
    df = drop_unused_fx_columns(df, target)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    # 2) Split into train/val using config
    print(f"Splitting data (mode: {validation_config.mode})...")
    train_df, val_df = split_train_val(df, validation_config)
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
    
    if len(train_df) == 0:
        raise ValueError(f"No training data available for {target}")
    
    # 3) Scale target and exogenous variables
    print("Scaling data...")
    exog_cols = [col for col in train_df.columns if col not in ["ds", "y"]]
    train_df_scaled, val_df_scaled, target_scaler, exog_scaler = scale_columns(
        train_df, val_df, target_col="y", exog_cols=exog_cols
    )
    
    # 4) Convert to NeuralForecast format
    print("Preparing data for NeuralForecast...")
    train_nf = train_df_scaled.copy()
    train_nf["unique_id"] = target
    train_nf = train_nf[["unique_id", "ds", "y"] + exog_cols]
    
    val_nf = val_df_scaled.copy()
    val_nf["unique_id"] = target
    val_nf = val_nf[["unique_id", "ds", "y"] + exog_cols]
    
    # 5) Create NHITS model
    exog_dim = len(exog_cols)
    print(f"Creating NHITS model with {exog_dim} exogenous variables...")
    nhits_model = create_nhits(train_config, exog_dim)
    
    # 6) Initialize NeuralForecast
    nf = NeuralForecast(
        models=[nhits_model],
        freq="M"  # Monthly frequency
    )
    
    # 7) Fit model
    print("Training model...")
    val_size = len(val_df) if len(val_df) > 0 else 0
    nf.fit(df=train_nf, val_size=val_size)
    
    # 8) Compute validation metrics if validation set exists
    val_metrics = None
    if len(val_df) > 0:
        print("Computing validation metrics...")
        # Generate predictions on validation set
        predictions = nf.predict(df=train_nf, level=[95])
        
        num_val = min(len(val_df), train_config.horizon)
        val_pred_scaled = predictions["NHITS"].values[:num_val]
        val_actual_scaled = val_nf["y"].values[:num_val]
        
        # Inverse transform
        val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
        val_actual = target_scaler.inverse_transform(val_actual_scaled.reshape(-1, 1)).flatten()
        
        # Compute metrics
        val_metrics = compute_all_metrics(val_actual, val_pred)
        print(f"Validation Metrics: RMSE={val_metrics['RMSE']:.3f}, MAE={val_metrics['MAE']:.3f}, MAPE={val_metrics['MAPE']:.3f}%")
    
    # 9) Compute full-period R² if requested
    r2_full = None
    if compute_full_r2:
        print("Computing full-period R²...")
        # For full-period R², we use the model's fitted predictions on training data
        # Generate predictions using the trained model on training data
        # This gives us in-sample performance
        full_predictions = nf.predict(df=train_nf, level=[95])
        
        # Use all available training predictions
        # Note: predictions are for the forecast horizon, so we take what we can
        num_pred = min(len(train_df), len(full_predictions))
        if num_pred > 0:
            train_pred_scaled = full_predictions["NHITS"].values[:num_pred]
            # Align with actual values (take last num_pred values from training set)
            train_actual_scaled = train_nf["y"].values[-num_pred:]
            
            # Inverse transform
            train_pred = target_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
            train_actual = target_scaler.inverse_transform(train_actual_scaled.reshape(-1, 1)).flatten()
            
            # Compute R²
            r2_full = r2(train_actual, train_pred)
            print(f"Full-period R²: {r2_full:.4f}")
        else:
            print("Warning: Could not compute full-period R² (no predictions available)")
    
    # 10) Save model checkpoint
    model_dir = Path(paths_config.model_dir) / target
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scalers
    scaler_path = model_dir / "scalers.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"target_scaler": target_scaler, "exog_scaler": exog_scaler}, f)
    
    # 11) Save validation metrics
    result_dir = Path(paths_config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    if val_metrics:
        val_metrics_path = result_dir / f"{target}_val_metrics.json"
        with open(val_metrics_path, "w") as f:
            json.dump(val_metrics, f, indent=2)
        print(f"Validation metrics saved to: {val_metrics_path}")
    
    # 12) Save full-period R²
    if r2_full is not None:
        r2_path = result_dir / f"{target}_full_r2.json"
        with open(r2_path, "w") as f:
            json.dump({"target": target, "r2_full": r2_full}, f, indent=2)
        print(f"Full-period R² saved to: {r2_path}")
    
    print(f"Model training completed. Checkpoint saved to: {model_dir}")
    
    return {
        "target": target,
        "val_metrics": val_metrics,
        "r2_full": r2_full
    }


def main():
    """Train models for all targets."""
    parser = argparse.ArgumentParser(description="Train NHITS models")
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
    print("NHiTS Training Pipeline")
    print("="*60)
    print(f"Config directory: {config_dir}")
    print(f"Validation mode: {validation_config.mode}")
    print(f"Targets: {len(targets_config.targets)}")
    print("="*60)
    
    all_results = []
    
    for target in targets_config.targets:
        try:
            result = train_target(
                target,
                train_config,
                validation_config,
                exog_config,
                paths_config,
                compute_full_r2=True
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error training {target}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"{'Target':<20} {'Val RMSE':<12} {'Val MAE':<12} {'Val MAPE':<12} {'R² Full':<12}")
    print("-"*60)
    for result in all_results:
        target = result["target"]
        val_metrics = result.get("val_metrics")
        r2_full = result.get("r2_full")
        
        if val_metrics:
            print(f"{target:<20} {val_metrics['RMSE']:<12.3f} {val_metrics['MAE']:<12.3f} {val_metrics['MAPE']:<12.3f} {r2_full:<12.4f if r2_full else 'N/A':<12}")
        else:
            print(f"{target:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {r2_full:<12.4f if r2_full else 'N/A':<12}")


if __name__ == "__main__":
    main()
