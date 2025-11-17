"""
Forecasting pipeline for NHITS models.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from neuralforecast import NeuralForecast

from ..config.yaml_loader import (
    load_paths_config,
    load_train_config,
    load_exogenous_config,
    load_targets_config,
    TrainConfig,
    ExogenousConfig,
    PathsConfig,
)
from ..data import load_target_df
from ..data.feature_engineering import drop_unused_fx_columns
from ..data.preprocess import scale_columns
from ..model import create_nhits


def forecast_target(
    target: str,
    train_config: TrainConfig,
    exog_config: ExogenousConfig,
    paths_config: PathsConfig
):
    """
    Generate forecasts for a single target.
    
    Args:
        target: Target name
        train_config: Training configuration
        exog_config: Exogenous configuration
        paths_config: Paths configuration
    """
    print(f"\n{'='*60}")
    print(f"Forecasting for: {target}")
    print(f"{'='*60}")
    
    # 1) Load full data (train+val)
    print("Loading data...")
    raw_data_dir = Path(paths_config.raw_data_dir)
    df = load_target_df(target, raw_data_dir, exog_config)
    df = drop_unused_fx_columns(df, target)
    
    # 2) Scale data (we need to fit scaler on full history)
    # For forecasting, we use all available data for scaling
    exog_cols = [col for col in df.columns if col not in ["ds", "y"]]
    
    # Create dummy train/val split just for scaling
    split_idx = len(df) - train_config.horizon
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    train_df_scaled, _, target_scaler, exog_scaler = scale_columns(
        train_df, val_df, target_col="y", exog_cols=exog_cols
    )
    
    # Scale full dataset
    full_df_scaled = df.copy()
    full_df_scaled["y"] = target_scaler.transform(df[["y"]]).flatten()
    if exog_cols:
        full_df_scaled[exog_cols] = exog_scaler.transform(df[exog_cols])
    
    # 3) Prepare data for NeuralForecast
    full_nf = full_df_scaled.copy()
    full_nf["unique_id"] = target
    full_nf = full_nf[["unique_id", "ds", "y"] + exog_cols]
    
    # 4) Create and fit model
    exog_dim = len(exog_cols)
    print(f"Creating NHITS model with {exog_dim} exogenous variables...")
    nhits_model = create_nhits(train_config, exog_dim)
    nf = NeuralForecast(models=[nhits_model], freq="M")
    
    print("Fitting model on full history...")
    nf.fit(df=full_nf, val_size=0)
    
    # 5) Generate forecast
    print(f"Generating forecast for next {train_config.horizon} months...")
    predictions = nf.predict(df=full_nf, level=[95])
    
    # Extract forecast values
    forecast_values = predictions["NHITS"].values
    forecast_lower = predictions["NHITS-lo-95"].values if "NHITS-lo-95" in predictions.columns else None
    forecast_upper = predictions["NHITS-hi-95"].values if "NHITS-hi-95" in predictions.columns else None
    
    # 6) Inverse transform forecast
    forecast_inverse = target_scaler.inverse_transform(
        forecast_values.reshape(-1, 1)
    ).flatten()
    
    forecast_lower_inverse = None
    forecast_upper_inverse = None
    if forecast_lower is not None and forecast_upper is not None:
        forecast_lower_inverse = target_scaler.inverse_transform(
            forecast_lower.reshape(-1, 1)
        ).flatten()
        forecast_upper_inverse = target_scaler.inverse_transform(
            forecast_upper.reshape(-1, 1)
        ).flatten()
    
    # 7) Create forecast dataframe
    # Generate future dates
    last_date = df["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=train_config.horizon,
        freq="M"
    )
    
    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "y": [np.nan] * len(future_dates),  # Unknown future values
        "y_hat": forecast_inverse
    })
    
    if forecast_lower_inverse is not None and forecast_upper_inverse is not None:
        forecast_df["y_hat_lower_95"] = forecast_lower_inverse
        forecast_df["y_hat_upper_95"] = forecast_upper_inverse
    
    # Combine with historical data
    historical_df = pd.DataFrame({
        "ds": df["ds"],
        "y": df["y"],
        "y_hat": [np.nan] * len(df)  # No predictions for history in this context
    })
    
    result_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    
    # 8) Save forecast
    result_dir = Path(paths_config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = result_dir / f"{target}_forecast.csv"
    result_df.to_csv(forecast_path, index=False)
    print(f"Forecast saved to: {forecast_path}")
    
    # Print forecast summary
    print("\nForecast Summary:")
    print(forecast_df[["ds", "y_hat"]].to_string(index=False))
    
    return result_df


def main():
    """Generate forecasts for all targets."""
    parser = argparse.ArgumentParser(description="Generate forecasts with NHITS models")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="config",
        help="Directory containing YAML config files"
    )
    
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    
    # Load configurations
    paths_config = load_paths_config(config_dir)
    train_config = load_train_config(config_dir)
    exog_config = load_exogenous_config(config_dir)
    targets_config = load_targets_config(config_dir)
    
    print("="*60)
    print("NHiTS Forecasting Pipeline")
    print("="*60)
    print(f"Config directory: {config_dir}")
    print("="*60)
    
    for target in targets_config.targets:
        try:
            forecast_target(target, train_config, exog_config, paths_config)
        except Exception as e:
            print(f"Error forecasting {target}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
