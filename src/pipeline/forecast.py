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
from .plotting import plot_forecast
from .uncertainty import mc_dropout_predict


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
    print(f"Dropout rate: {train_config.dropout_rate} (for Monte Carlo Dropout)")
    nhits_model = create_nhits(train_config, exog_dim, dropout_rate=train_config.dropout_rate)
    nf = NeuralForecast(models=[nhits_model], freq="ME")  # Use 'ME' instead of deprecated 'M'
    
    print("Fitting model on full history...")
    nf.fit(df=full_nf, val_size=0)
    
    # 5) Generate forecast using MC Dropout (if enabled) or deterministic
    from ..config.yaml_loader import load_uncertainty_config
    uncertainty_config = load_uncertainty_config(Path("config"))
    
    print(f"Generating forecast for next {train_config.horizon} months...")
    if uncertainty_config.enabled and uncertainty_config.method == "mc_dropout":
        print(f"Using Monte Carlo Dropout with {uncertainty_config.n_samples} samples...")
        forecast_inverse, forecast_lower_inverse, forecast_upper_inverse = mc_dropout_predict(
            nf=nf,
            df=full_nf,
            n_samples=uncertainty_config.n_samples,
            ci_level=uncertainty_config.ci_level,
            model_name="NHITS",
            target_scaler=target_scaler
        )
    else:
        print("Using deterministic forecast...")
        predictions = nf.predict(df=full_nf)
        forecast_scaled = predictions["NHITS"].values if "NHITS" in predictions.columns else predictions.iloc[:, 0].values
        forecast_inverse = target_scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        forecast_lower_inverse = None
        forecast_upper_inverse = None
    
    # 7) Create forecast dataframe
    # Generate future dates
    last_date = df["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=train_config.horizon,
        freq="M"
    )
    
    # Create forecast dataframe with confidence intervals
    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "y": [np.nan] * len(future_dates),  # Unknown future values
        "y_hat": forecast_inverse
    })
    
    # Always include confidence interval columns (even if None/NaN)
    if forecast_lower_inverse is not None and forecast_upper_inverse is not None:
        forecast_df["y_hat_lower_95"] = forecast_lower_inverse
        forecast_df["y_hat_upper_95"] = forecast_upper_inverse
    else:
        # If MC Dropout is disabled or failed, fill with NaN
        forecast_df["y_hat_lower_95"] = [np.nan] * len(future_dates)
        forecast_df["y_hat_upper_95"] = [np.nan] * len(future_dates)
    
    # Combine with historical data
    historical_df = pd.DataFrame({
        "ds": df["ds"],
        "y": df["y"],
        "y_hat": [np.nan] * len(df),  # No predictions for history in this context
        "y_hat_lower_95": [np.nan] * len(df),  # No CI for history
        "y_hat_upper_95": [np.nan] * len(df)   # No CI for history
    })
    
    result_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    
    # 8) Save forecast
    result_dir = Path(paths_config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = result_dir / f"{target}_forecast.csv"
    result_df.to_csv(forecast_path, index=False)
    print(f"Forecast saved to: {forecast_path}")
    
    # 9) Generate plot for forecast
    print("Generating forecast plot...")
    # Combine historical and forecast data for plotting
    historical_dates = pd.Series(df["ds"].values)
    historical_values = pd.Series(df["y"].values)
    forecast_dates = pd.Series(future_dates)
    forecast_values = pd.Series(forecast_inverse)
    
    # Combine for plotting
    all_dates = pd.concat([historical_dates, forecast_dates], ignore_index=True)
    all_actual = pd.concat([historical_values, pd.Series([np.nan] * len(forecast_dates))], ignore_index=True)
    all_forecast = pd.concat([pd.Series([np.nan] * len(historical_dates)), forecast_values], ignore_index=True)
    
    forecast_lower_series = None
    forecast_upper_series = None
    if forecast_lower_inverse is not None and forecast_upper_inverse is not None:
        forecast_lower_series = pd.concat([pd.Series([np.nan] * len(historical_dates)), pd.Series(forecast_lower_inverse)], ignore_index=True)
        forecast_upper_series = pd.concat([pd.Series([np.nan] * len(historical_dates)), pd.Series(forecast_upper_inverse)], ignore_index=True)
    
    plot_path = result_dir / f"{target}_future_forecast.png"
    plot_forecast(
        dates=all_dates,
        y_actual=all_actual,
        y_pred=all_forecast,
        y_lower=forecast_lower_series,
        y_upper=forecast_upper_series,
        title=f"{target} - Future Forecast",
        metrics_text=f"Forecast horizon: {train_config.horizon} months",
        save_path=plot_path,
        show=False
    )
    print(f"Forecast plot saved to: {plot_path}")
    
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
    
    # Use parse_known_args to ignore Jupyter/Colab kernel arguments
    args, unknown = parser.parse_known_args()
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
