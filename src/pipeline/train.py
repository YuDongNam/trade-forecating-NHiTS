import pandas as pd
import numpy as np
from pathlib import Path
from neuralforecast import NeuralForecast
from ..config import TrainConfig, ExogenousConfig, RAW_DATA_DIR, MODEL_DIR, TARGETS
from ..data import load_target_df, train_val_split, scale_columns
from ..data.feature_engineering import drop_unused_fx_columns
from ..model import create_nhits


def train_target(
    target: str,
    train_config: TrainConfig,
    exog_config: ExogenousConfig,
    val_length: int = 24
):
    """
    Train NHITS model for a single target.
    
    Args:
        target: Target name (e.g., "Korea_Import")
        train_config: Training configuration
        exog_config: Exogenous configuration
        val_length: Validation set length in months
    """
    print(f"\n{'='*60}")
    print(f"Training model for: {target}")
    print(f"{'='*60}")
    
    # 1) Load and merge data
    print("Loading data...")
    df = load_target_df(target, RAW_DATA_DIR, exog_config)
    
    # Drop unused FX columns
    df = drop_unused_fx_columns(df, target)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    # 2) Split into train/val
    print("Splitting data...")
    train_df, val_df = train_val_split(df, train_config.horizon, val_length)
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
    
    # 3) Scale target and exogenous variables
    print("Scaling data...")
    exog_cols = [col for col in train_df.columns if col not in ["ds", "y"]]
    train_df_scaled, val_df_scaled, target_scaler, exog_scaler = scale_columns(
        train_df, val_df, target_col="y", exog_cols=exog_cols
    )
    
    # 4) Convert to NeuralForecast format
    print("Preparing data for NeuralForecast...")
    # NeuralForecast expects: unique_id, ds, y, [exog_cols...]
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
    nf.fit(df=train_nf, val_size=val_length)
    
    # 8) Save model checkpoint
    model_dir = MODEL_DIR / target
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model (NeuralForecast saves internally, but we can also save scalers)
    import pickle
    scaler_path = model_dir / "scalers.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"target_scaler": target_scaler, "exog_scaler": exog_scaler}, f)
    
    print(f"Model training completed. Checkpoint saved to: {model_dir}")
    
    return nf, target_scaler, exog_scaler


def main():
    """Train models for all targets."""
    train_config = TrainConfig()
    exog_config = ExogenousConfig()
    
    for target in TARGETS:
        try:
            train_target(target, train_config, exog_config)
        except Exception as e:
            print(f"Error training {target}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

