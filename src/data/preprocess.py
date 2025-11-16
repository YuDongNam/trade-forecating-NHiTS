import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


def train_val_split(
    df: pd.DataFrame, 
    horizon: int, 
    val_length: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets.
    
    Args:
        df: Full dataset with "ds" column
        horizon: Forecast horizon in months
        val_length: Number of months to use for validation (default: 24)
    
    Returns:
        train_df, val_df
    """
    df = df.sort_values("ds").reset_index(drop=True)
    
    # Use last val_length months as validation
    split_idx = len(df) - val_length
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    return train_df, val_df


def train_val_split_by_date(
    df: pd.DataFrame,
    train_end_date: str = "2023-12-01"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets by date.
    
    Args:
        df: Full dataset with "ds" column
        train_end_date: Last date for training data (format: "YYYY-MM-DD")
    
    Returns:
        train_df, val_df
    """
    df = df.sort_values("ds").reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"])
    
    train_end = pd.to_datetime(train_end_date)
    
    train_df = df[df["ds"] <= train_end].copy()
    val_df = df[df["ds"] > train_end].copy()
    
    return train_df, val_df


def scale_columns(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "y",
    exog_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Optional[StandardScaler]]:
    """
    Scale target and exogenous columns using StandardScaler.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        target_col: Name of target column
        exog_cols: List of exogenous column names
    
    Returns:
        scaled_train_df, scaled_val_df, target_scaler, exog_scaler
    """
    if exog_cols is None:
        exog_cols = [col for col in train_df.columns if col not in ["ds", target_col]]
    
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    
    # Scale target
    target_scaler = StandardScaler()
    train_df_scaled[target_col] = target_scaler.fit_transform(
        train_df[[target_col]]
    ).flatten()
    val_df_scaled[target_col] = target_scaler.transform(
        val_df[[target_col]]
    ).flatten()
    
    # Scale exogenous variables
    exog_scaler = None
    if exog_cols:
        exog_scaler = StandardScaler()
        train_df_scaled[exog_cols] = exog_scaler.fit_transform(
            train_df[exog_cols]
        )
        val_df_scaled[exog_cols] = exog_scaler.transform(
            val_df[exog_cols]
        )
    
    return train_df_scaled, val_df_scaled, target_scaler, exog_scaler

