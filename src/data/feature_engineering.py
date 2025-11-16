import pandas as pd
from typing import List, Optional


def get_fx_column(target: str) -> Optional[str]:
    """
    Get the appropriate FX column name for a target.
    
    Args:
        target: Target name (e.g., "Korea_Import")
    
    Returns:
        FX column name or None
    """
    if "Korea" in target:
        return "USD_KRW"
    elif "China" in target:
        return "USD_CNY"
    elif "Taiwan" in target:
        return "USD_TWD"
    else:
        return None


def drop_unused_fx_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Drop FX columns that are not relevant for the target.
    
    Args:
        df: DataFrame with potential FX columns
        target: Target name
    
    Returns:
        DataFrame with unused FX columns removed
    """
    fx_columns = ["USD_KRW", "USD_CNY", "USD_TWD"]
    target_fx = get_fx_column(target)
    
    cols_to_drop = [col for col in fx_columns if col in df.columns and col != target_fx]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    
    return df


def build_design_matrix(
    df: pd.DataFrame,
    target_col: str = "y",
    exog_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build final design matrix with target and exogenous columns.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        exog_cols: List of exogenous column names (if None, auto-detect)
    
    Returns:
        DataFrame with columns: ["ds", target_col, ...exog_cols]
    """
    if exog_cols is None:
        exog_cols = [col for col in df.columns if col not in ["ds", target_col]]
    
    required_cols = ["ds", target_col] + exog_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")
    
    return df[required_cols].copy()


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple calendar features (month dummies).
    
    Args:
        df: DataFrame with "ds" column
    
    Returns:
        DataFrame with added calendar features
    """
    df = df.copy()
    df["month"] = df["ds"].dt.month
    
    # Create month dummy variables
    for month in range(1, 13):
        df[f"month_{month}"] = (df["month"] == month).astype(int)
    
    df = df.drop(columns=["month"])
    
    return df

