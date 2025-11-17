import pandas as pd
from pathlib import Path
from typing import Optional
from ..config.yaml_loader import ExogenousConfig


def load_target_df(target: str, raw_data_dir: Path, exog_config: ExogenousConfig) -> pd.DataFrame:
    """
    Load target data and merge with exogenous variables.
    
    Args:
        target: Target name (e.g., "Korea_Import")
        raw_data_dir: Path to raw data directory
        exog_config: Exogenous configuration
    
    Returns:
        DataFrame with columns: ["ds", "y", ...exog_cols]
    """
    # 1) Read base target file
    target_file = raw_data_dir / f"{target}.csv"
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")
    
    df = pd.read_csv(target_file)
    
    # 2) Parse and normalize date column
    # Handle different column name formats
    if "Date" in df.columns:
        df["ds"] = pd.to_datetime(df["Date"])
        df = df.drop(columns=["Date"])
    elif "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
    else:
        raise ValueError(f"Date column not found in {target_file}. Expected 'Date' or 'ds'")
    
    # Handle target value column
    if "PrimaryValue" in df.columns:
        df["y"] = df["PrimaryValue"]
        df = df.drop(columns=["PrimaryValue"])
    elif "y" not in df.columns:
        raise ValueError(f"Target column not found in {target_file}. Expected 'PrimaryValue' or 'y'")
    
    # Sort by date and ensure unique
    df = df.sort_values("ds").reset_index(drop=True)
    df = df.drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)
    
    # 3) Load exogenous variables
    exog_dfs = []
    
    # Tariff data
    if exog_config.use_tariff:
        tariff_file = raw_data_dir / "tariff.csv"
        if tariff_file.exists():
            tariff_df = pd.read_csv(tariff_file)
            # Normalize date column
            if "date" in tariff_df.columns:
                tariff_df["ds"] = pd.to_datetime(tariff_df["date"])
                tariff_df = tariff_df.drop(columns=["date"])
            elif "ds" in tariff_df.columns:
                tariff_df["ds"] = pd.to_datetime(tariff_df["ds"])
            else:
                raise ValueError(f"Date column not found in tariff.csv")
            
            # Keep all tariff columns
            tariff_cols = [col for col in tariff_df.columns if col != "ds"]
            exog_dfs.append(tariff_df[["ds"] + tariff_cols])
    
    # WTI data
    if exog_config.use_wti:
        wti_file = raw_data_dir / "WTI.csv"
        if wti_file.exists():
            wti_df = pd.read_csv(wti_file)
            wti_df["ds"] = pd.to_datetime(wti_df["ds"])
            # Use "y" column as WTI price
            wti_df = wti_df[["ds", "y"]].rename(columns={"y": "WTI"})
            exog_dfs.append(wti_df)
    
    # Copper data
    if exog_config.use_copper:
        copper_file = raw_data_dir / "copper.csv"
        if copper_file.exists():
            copper_df = pd.read_csv(copper_file)
            copper_df["ds"] = pd.to_datetime(copper_df["ds"])
            # Use "y" column as copper price
            copper_df = copper_df[["ds", "y"]].rename(columns={"y": "copper"})
            exog_dfs.append(copper_df)
    
    # FX data - select based on target country
    if exog_config.use_fx:
        fx_file = None
        if "Korea" in target:
            fx_file = raw_data_dir / "USD_KRW.csv"
            fx_col_name = "USD_KRW"
        elif "China" in target:
            fx_file = raw_data_dir / "USD_CNY.csv"
            fx_col_name = "USD_CNY"
        elif "Taiwan" in target:
            fx_file = raw_data_dir / "USD_TWD.csv"
            fx_col_name = "USD_TWD"
        # World_* targets don't have a specific FX mapping
        
        if fx_file and fx_file.exists():
            fx_df = pd.read_csv(fx_file)
            fx_df["ds"] = pd.to_datetime(fx_df["ds"])
            # Handle string values with commas
            if fx_df["y"].dtype == "object":
                fx_df["y"] = fx_df["y"].str.replace(",", "").astype(float)
            fx_df = fx_df[["ds", "y"]].rename(columns={"y": fx_col_name})
            exog_dfs.append(fx_df)
    
    # 4) Merge all exogenous variables
    for exog_df in exog_dfs:
        df = df.merge(exog_df, on="ds", how="left")
    
    # 5) Forward-fill and back-fill missing exogenous values
    exog_cols = [col for col in df.columns if col not in ["ds", "y"]]
    for col in exog_cols:
        df[col] = df[col].ffill().bfill()
    
    # Ensure final columns: ["ds", "y", ...exog_cols]
    df = df[["ds", "y"] + exog_cols]
    
    return df

