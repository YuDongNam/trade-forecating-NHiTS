"""
결과를 로드해서 플롯을 그리는 스크립트

사용법:
    python plot_results.py
    
또는 Python 코드에서:
    from plot_results import plot_validation_results, plot_all_targets
    plot_all_targets()
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config.yaml_loader import load_paths_config, load_targets_config
from src.data import load_target_df
from src.config.yaml_loader import load_exogenous_config
from src.pipeline.plotting import plot_forecast, plot_full_period_with_validation


def plot_validation_results(target: str, config_dir: Path = Path("config"), show: bool = True):
    """
    특정 타겟의 검증 결과를 플롯합니다.
    
    Args:
        target: 타겟 이름 (예: "Korea_Import")
        config_dir: 설정 디렉토리 경로
        show: 플롯을 화면에 표시할지 여부
    """
    paths_config = load_paths_config(config_dir)
    result_dir = Path(paths_config.result_dir)
    
    # 검증 결과 로드
    validation_file = result_dir / f"{target}_validation.csv"
    if not validation_file.exists():
        print(f"Error: {validation_file} not found.")
        return None
    
    val_df = pd.read_csv(validation_file)
    val_df["ds"] = pd.to_datetime(val_df["ds"])
    val_df = val_df.sort_values("ds").reset_index(drop=True)
    
    # 메트릭 로드
    metrics_file = result_dir / f"{target}_val_metrics.json"
    metrics_text = None
    if metrics_file.exists():
        import json
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        metrics_text = f"RMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}, MAPE: {metrics['MAPE']:.3f}%"
        
        # R² 로드
        r2_file = result_dir / f"{target}_full_r2.json"
        if r2_file.exists():
            with open(r2_file, "r") as f:
                r2_data = json.load(f)
            metrics_text += f", R²(full): {r2_data['r2_full']:.4f}"
    
    # 플롯 생성
    y_lower = pd.Series(val_df["y_pred_lower_95"]) if "y_pred_lower_95" in val_df.columns else None
    y_upper = pd.Series(val_df["y_pred_upper_95"]) if "y_pred_upper_95" in val_df.columns else None
    
    fig = plot_forecast(
        dates=pd.Series(val_df["ds"]),
        y_actual=pd.Series(val_df["y_actual"]),
        y_pred=pd.Series(val_df["y_pred"]),
        y_lower=y_lower,
        y_upper=y_upper,
        title=target,
        metrics_text=metrics_text,
        save_path=result_dir / f"{target}_forecast_custom.png",
        show=show
    )
    
    return fig


def plot_full_period(target: str, config_dir: Path = Path("config"), show: bool = True):
    """
    전체 기간(학습 + 검증)을 포함한 플롯을 그립니다.
    
    Args:
        target: 타겟 이름
        config_dir: 설정 디렉토리 경로
        show: 플롯을 화면에 표시할지 여부
    """
    paths_config = load_paths_config(config_dir)
    result_dir = Path(paths_config.result_dir)
    raw_data_dir = Path(paths_config.raw_data_dir)
    
    # 검증 결과 로드
    validation_file = result_dir / f"{target}_validation.csv"
    if not validation_file.exists():
        print(f"Error: {validation_file} not found.")
        return None
    
    val_df = pd.read_csv(validation_file)
    val_df["ds"] = pd.to_datetime(val_df["ds"])
    val_df = val_df.sort_values("ds").reset_index(drop=True)
    
    # 전체 데이터 로드
    exog_config = load_exogenous_config(config_dir)
    try:
        from src.data import load_target_df
        from src.data.feature_engineering import drop_unused_fx_columns
        
        full_df = load_target_df(target, raw_data_dir, exog_config)
        full_df = drop_unused_fx_columns(full_df, target)
        full_df["ds"] = pd.to_datetime(full_df["ds"])
        full_df = full_df.sort_values("ds").reset_index(drop=True)
        
        # 학습/검증 분할 (검증 시작 날짜 기준)
        val_start_date = val_df["ds"].min()
        train_df = full_df[full_df["ds"] < val_start_date]
        val_actual_df = full_df[full_df["ds"] >= val_start_date]
        
    except Exception as e:
        print(f"Warning: Could not load full data for {target}: {e}")
        train_df = None
        val_actual_df = None
    
    # 메트릭 텍스트
    metrics_file = result_dir / f"{target}_val_metrics.json"
    metrics_text = None
    if metrics_file.exists():
        import json
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        metrics_text = f"RMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}, MAPE: {metrics['MAPE']:.3f}%"
        
        r2_file = result_dir / f"{target}_full_r2.json"
        if r2_file.exists():
            with open(r2_file, "r") as f:
                r2_data = json.load(f)
            metrics_text += f", R²(full): {r2_data['r2_full']:.4f}"
    
    # 플롯 생성
    if train_df is not None and len(val_actual_df) > 0:
        y_lower = pd.Series(val_df["y_pred_lower_95"]) if "y_pred_lower_95" in val_df.columns else None
        y_upper = pd.Series(val_df["y_pred_upper_95"]) if "y_pred_upper_95" in val_df.columns else None
        
        fig = plot_full_period_with_validation(
            train_dates=pd.Series(train_df["ds"]),
            train_actual=pd.Series(train_df["y"]),
            val_dates=pd.Series(val_df["ds"]),
            val_actual=pd.Series(val_actual_df["y"]) if len(val_actual_df) > 0 else pd.Series(val_df["y_actual"]),
            val_pred=pd.Series(val_df["y_pred"]),
            val_lower=y_lower,
            val_upper=y_upper,
            target_name=target,
            metrics_text=metrics_text,
            save_path=result_dir / f"{target}_full_period.png",
            show=show
        )
        return fig
    else:
        # 전체 데이터가 없으면 검증 기간만 플롯
        return plot_validation_results(target, config_dir, show)


def plot_all_targets(config_dir: Path = Path("config"), show: bool = True):
    """
    모든 타겟의 검증 결과를 플롯합니다.
    
    Args:
        config_dir: 설정 디렉토리 경로
        show: 플롯을 화면에 표시할지 여부
    """
    targets_config = load_targets_config(config_dir)
    paths_config = load_paths_config(config_dir)
    result_dir = Path(paths_config.result_dir)
    
    print("="*60)
    print("Plotting validation results for all targets")
    print("="*60)
    
    for target in targets_config.targets:
        validation_file = result_dir / f"{target}_validation.csv"
        if validation_file.exists():
            print(f"\nPlotting {target}...")
            try:
                plot_validation_results(target, config_dir, show=show)
                print(f"✓ {target} plotted successfully")
            except Exception as e:
                print(f"✗ Error plotting {target}: {e}")
        else:
            print(f"⚠ {target}: validation file not found, skipping")


def plot_all_full_period(config_dir: Path = Path("config"), show: bool = True):
    """
    모든 타겟의 전체 기간 플롯을 그립니다.
    
    Args:
        config_dir: 설정 디렉토리 경로
        show: 플롯을 화면에 표시할지 여부
    """
    targets_config = load_targets_config(config_dir)
    paths_config = load_paths_config(config_dir)
    result_dir = Path(paths_config.result_dir)
    
    print("="*60)
    print("Plotting full period results for all targets")
    print("="*60)
    
    for target in targets_config.targets:
        validation_file = result_dir / f"{target}_validation.csv"
        if validation_file.exists():
            print(f"\nPlotting {target} (full period)...")
            try:
                plot_full_period(target, config_dir, show=show)
                print(f"✓ {target} plotted successfully")
            except Exception as e:
                print(f"✗ Error plotting {target}: {e}")
        else:
            print(f"⚠ {target}: validation file not found, skipping")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot validation results")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Specific target to plot (if not provided, plots all targets)"
    )
    parser.add_argument(
        "--full_period",
        action="store_true",
        help="Plot full period (training + validation) instead of validation only"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="config",
        help="Config directory path"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display plots (only save)"
    )
    
    args, unknown = parser.parse_known_args()
    config_dir = Path(args.config_dir)
    show = not args.no_show
    
    if args.target:
        # 특정 타겟만 플롯
        if args.full_period:
            plot_full_period(args.target, config_dir, show=show)
        else:
            plot_validation_results(args.target, config_dir, show=show)
    else:
        # 모든 타겟 플롯
        if args.full_period:
            plot_all_full_period(config_dir, show=show)
        else:
            plot_all_targets(config_dir, show=show)

