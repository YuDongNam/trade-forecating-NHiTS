"""
End-to-end pipeline runner: train, evaluate, and forecast for all targets.
"""
import argparse
import json
from pathlib import Path

from ..config.yaml_loader import (
    load_paths_config,
    load_train_config,
    load_validation_config,
    load_exogenous_config,
    load_targets_config,
    PathsConfig,
    TrainConfig,
    ValidationConfig,
    ExogenousConfig,
)
from .train import train_target
from .evaluate import evaluate_target
from .forecast import forecast_target


def main():
    """Run complete pipeline: train, evaluate, and forecast for all targets."""
    parser = argparse.ArgumentParser(description="Run complete NHiTS pipeline")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="config",
        help="Directory containing YAML config files"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training step (use existing models)"
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation step"
    )
    parser.add_argument(
        "--skip_forecast",
        action="store_true",
        help="Skip forecasting step"
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
    
    print("="*80)
    print("NHiTS Trade Forecasting Pipeline - Complete Run")
    print("="*80)
    print(f"Config directory: {config_dir}")
    print(f"Validation mode: {validation_config.mode}")
    print(f"Targets: {len(targets_config.targets)}")
    print("="*80)
    
    all_results = []
    
    # Step 1: Training
    if not args.skip_training:
        print("\n[STEP 1/3] Training models...")
        print("="*80)
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
    else:
        print("\n[STEP 1/3] Training models... SKIPPED")
        # Load existing metrics if available
        result_dir = Path(paths_config.result_dir)
        for target in targets_config.targets:
            r2_path = result_dir / f"{target}_full_r2.json"
            val_metrics_path = result_dir / f"{target}_val_metrics.json"
            r2_full = None
            val_metrics = None
            
            if r2_path.exists():
                with open(r2_path, "r") as f:
                    r2_data = json.load(f)
                    r2_full = r2_data.get("r2_full")
            
            if val_metrics_path.exists():
                with open(val_metrics_path, "r") as f:
                    val_metrics = json.load(f)
            
            all_results.append({
                "target": target,
                "val_metrics": val_metrics,
                "r2_full": r2_full
            })
    
    # Step 2: Evaluation
    if not args.skip_evaluation:
        print("\n[STEP 2/3] Evaluating models...")
        print("="*80)
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
                    # Update existing result or add new
                    for i, r in enumerate(all_results):
                        if r["target"] == target:
                            all_results[i] = result
                            break
                    else:
                        all_results.append(result)
            except Exception as e:
                print(f"Error evaluating {target}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        print("\n[STEP 2/3] Evaluating models... SKIPPED")
    
    # Step 3: Forecasting
    if not args.skip_forecast:
        print("\n[STEP 3/3] Generating forecasts...")
        print("="*80)
        for target in targets_config.targets:
            try:
                forecast_target(target, train_config, exog_config, paths_config)
            except Exception as e:
                print(f"Error forecasting {target}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        print("\n[STEP 3/3] Generating forecasts... SKIPPED")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Target':<20} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'R² Full':<12}")
    print("-"*80)
    
    for result in all_results:
        target = result["target"]
        val_metrics = result.get("val_metrics")
        r2_full = result.get("r2_full")
        
        if val_metrics:
            rmse = val_metrics.get("RMSE", "N/A")
            mae = val_metrics.get("MAE", "N/A")
            mape = val_metrics.get("MAPE", "N/A")
            r2_str = f"{r2_full:.4f}" if r2_full is not None else "N/A"
            
            if isinstance(rmse, (int, float)):
                print(f"{target:<20} {rmse:<12.3f} {mae:<12.3f} {mape:<12.3f} {r2_str:<12}")
            else:
                print(f"{target:<20} {rmse:<12} {mae:<12} {mape:<12} {r2_str:<12}")
        else:
            r2_str = f"{r2_full:.4f}" if r2_full is not None else "N/A"
            print(f"{target:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {r2_str:<12}")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    print(f"Results saved in: {paths_config.result_dir}/")
    print(f"Models saved in: {paths_config.model_dir}/")


if __name__ == "__main__":
    main()
