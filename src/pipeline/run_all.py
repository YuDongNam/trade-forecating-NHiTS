"""
Multi-target runner script that trains, evaluates, and forecasts for all targets.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .train import main as train_main
from .evaluate import main as evaluate_main
from .forecast import main as forecast_main


def main():
    """Run training, evaluation, and forecasting for all targets."""
    print("="*60)
    print("NHiTS Trade Forecasting Pipeline")
    print("="*60)
    
    # Step 1: Train models
    print("\n[STEP 1/3] Training models...")
    train_main()
    
    # Step 2: Evaluate models
    print("\n[STEP 2/3] Evaluating models...")
    evaluate_main()
    
    # Step 3: Generate forecasts
    print("\n[STEP 3/3] Generating forecasts...")
    forecast_main()
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print(f"Results saved in: results/")
    print(f"Models saved in: models/")


if __name__ == "__main__":
    main()

