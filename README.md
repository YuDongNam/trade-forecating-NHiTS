# NHiTS Trade Forecasting

A production-grade forecasting pipeline for predicting trade balance using the NHiTS (Neural Hierarchical Interpolation for Time Series) model. This project forecasts trade balance for 8 target series (Korea, China, Taiwan, and World - each with Import and Export) using exogenous variables including tariff rates, foreign exchange rates, WTI oil prices, and copper prices.

## Project Structure

```
trade-forecasting/
│
├─ data/
│   ├─ China_Export.csv
│   ├─ China_Import.csv
│   ├─ Korea_Export.csv
│   ├─ Korea_Import.csv
│   ├─ Taiwan_Export.csv
│   ├─ Taiwan_Import.csv
│   ├─ World_Export.csv
│   ├─ World_Import.csv
│   │
│   ├─ tariff.csv
│   ├─ USD_KRW.csv
│   ├─ USD_CNY.csv
│   ├─ USD_TWD.csv
│   ├─ WTI.csv
│   ├─ copper.csv
│
├─ src/
│   ├─ config/
│   │   ├─ base_config.py      # Training and exogenous configurations
│   │   ├─ paths.py            # Directory paths
│   │   ├─ targets.py          # Target series definitions
│   ├─ data/
│   │   ├─ load_data.py        # Data loading and merging
│   │   ├─ preprocess.py       # Train/val split and scaling
│   │   ├─ feature_engineering.py  # Feature selection and engineering
│   ├─ model/
│   │   ├─ nhits_model.py      # NHiTS model creation
│   ├─ pipeline/
│   │   ├─ train.py            # Training script
│   │   ├─ evaluate.py         # Evaluation script
│   │   ├─ forecast.py         # Forecasting script
│   │   ├─ run_all.py          # Multi-target runner
│
├─ models/                      # Saved model checkpoints
├─ results/                     # Forecasts, metrics, and plots
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## Data Format

Each of the 8 target CSV files should contain at minimum:
- `ds`: Date column (monthly, datetime format)
- `y`: Target trade balance value (numeric)

**Note:** Some files may use `Date` and `PrimaryValue` column names, which are automatically normalized during loading.

### Exogenous Variables

- **tariff.csv**: Contains tariff-related features (tariff_rate, annc_pulse, eff_step, rate_change)
- **USD_KRW.csv, USD_CNY.csv, USD_TWD.csv**: Foreign exchange rates (used based on target country)
- **WTI.csv**: WTI crude oil prices (Close column)
- **copper.csv**: Copper futures prices (Close column)

## Model Description

The pipeline uses the **NHiTS** (Neural Hierarchical Interpolation for Time Series) model from the `neuralforecast` library.

### Model Configuration

- **Input Size**: 36 months (3 years of history)
- **Forecast Horizon**: 12 months
- **Training Steps**: 1500
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **Scaler**: StandardScaler

### Exogenous Variables

The model incorporates the following exogenous variables:
- Tariff rates and related features
- Foreign exchange rates (country-specific)
- WTI crude oil prices
- Copper futures prices

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd NHiTS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train Models

Train models for all targets:
```bash
python -m src.pipeline.train
```

### Evaluate Models

Evaluate models and generate metrics/plots:
```bash
python -m src.pipeline.evaluate
```

### Generate Forecasts

Generate forecasts for all targets:
```bash
python -m src.pipeline.forecast
```

### Run Complete Pipeline

Run training, evaluation, and forecasting in sequence:
```bash
python -m src.pipeline.run_all
```

## Output

### Models
- Model checkpoints are saved in `models/{target}/`
- Scalers are saved as `models/{target}/scalers.pkl`

### Results
- **Forecasts**: `results/{target}_forecast.csv` - Contains historical data and future forecasts
- **Metrics**: `results/{target}_metrics.json` - RMSE, MAE, MAPE metrics
- **Plots**: `results/{target}_forecast.png` - Visualization of actual vs forecasted values
- **Summary**: `results/summary_metrics.json` - Aggregated metrics for all targets

## GPU Support

The pipeline uses PyTorch as the backend for neuralforecast. If a GPU is available, PyTorch will automatically utilize it for training. To ensure GPU support:

1. Install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

2. Verify GPU availability:
```python
import torch
print(torch.cuda.is_available())
```

## Configuration

Model and data configurations can be modified in:
- `src/config/base_config.py`: Training parameters and exogenous variable settings
- `src/config/targets.py`: List of target series

## Dependencies

- `neuralforecast`: NHiTS model implementation
- `torch`: Deep learning backend
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Data preprocessing (StandardScaler)
- `matplotlib`: Plotting
- `pydantic`: Configuration management
- `tqdm`: Progress bars

## Notes

- The pipeline automatically handles different CSV column name formats
- Missing exogenous values are forward-filled and back-filled
- Each target uses country-specific FX rates (Korea→USD_KRW, China→USD_CNY, Taiwan→USD_TWD)
- World targets do not use specific FX rates
- Validation set uses the last 24 months by default

## License

[Add your license information here]

