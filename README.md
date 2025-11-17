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
├─ config/                      # YAML configuration files
│   ├─ paths.yaml              # Directory paths
│   ├─ train.yaml              # Training configuration
│   ├─ validation.yaml         # Validation configuration
│   ├─ model_nhits.yaml        # Model-specific parameters
│   ├─ targets.yaml            # Target series definitions
│   └─ exogenous.yaml          # Exogenous variables configuration
│
├─ src/
│   ├─ config/
│   │   ├─ yaml_loader.py      # YAML configuration loader
│   │   ├─ base_config.py      # Legacy configs (deprecated)
│   │   ├─ paths.py            # Legacy paths (deprecated)
│   │   └─ targets.py          # Legacy targets (deprecated)
│   ├─ data/
│   │   ├─ load_data.py        # Data loading and merging
│   │   ├─ preprocess.py       # Train/val split and scaling
│   │   └─ feature_engineering.py  # Feature selection and engineering
│   ├─ model/
│   │   └─ nhits_model.py      # NHiTS model creation
│   └─ pipeline/
│       ├─ train.py            # Training script
│       ├─ evaluate.py         # Evaluation script
│       ├─ forecast.py         # Forecasting script
│       ├─ run_all.py          # End-to-end pipeline runner
│       ├─ metrics.py          # Metrics computation (RMSE, MAE, MAPE, R²)
│       └─ plotting.py         # Plotting utilities
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

## Configuration

The pipeline uses **YAML-based configuration** for flexible experiment management. All configuration files are located in the `config/` directory.

### Configuration Files

- **`config/paths.yaml`**: Directory paths (data, results, models)
- **`config/train.yaml`**: Training parameters (input_size, horizon, max_steps, batch_size, lr, etc.)
- **`config/validation.yaml`**: Validation configuration (mode: tail/range/none, tail_months, start/end dates)
- **`config/model_nhits.yaml`**: Model-specific parameters (num_stacks, num_blocks, etc.)
- **`config/targets.yaml`**: List of target series to process
- **`config/exogenous.yaml`**: Exogenous variables configuration (use_tariff, use_fx, use_wti, use_copper)

### Validation Modes

The validation configuration supports three modes:

1. **`tail`**: Use the last N months as validation (e.g., last 24 months)
2. **`range`**: Use a specific date range for validation (e.g., 2023-01-01 to 2023-12-01)
3. **`none`**: No validation split (use all data for training)

Edit `config/validation.yaml` to change the validation strategy.

## Model Description

The pipeline uses the **NHiTS** (Neural Hierarchical Interpolation for Time Series) model from the `neuralforecast` library.

### Default Model Configuration

- **Input Size**: 36 months (3 years of history)
- **Forecast Horizon**: 12 months
- **Training Steps**: 1500
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **Scaler**: StandardScaler

These can be modified in `config/train.yaml`.

### Exogenous Variables

The model incorporates the following exogenous variables (configurable in `config/exogenous.yaml`):
- Tariff rates and related features
- Foreign exchange rates (country-specific)
- WTI crude oil prices
- Copper futures prices

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd NHiTS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Google Colab Installation

Colab에서 실행하려면 `colab_setup.ipynb` 노트북을 사용하거나 `COLAB_GUIDE.md`를 참고하세요.

**빠른 시작:**
1. Colab에서 새 노트북 생성
2. `colab_setup.ipynb` 업로드 또는 아래 코드 실행:
```python
# 디렉토리 생성
import os
os.makedirs("data", exist_ok=True)
os.makedirs("src/config", exist_ok=True)
os.makedirs("src/data", exist_ok=True)
os.makedirs("src/model", exist_ok=True)
os.makedirs("src/pipeline", exist_ok=True)

# 패키지 설치
!pip install neuralforecast torch pandas numpy scikit-learn matplotlib pydantic tqdm

# 파일 업로드 (Colab 파일 메뉴 사용)
# - src/ 폴더 전체
# - data/ 폴더의 모든 CSV 파일

# 실행
!python -m src.pipeline.run_all
```

자세한 내용은 `COLAB_GUIDE.md`를 참고하세요.

## Usage

### Configuration

All configuration is done through YAML files in the `config/` directory. Edit these files to customize:

- Training parameters: `config/train.yaml`
- Validation strategy: `config/validation.yaml`
- Target series: `config/targets.yaml`
- Exogenous variables: `config/exogenous.yaml`

### Train Models

Train models for all targets:
```bash
# Use default config directory
python -m src.pipeline.train

# Specify custom config directory
python -m src.pipeline.train --config_dir config

# Override validation mode via CLI
python -m src.pipeline.train --val_mode range --val_start 2023-01-01 --val_end 2023-12-01
```

### Evaluate Models

Evaluate models and generate metrics/plots:
```bash
# Use default config
python -m src.pipeline.evaluate

# Override validation configuration
python -m src.pipeline.evaluate --val_mode tail --val_tail_months 12
```

### Generate Forecasts

Generate forecasts for all targets:
```bash
python -m src.pipeline.forecast --config_dir config
```

### Run Complete Pipeline

Run training, evaluation, and forecasting in sequence:
```bash
# Full pipeline
python -m src.pipeline.run_all --config_dir config

# Skip specific steps
python -m src.pipeline.run_all --skip_training  # Use existing models
python -m src.pipeline.run_all --skip_evaluation
python -m src.pipeline.run_all --skip_forecast
```

## Output

### Models
- Model checkpoints are saved in `models/{target}/` (configured in `config/paths.yaml`)
- Scalers are saved as `models/{target}/scalers.pkl`

### Results
- **Validation Metrics**: `results/{target}_val_metrics.json` - RMSE, MAE, MAPE, R² for validation period
- **Full-Period R²**: `results/{target}_full_r2.json` - R² computed over entire training period
- **Forecasts**: `results/{target}_forecast.csv` - Contains historical data and future forecasts with confidence intervals
- **Validation Details**: `results/{target}_validation.csv` - Detailed validation results with errors
- **Plots**: `results/{target}_forecast.png` - Visualization of actual vs forecasted values with 95% confidence intervals
- **Summary**: `results/summary_metrics.json` - Aggregated metrics for all targets

### Metrics Explained

- **RMSE** (Root Mean Squared Error): Measures average prediction error magnitude
- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual values
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
- **R²** (Coefficient of Determination): Measures how well the model explains the variance in the data
  - R² = 1.0: Perfect predictions
  - R² = 0.0: Model performs as well as predicting the mean
  - R² < 0.0: Model performs worse than predicting the mean

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

## Advanced Configuration

### Changing Validation Period

Edit `config/validation.yaml`:

```yaml
# Use last 24 months
mode: "tail"
tail_months: 24

# Or use specific date range
mode: "range"
start: "2023-01-01"
end: "2023-12-01"

# Or no validation split
mode: "none"
```

You can also override via CLI arguments (see Usage section above).

### Custom Training Parameters

Edit `config/train.yaml` to adjust:
- `input_size`: Number of past months to use (default: 36)
- `horizon`: Forecast horizon in months (default: 12)
- `max_steps`: Maximum training steps (default: 1500)
- `batch_size`: Batch size (default: 32)
- `lr`: Learning rate (default: 0.001)

## Dependencies

- `neuralforecast`: NHiTS model implementation
- `torch`: Deep learning backend
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Data preprocessing (StandardScaler)
- `matplotlib`: Plotting
- `pydantic`: Configuration validation
- `tqdm`: Progress bars
- `pyyaml`: YAML configuration file parsing

## Notes

- The pipeline automatically handles different CSV column name formats
- Missing exogenous values are forward-filled and back-filled
- Each target uses country-specific FX rates (Korea→USD_KRW, China→USD_CNY, Taiwan→USD_TWD)
- World targets do not use specific FX rates
- Validation configuration is fully controlled via `config/validation.yaml` or CLI arguments
- All metrics (RMSE, MAE, MAPE, R²) are computed using shared utilities in `src/pipeline/metrics.py`
- Plots include 95% confidence intervals when available

## License

[Add your license information here]

