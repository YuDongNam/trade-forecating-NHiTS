"""
YAML configuration loader.
"""
import yaml
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class PathsConfig:
    raw_data_dir: str
    result_dir: str
    model_dir: str


@dataclass
class TrainConfig:
    input_size: int
    horizon: int
    max_steps: int
    batch_size: int
    lr: float
    scaler: str
    use_gpu: bool
    dropout_rate: float = 0.1
    mc_samples: int = 100


@dataclass
class ValidationConfig:
    mode: str  # "tail", "range", or "none"
    tail_months: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None


@dataclass
class ModelNHITSConfig:
    num_stacks: int
    num_blocks: int
    num_layers: int
    activation: str


@dataclass
class TargetsConfig:
    targets: List[str]


@dataclass
class ExogenousConfig:
    use_tariff: bool
    use_fx: bool
    use_wti: bool
    use_copper: bool


@dataclass
class UncertaintyConfig:
    method: str  # "mc_dropout" or other methods
    enabled: bool
    n_samples: int
    ci_level: float


def load_yaml(config_path: Path) -> dict:
    """Load a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths_config(config_dir: Path = Path("config")) -> PathsConfig:
    """Load paths configuration."""
    config_path = config_dir / "paths.yaml"
    data = load_yaml(config_path)
    return PathsConfig(**data)


def load_train_config(config_dir: Path = Path("config")) -> TrainConfig:
    """Load training configuration."""
    config_path = config_dir / "train.yaml"
    data = load_yaml(config_path)
    return TrainConfig(**data)


def load_validation_config(config_dir: Path = Path("config")) -> ValidationConfig:
    """Load validation configuration."""
    config_path = config_dir / "validation.yaml"
    data = load_yaml(config_path)
    return ValidationConfig(**data)


def load_model_config(config_dir: Path = Path("config")) -> ModelNHITSConfig:
    """Load model-specific configuration."""
    config_path = config_dir / "model_nhits.yaml"
    data = load_yaml(config_path)
    return ModelNHITSConfig(**data)


def load_targets_config(config_dir: Path = Path("config")) -> TargetsConfig:
    """Load targets configuration."""
    config_path = config_dir / "targets.yaml"
    data = load_yaml(config_path)
    return TargetsConfig(**data)


def load_exogenous_config(config_dir: Path = Path("config")) -> ExogenousConfig:
    """Load exogenous variables configuration."""
    config_path = config_dir / "exogenous.yaml"
    data = load_yaml(config_path)
    return ExogenousConfig(**data)


def load_uncertainty_config(config_dir: Path = Path("config")) -> UncertaintyConfig:
    """Load uncertainty quantification configuration."""
    config_path = config_dir / "uncertainty.yaml"
    data = load_yaml(config_path)
    return UncertaintyConfig(**data)

