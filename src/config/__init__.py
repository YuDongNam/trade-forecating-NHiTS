# Legacy imports (for backward compatibility)
from .base_config import TrainConfig as LegacyTrainConfig, ExogenousConfig as LegacyExogenousConfig
from .paths import RAW_DATA_DIR, RESULT_DIR, MODEL_DIR
from .targets import TARGETS

# New YAML-based configs
from .yaml_loader import (
    TrainConfig,
    ExogenousConfig,
    ValidationConfig,
    PathsConfig,
    ModelNHITSConfig,
    TargetsConfig,
    load_paths_config,
    load_train_config,
    load_validation_config,
    load_exogenous_config,
    load_targets_config,
    load_model_config,
)

__all__ = [
    # Legacy
    "LegacyTrainConfig",
    "LegacyExogenousConfig",
    "RAW_DATA_DIR",
    "RESULT_DIR",
    "MODEL_DIR",
    "TARGETS",
    # New YAML-based
    "TrainConfig",
    "ExogenousConfig",
    "ValidationConfig",
    "PathsConfig",
    "ModelNHITSConfig",
    "TargetsConfig",
    "load_paths_config",
    "load_train_config",
    "load_validation_config",
    "load_exogenous_config",
    "load_targets_config",
    "load_model_config",
]

