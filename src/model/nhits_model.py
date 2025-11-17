from neuralforecast.models import NHITS
from ..config.yaml_loader import TrainConfig


def create_nhits(config: TrainConfig, exog_dim: int = 0) -> NHITS:
    """
    Create and configure an NHITS model.
    
    Args:
        config: Training configuration
        exog_dim: Number of exogenous variables (optional, auto-detected from data)
    
    Returns:
        Configured NHITS model
    """
    # neuralforecast automatically detects exogenous variables from the dataframe
    # n_x parameter is not needed and causes errors in some versions
    model_kwargs = {
        "h": config.horizon,
        "input_size": config.input_size,
        "max_steps": config.max_steps,
        "scaler_type": "standard",
        "batch_size": config.batch_size,
        "learning_rate": config.lr,
    }
    
    # Only add n_x if exog_dim > 0 and the version supports it
    # For now, let neuralforecast auto-detect from data
    # if exog_dim > 0:
    #     model_kwargs["n_x"] = exog_dim
    
    return NHITS(**model_kwargs)

