from neuralforecast.models import NHITS
from ..config.yaml_loader import TrainConfig


def create_nhits(config: TrainConfig, exog_dim: int = 0, dropout_rate: float = 0.1) -> NHITS:
    """
    Create and configure an NHITS model with dropout for Monte Carlo Dropout.
    
    Args:
        config: Training configuration
        exog_dim: Number of exogenous variables (optional, auto-detected from data)
        dropout_rate: Dropout rate for Monte Carlo Dropout (default: 0.1)
    
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
    
    # Add dropout if supported by NHITS
    # Note: Some versions of neuralforecast support dropout_rate parameter
    if dropout_rate > 0:
        # Try to add dropout_rate if the model supports it
        try:
            model_kwargs["dropout_rate"] = dropout_rate
        except TypeError:
            # If dropout_rate is not supported, model will use default dropout
            pass
    
    # Create model
    model = NHITS(**model_kwargs)
    
    # Only add n_x if exog_dim > 0 and the version supports it
    # For now, let neuralforecast auto-detect from data
    # if exog_dim > 0:
    #     model_kwargs["n_x"] = exog_dim
    
    return model

