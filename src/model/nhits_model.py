from neuralforecast.models import NHITS
from ..config.base_config import TrainConfig


def create_nhits(config: TrainConfig, exog_dim: int) -> NHITS:
    """
    Create and configure an NHITS model.
    
    Args:
        config: Training configuration
        exog_dim: Number of exogenous variables
    
    Returns:
        Configured NHITS model
    """
    return NHITS(
        h=config.horizon,
        input_size=config.input_size,
        max_steps=config.max_steps,
        scaler_type="standard",
        batch_size=config.batch_size,
        learning_rate=config.lr,
        n_x=exog_dim,
    )

