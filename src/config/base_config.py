from pydantic import BaseModel


class TrainConfig(BaseModel):
    input_size: int = 36         # months of history (3 years)
    horizon: int = 12            # forecast horizon in months
    max_steps: int = 1500
    batch_size: int = 32
    lr: float = 1e-3
    scaler: str = "standard"


class ExogenousConfig(BaseModel):
    use_tariff: bool = True
    use_fx: bool = True
    use_wti: bool = True
    use_copper: bool = True

