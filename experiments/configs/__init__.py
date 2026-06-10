"""Training configuration dataclasses for Carbon-SLA-Net."""

from experiments.configs.main_training import (
    TrainingConfig,
    get_default_config,
    get_fast_config,
)
from experiments.configs.ablation_configs import (
    AblationConfig,
    get_ablation_configs,
)

__all__ = [
    "TrainingConfig",
    "get_default_config",
    "get_fast_config",
    "AblationConfig",
    "get_ablation_configs",
]
