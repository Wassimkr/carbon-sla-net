"""Ablation study configurations for Carbon-SLA-Net Phase 5."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional

from experiments.configs.main_training import TrainingConfig, get_default_config


@dataclass
class AblationConfig:
    """Configuration for one ablation variant of Carbon-SLA-Net.

    Boolean flags control which component is removed; all False means
    the full model with all features enabled.
    """

    variant_name: str
    base_config: TrainingConfig
    zero_carbon_in_obs: bool = False
    zero_battery_in_obs: bool = False
    skip_bc_pretrain: bool = False
    disable_action_masking: bool = False
    reward_weights: Optional[Dict[str, float]] = None


def get_ablation_configs() -> List[AblationConfig]:
    """Return one AblationConfig per ablation variant (5 total).

    All ablations use ppo_total_timesteps=200_000 and ppo_n_envs=2,
    which is reduced from the full run but consistent across variants.
    """
    ablation_base = dataclasses.replace(
        get_default_config(),
        ppo_total_timesteps=200_000,
        ppo_n_envs=2,
    )

    return [
        AblationConfig(
            variant_name="full",
            base_config=dataclasses.replace(ablation_base),
        ),
        AblationConfig(
            variant_name="no_carbon_signal",
            base_config=dataclasses.replace(ablation_base),
            zero_carbon_in_obs=True,
        ),
        AblationConfig(
            variant_name="no_battery_signal",
            base_config=dataclasses.replace(ablation_base),
            zero_battery_in_obs=True,
        ),
        AblationConfig(
            variant_name="no_bc_pretrain",
            base_config=dataclasses.replace(ablation_base),
            skip_bc_pretrain=True,
        ),
        AblationConfig(
            variant_name="no_masking",
            base_config=dataclasses.replace(ablation_base),
            disable_action_masking=True,
        ),
    ]
