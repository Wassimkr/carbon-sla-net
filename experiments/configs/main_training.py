"""Training hyperparameter configuration for Carbon-SLA-Net Phase 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TrainingConfig:
    """All hyperparameters for a Carbon-SLA-Net training run.

    Mutable fields (reward_weights) are initialised in __post_init__ to
    avoid the Python dataclass shared-mutable-default pitfall.
    """

    # Environment
    N: int = 100
    T: int = 8
    renewable_condition: str = "sunny"
    workload_pattern: str = "uniform"

    # BC pre-training
    bc_dataset_path: str = "data/bc_dataset/bc_500.pkl"
    bc_epochs: int = 50
    bc_batch_size: int = 256
    bc_learning_rate: float = 1e-3
    bc_dropout: float = 0.1
    bc_patience: int = 10
    bc_checkpoint_dir: str = "checkpoints/bc"
    bc_run_name: str = "carbon_sla_net_bc"

    # PPO fine-tuning
    ppo_total_timesteps: int = 1_000_000
    ppo_n_envs: int = 2
    ppo_eval_freq: int = 10_000
    ppo_n_eval_episodes: int = 20
    ppo_checkpoint_dir: str = "checkpoints/ppo"
    ppo_model_name: str = "carbon_sla_net"
    ppo_seed: int = 42

    # Reward weights — set to None here; __post_init__ fills the default
    reward_weights: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.reward_weights is None:
            self.reward_weights = {
                "energy": 0.3,
                "carbon": 0.3,
                "sla": 0.3,
                "renewable": 0.1,
            }


def get_default_config() -> TrainingConfig:
    """Return a TrainingConfig with all production defaults."""
    return TrainingConfig()


def get_fast_config() -> TrainingConfig:
    """Return a reduced TrainingConfig for fast iteration and testing.

    BC: 5 epochs, patience 3.
    PPO: 10 000 steps, 1 env, eval every 5 000 steps, 3 eval episodes.
    N: 50 (smaller env for speed).
    """
    return TrainingConfig(
        N=50,
        bc_epochs=5,
        bc_patience=3,
        ppo_total_timesteps=10_000,
        ppo_n_envs=1,
        ppo_eval_freq=5_000,
        ppo_n_eval_episodes=3,
    )
