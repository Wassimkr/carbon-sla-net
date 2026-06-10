"""Tests for Phase 5: experiment configs, BC training, PPO training, Pareto sweep."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments.configs.main_training import (
    TrainingConfig,
    get_default_config,
    get_fast_config,
)
from experiments.configs.ablation_configs import (
    AblationConfig,
    get_ablation_configs,
)
from experiments.run_pareto_sweep import PARETO_WEIGHT_GRID
from evaluation.ablation import AblationEvaluator


# ---------------------------------------------------------------------------
# 1–4: TrainingConfig tests
# ---------------------------------------------------------------------------

def test_get_default_config_n():
    """1. get_default_config() returns TrainingConfig with N=100."""
    cfg = get_default_config()
    assert isinstance(cfg, TrainingConfig)
    assert cfg.N == 100


def test_get_fast_config_n_and_timesteps():
    """2. get_fast_config() returns TrainingConfig with N=50 and ppo_total_timesteps=10_000."""
    cfg = get_fast_config()
    assert isinstance(cfg, TrainingConfig)
    assert cfg.N == 50
    assert cfg.ppo_total_timesteps == 10_000


def test_default_reward_weights_set_by_post_init():
    """3. TrainingConfig.__post_init__ sets reward_weights to default dict when None."""
    cfg = TrainingConfig()
    assert cfg.reward_weights is not None
    assert isinstance(cfg.reward_weights, dict)
    assert set(cfg.reward_weights.keys()) == {"energy", "carbon", "sla", "renewable"}


def test_default_reward_weights_sum_to_one():
    """4. reward_weights values sum to 1.0 in the default config."""
    cfg = get_default_config()
    total = sum(cfg.reward_weights.values())
    assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 5–8: AblationConfig tests
# ---------------------------------------------------------------------------

def test_get_ablation_configs_count():
    """5. get_ablation_configs() returns exactly 5 AblationConfig objects."""
    configs = get_ablation_configs()
    assert len(configs) == 5
    assert all(isinstance(c, AblationConfig) for c in configs)


def test_ablation_variant_names_match_evaluator():
    """6. Ablation variant names match AblationEvaluator.VARIANTS exactly."""
    configs = get_ablation_configs()
    names = [c.variant_name for c in configs]
    assert names == AblationEvaluator.VARIANTS


def test_full_variant_all_flags_false():
    """7. variant_name='full' has all boolean flags False."""
    configs = get_ablation_configs()
    full = next(c for c in configs if c.variant_name == "full")
    assert not full.zero_carbon_in_obs
    assert not full.zero_battery_in_obs
    assert not full.skip_bc_pretrain
    assert not full.disable_action_masking


def test_no_bc_pretrain_variant_flags():
    """8. variant_name='no_bc_pretrain' has skip_bc_pretrain=True, all others False."""
    configs = get_ablation_configs()
    variant = next(c for c in configs if c.variant_name == "no_bc_pretrain")
    assert variant.skip_bc_pretrain
    assert not variant.zero_carbon_in_obs
    assert not variant.zero_battery_in_obs
    assert not variant.disable_action_masking


# ---------------------------------------------------------------------------
# 9–10: BC training integration tests (marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_bc_training_returns_result():
    """9. run_bc_training(get_fast_config()) completes and returns BCTrainingResult."""
    from agent.bc_trainer import BCTrainingResult
    from experiments.run_bc_training import run_bc_training

    result = run_bc_training(get_fast_config())
    assert isinstance(result, BCTrainingResult)
    assert result.best_val_loss < float("inf")
    assert result.best_epoch >= 1
    assert result.total_epochs_run >= 1
    assert result.checkpoint_path.endswith(".pt")


@pytest.mark.slow
def test_bc_checkpoint_file_exists(tmp_path):
    """10. After BC training, the checkpoint file exists at BCTrainingResult.checkpoint_path."""
    import dataclasses
    from experiments.run_bc_training import run_bc_training

    cfg = dataclasses.replace(
        get_fast_config(),
        bc_checkpoint_dir=str(tmp_path / "bc"),
        bc_run_name="test_bc",
    )
    result = run_bc_training(cfg)
    assert Path(result.checkpoint_path).exists(), (
        f"Checkpoint not found: {result.checkpoint_path}"
    )


# ---------------------------------------------------------------------------
# 11–12: PPO training integration tests (marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_rl_training_returns_result():
    """11. run_rl_training(get_fast_config()) completes and returns RLTrainingResult."""
    from agent.rl_trainer import RLTrainingResult
    from experiments.run_rl_training import run_rl_training

    result = run_rl_training(get_fast_config())
    assert isinstance(result, RLTrainingResult)
    assert result.total_timesteps == get_fast_config().ppo_total_timesteps
    assert isinstance(result.best_mean_reward, float)
    assert result.training_time_s > 0.0


@pytest.mark.slow
def test_ppo_checkpoint_file_exists():
    """12. After PPO training with fast config, a best_model.zip checkpoint exists."""
    from experiments.run_rl_training import run_rl_training

    result = run_rl_training(get_fast_config())
    assert Path(result.best_model_path).exists(), (
        f"PPO checkpoint not found: {result.best_model_path}"
    )


# ---------------------------------------------------------------------------
# 13–15: PARETO_WEIGHT_GRID tests
# ---------------------------------------------------------------------------

def test_pareto_weight_grid_count():
    """13. PARETO_WEIGHT_GRID has exactly 7 entries."""
    assert len(PARETO_WEIGHT_GRID) == 7


def test_pareto_weight_grid_keys():
    """14. All PARETO_WEIGHT_GRID entries have keys: energy, carbon, sla, renewable."""
    required = {"energy", "carbon", "sla", "renewable"}
    for i, entry in enumerate(PARETO_WEIGHT_GRID):
        assert set(entry.keys()) == required, (
            f"Entry {i} has wrong keys: {set(entry.keys())}"
        )


def test_pareto_weight_grid_sums_to_one():
    """15. All PARETO_WEIGHT_GRID weight values sum to 1.0 (within 1e-9)."""
    for i, entry in enumerate(PARETO_WEIGHT_GRID):
        total = sum(entry.values())
        assert abs(total - 1.0) < 1e-9, (
            f"Entry {i} weights sum to {total:.9f}, expected 1.0"
        )
