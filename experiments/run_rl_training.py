"""Phase 5b: MaskablePPO fine-tuning script.

Run:
    python experiments/run_rl_training.py           # production config
    python experiments/run_rl_training.py --fast    # fast config for testing
    python experiments/run_rl_training.py --no-bc   # skip BC warm-start

Must be invoked as a script (not imported at module level) on macOS due to
the 'spawn' multiprocessing start method required by SubprocVecEnv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.bc_trainer import BCTrainingResult
from agent.rl_trainer import RLTrainer, RLTrainingResult
from experiments.configs.main_training import (
    TrainingConfig,
    get_default_config,
    get_fast_config,
)


def run_rl_training(
    config: TrainingConfig,
    bc_result: Optional[BCTrainingResult] = None,
) -> RLTrainingResult:
    """Run MaskablePPO fine-tuning and return a summary result.

    BC checkpoint resolution order:
    1. ``bc_result.checkpoint_path`` if *bc_result* is provided.
    2. ``{config.bc_checkpoint_dir}/{config.bc_run_name}_best.pt`` if it exists.
    3. No warm-start (random initialisation) — a warning is printed.

    Parameters
    ----------
    config:
        A :class:`TrainingConfig` controlling all PPO hyperparameters.
    bc_result:
        Optional result from a preceding BC training run.  If supplied,
        its ``checkpoint_path`` is used for weight transfer.

    Returns
    -------
    :class:`~agent.rl_trainer.RLTrainingResult` with the best model path
    and training summary.
    """
    print("=== Phase 5b: PPO Fine-tuning ===")

    # Resolve BC checkpoint
    bc_checkpoint_path: Optional[str] = None
    if bc_result is not None:
        bc_checkpoint_path = bc_result.checkpoint_path
        print(f"  BC warm-start: {bc_checkpoint_path}")
    else:
        candidate = (
            Path(config.bc_checkpoint_dir) / f"{config.bc_run_name}_best.pt"
        )
        if candidate.exists():
            bc_checkpoint_path = str(candidate)
            print(f"  BC warm-start: {bc_checkpoint_path} (auto-detected)")
        else:
            print(
                "  WARNING: No BC checkpoint found — training from random init"
            )

    print(f"  N={config.N}  T={config.T}  n_envs={config.ppo_n_envs}")
    print(f"  Total timesteps : {config.ppo_total_timesteps:,}")
    print(f"  Eval freq       : {config.ppo_eval_freq:,}  "
          f"n_eval_eps={config.ppo_n_eval_episodes}")
    print(f"  Reward weights  : {config.reward_weights}")
    print(f"  Out dir         : {config.ppo_checkpoint_dir}")
    print()

    trainer = RLTrainer(
        N=config.N,
        T=config.T,
        n_envs=config.ppo_n_envs,
        reward_weights=config.reward_weights,
        workload_pattern=config.workload_pattern,
        renewable_condition=config.renewable_condition,
        checkpoint_dir=config.ppo_checkpoint_dir,
        seed=config.ppo_seed,
    )

    result = trainer.train(
        total_timesteps=config.ppo_total_timesteps,
        eval_freq=config.ppo_eval_freq,
        n_eval_episodes=config.ppo_n_eval_episodes,
        bc_checkpoint_path=bc_checkpoint_path,
        model_name=config.ppo_model_name,
    )

    elapsed_min = result.training_time_s / 60.0
    print()
    print("PPO training complete.")
    print(f"  Total timesteps  : {result.total_timesteps:,}")
    print(f"  Best mean reward : {result.best_mean_reward:.4f}")
    print(f"  Training time    : {result.training_time_s:.1f}s ({elapsed_min:.1f} min)")
    print(f"  Best model       : {result.best_model_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5b: PPO fine-tuning for Carbon-SLA-Net"
    )
    parser.add_argument("--fast", action="store_true", help="Use fast config for testing")
    parser.add_argument("--no-bc", action="store_true", help="Skip BC warm-start")
    args = parser.parse_args()

    config = get_fast_config() if args.fast else get_default_config()

    bc_result = None
    if not args.no_bc:
        bc_path = Path(config.bc_checkpoint_dir) / f"{config.bc_run_name}_best.pt"
        if not bc_path.exists():
            print("BC checkpoint not found — run run_bc_training.py first")
        # bc_result stays None; run_rl_training handles missing checkpoint gracefully

    run_rl_training(config, bc_result)
