"""Orchestrates the complete Carbon-SLA-Net training pipeline.

Runs in sequence: BC pre-training → PPO fine-tuning → Pareto sweep.
Saves a JSON audit trail to results/pipeline_run_summary.json.

Run:
    python experiments/run_full_pipeline.py                   # full run (~4.5 h)
    python experiments/run_full_pipeline.py --fast            # fast run for testing
    python experiments/run_full_pipeline.py --skip-pareto     # BC + PPO only
    python experiments/run_full_pipeline.py --n-envs 4        # override n_envs
    python experiments/run_full_pipeline.py --timesteps 500000

Must be invoked as a script on macOS due to SubprocVecEnv 'spawn' requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, Optional

from experiments.configs.main_training import (
    TrainingConfig,
    get_default_config,
    get_fast_config,
)
from experiments.run_bc_training import run_bc_training
from experiments.run_rl_training import run_rl_training
from experiments.run_pareto_sweep import run_pareto_sweep


def run_full_pipeline(
    config: TrainingConfig,
    skip_pareto: bool = False,
    pareto_timesteps: int = 200_000,
) -> Dict:
    """Run BC pre-training → PPO fine-tuning → Pareto sweep in sequence.

    Parameters
    ----------
    config:
        Training configuration.  ``reward_weights`` must sum to 1.0.
    skip_pareto:
        If True, skip the Pareto sweep (BC + PPO only).
    pareto_timesteps:
        PPO timesteps per Pareto weight configuration.  Set to a small
        value (e.g. 2_048) for fast verification runs.

    Returns
    -------
    Dict with keys ``bc_result``, ``rl_result``, ``pareto_results``,
    and timing/summary metadata.

    Raises
    ------
    ValueError
        If ``config.reward_weights`` values do not sum to 1.0 (within 1e-6).
    """
    # Validate reward weights before starting
    weight_sum = sum(config.reward_weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"reward_weights must sum to 1.0; got {weight_sum:.6f}. "
            f"Weights: {config.reward_weights}"
        )

    pipeline_start = time.perf_counter()
    start_time_str = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  Carbon-SLA-Net Full Training Pipeline")
    print(f"  Started: {start_time_str}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Stage 1: BC pre-training
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    bc_result = run_bc_training(config)
    bc_elapsed = time.perf_counter() - t0
    print(f"\n[Pipeline] BC pre-training complete in {bc_elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # Stage 2: PPO fine-tuning
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    rl_result = run_rl_training(config, bc_result)
    rl_elapsed = time.perf_counter() - t0
    print(f"\n[Pipeline] PPO fine-tuning complete in {rl_elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # Stage 3: Pareto sweep (optional)
    # ------------------------------------------------------------------
    pareto_results = []
    pareto_elapsed = 0.0
    if not skip_pareto:
        t0 = time.perf_counter()
        pareto_results = run_pareto_sweep(
            config,
            bc_checkpoint_path=bc_result.checkpoint_path,
            timesteps_per_config=pareto_timesteps,
        )
        pareto_elapsed = time.perf_counter() - t0
        print(f"\n[Pipeline] Pareto sweep complete in {pareto_elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - pipeline_start
    end_time_str = time.strftime("%Y-%m-%dT%H:%M:%S")

    summary = {
        "start_time": start_time_str,
        "end_time": end_time_str,
        "total_duration_s": round(total_elapsed, 2),
        "bc": {
            "best_val_accuracy": round(bc_result.best_val_accuracy, 4),
            "best_val_loss": round(bc_result.best_val_loss, 4),
            "best_epoch": bc_result.best_epoch,
            "total_epochs_run": bc_result.total_epochs_run,
            "checkpoint_path": bc_result.checkpoint_path,
            "elapsed_s": round(bc_elapsed, 2),
        },
        "ppo": {
            "best_mean_reward": round(rl_result.best_mean_reward, 6),
            "final_mean_reward": round(rl_result.final_mean_reward, 6),
            "total_timesteps": rl_result.total_timesteps,
            "training_time_s": round(rl_result.training_time_s, 2),
            "best_model_path": rl_result.best_model_path,
            "elapsed_s": round(rl_elapsed, 2),
        },
        "pareto": {
            "n_configs": len(pareto_results),
            "elapsed_s": round(pareto_elapsed, 2),
            "results": [
                {
                    "weights": r["weights"],
                    "model_path": r["model_path"],
                    "mean_energy_wh": round(r["mean_energy_wh"], 4),
                    "mean_carbon_gco2": round(r["mean_carbon_gco2"], 4),
                    "mean_sla_violation": round(r["mean_sla_violation"], 4),
                }
                for r in pareto_results
            ],
        },
    }

    # Persist audit trail
    Path("results").mkdir(parents=True, exist_ok=True)
    summary_path = Path("results/pipeline_run_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"{'='*60}")
    print(f"  Pipeline complete.")
    print(f"  Total time  : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  BC accuracy : {bc_result.best_val_accuracy:.3f}")
    print(f"  PPO reward  : {rl_result.best_mean_reward:.4f}")
    print(f"  Model       : {rl_result.best_model_path}")
    print(f"  Summary     : {summary_path}")
    print(f"{'='*60}\n")

    return {
        "bc_result": bc_result,
        "rl_result": rl_result,
        "pareto_results": pareto_results,
        "summary": summary,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run full Carbon-SLA-Net training pipeline"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast config for testing"
    )
    parser.add_argument(
        "--skip-pareto", action="store_true", help="Skip Pareto sweep"
    )
    parser.add_argument(
        "--n-envs", type=int, default=None, help="Override ppo_n_envs"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Override PPO total_timesteps"
    )
    args = parser.parse_args()

    config = get_fast_config() if args.fast else get_default_config()
    if args.n_envs:
        config.ppo_n_envs = args.n_envs
    if args.timesteps:
        config.ppo_total_timesteps = args.timesteps

    run_full_pipeline(config, skip_pareto=args.skip_pareto)
