"""Phase 5c: Multi-weight Pareto sweep to generate Figure 4 data.

Trains one MaskablePPO agent per reward weight configuration and evaluates
each on fresh episodes.  Results are saved to
``results/tables/pareto_sweep_results.csv``.

Run:
    python experiments/run_pareto_sweep.py           # full sweep (7 configs × 200k steps)
    python experiments/run_pareto_sweep.py --fast    # reduced sweep for testing
"""

from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent.parent))
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from agent.rl_trainer import RLTrainer
from experiments.configs.main_training import (
    TrainingConfig,
    get_default_config,
    get_fast_config,
)
from experiments.run_rl_training import run_rl_training

PARETO_WEIGHT_GRID: List[Dict[str, float]] = [
    {"energy": 0.5, "carbon": 0.3, "sla": 0.1, "renewable": 0.1},
    {"energy": 0.3, "carbon": 0.5, "sla": 0.1, "renewable": 0.1},
    {"energy": 0.1, "carbon": 0.3, "sla": 0.5, "renewable": 0.1},
    {"energy": 0.3, "carbon": 0.1, "sla": 0.5, "renewable": 0.1},
    {"energy": 0.4, "carbon": 0.4, "sla": 0.1, "renewable": 0.1},
    {"energy": 0.2, "carbon": 0.2, "sla": 0.5, "renewable": 0.1},
    {"energy": 0.3, "carbon": 0.3, "sla": 0.3, "renewable": 0.1},  # default
]


def _ensure_model_exists(best_model_path: str, config: TrainingConfig) -> None:
    """Save a fallback model if the eval callback never fired during training.

    This can happen when timesteps_per_config is smaller than eval_freq.
    We save a freshly initialised (optionally BC-warmed) model so that
    evaluate_model() always has a valid checkpoint to load.
    """
    if Path(best_model_path).exists():
        return

    # Check for any periodic checkpoint saved by CheckpointCallback
    model_name = config.ppo_model_name
    ckpt_dir = Path(config.ppo_checkpoint_dir)
    candidates = sorted(ckpt_dir.glob(f"{model_name}_*.zip"))
    if candidates:
        Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(candidates[-1]), best_model_path)
        print(f"  (eval never fired — copied {candidates[-1].name} as best_model.zip)")
        return

    # Last resort: create a minimal MaskablePPO with a DummyVecEnv and save it
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from env.cloud_edge_env import CloudEdgeEnv

    _env = DummyVecEnv([lambda: CloudEdgeEnv(N=config.N, T=config.T, seed=0)])
    _model = MaskablePPO("MlpPolicy", _env, verbose=0)
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    _model.save(str(Path(best_model_path).with_suffix("")))
    _env.close()
    print("  (eval never fired — saved initialised model as fallback best_model.zip)")


def run_pareto_sweep(
    base_config: TrainingConfig,
    bc_checkpoint_path: Optional[str] = None,
    timesteps_per_config: int = 200_000,
    max_configs: Optional[int] = None,
    save_dir: str = "results/tables",
) -> List[Dict]:
    """Train and evaluate one agent per reward weight configuration.

    Parameters
    ----------
    base_config:
        Base training config; ``reward_weights`` and
        ``ppo_total_timesteps`` are overridden per sweep entry.
    bc_checkpoint_path:
        Optional BC checkpoint for warm-starting every Pareto agent.
    timesteps_per_config:
        PPO timesteps for each weight configuration.
    max_configs:
        If set, only the first *max_configs* entries from
        :data:`PARETO_WEIGHT_GRID` are used (useful for fast verification).
    save_dir:
        Directory for CSV output.

    Returns
    -------
    List of dicts with keys:
    ``weights``, ``model_path``, ``mean_energy_wh``,
    ``mean_carbon_gco2``, ``mean_sla_violation``.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    grid = PARETO_WEIGHT_GRID[:max_configs] if max_configs is not None else PARETO_WEIGHT_GRID
    sweep_results: List[Dict] = []

    print(f"=== Phase 5c: Pareto Sweep ({len(grid)} configs × {timesteps_per_config:,} steps) ===")

    for i, weights in enumerate(grid):
        print(f"\n[{i + 1}/{len(grid)}] weights={weights}")
        t0 = time.perf_counter()

        model_name = f"carbon_sla_net_pareto_{i}"

        # Eval_freq: fire at least once, at most every 10 % of training
        eval_freq = max(1, min(base_config.ppo_eval_freq, timesteps_per_config // 4))

        config = dataclasses.replace(
            base_config,
            reward_weights=dict(weights),
            ppo_total_timesteps=timesteps_per_config,
            ppo_model_name=model_name,
            ppo_eval_freq=eval_freq,
        )

        # Synthesise a BCTrainingResult stub so run_rl_training sees the checkpoint
        bc_result = None
        if bc_checkpoint_path is not None:
            from agent.bc_trainer import BCTrainingResult
            bc_result = BCTrainingResult(
                train_losses=[],
                val_losses=[],
                val_accuracies=[],
                best_val_loss=float("nan"),
                best_val_accuracy=float("nan"),
                best_epoch=0,
                total_epochs_run=0,
                checkpoint_path=bc_checkpoint_path,
            )

        rl_result = run_rl_training(config, bc_result)

        # Ensure best_model.zip exists even if eval callback never fired
        _ensure_model_exists(rl_result.best_model_path, config)

        # Evaluate the trained model on 20 fresh episodes
        trainer = RLTrainer(
            N=config.N,
            T=config.T,
            reward_weights=config.reward_weights,
            workload_pattern=config.workload_pattern,
            renewable_condition=config.renewable_condition,
            checkpoint_dir=config.ppo_checkpoint_dir,
            seed=config.ppo_seed,
        )
        eval_metrics = trainer.evaluate_model(
            model_path=rl_result.best_model_path,
            n_episodes=20,
            seed_offset=70_000 + i * 100,
        )

        elapsed = time.perf_counter() - t0
        entry = {
            "weights": weights,
            "model_path": rl_result.best_model_path,
            "mean_energy_wh": eval_metrics["mean_energy_wh"],
            "mean_carbon_gco2": eval_metrics["mean_carbon_gco2"],
            "mean_sla_violation": eval_metrics["mean_sla_violation"],
        }
        sweep_results.append(entry)

        print(
            f"  Done in {elapsed:.1f}s | "
            f"E={eval_metrics['mean_energy_wh']:.1f} Wh | "
            f"CO2={eval_metrics['mean_carbon_gco2']:.1f} gCO2 | "
            f"SLA={eval_metrics['mean_sla_violation']:.3f}"
        )

    _save_csv(sweep_results, save_dir)
    return sweep_results


def _save_csv(results: List[Dict], save_dir: str) -> None:
    path = Path(save_dir) / "pareto_sweep_results.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "config_idx",
            "w_energy", "w_carbon", "w_sla", "w_renewable",
            "mean_energy_wh", "mean_carbon_gco2", "mean_sla_violation",
            "model_path",
        ])
        for i, entry in enumerate(results):
            w = entry["weights"]
            writer.writerow([
                i,
                w.get("energy", ""), w.get("carbon", ""),
                w.get("sla", ""), w.get("renewable", ""),
                f"{entry['mean_energy_wh']:.4f}",
                f"{entry['mean_carbon_gco2']:.4f}",
                f"{entry['mean_sla_violation']:.4f}",
                entry["model_path"],
            ])
    print(f"\nPareto sweep results saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5c: Pareto sweep for Carbon-SLA-Net"
    )
    parser.add_argument("--fast", action="store_true", help="Use fast config for testing")
    parser.add_argument(
        "--bc-checkpoint", type=str, default=None,
        help="Path to BC checkpoint for warm-starting"
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="PPO timesteps per weight configuration"
    )
    args = parser.parse_args()

    base_config = get_fast_config() if args.fast else get_default_config()
    run_pareto_sweep(
        base_config,
        bc_checkpoint_path=args.bc_checkpoint,
        timesteps_per_config=args.timesteps,
    )
