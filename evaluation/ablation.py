"""Ablation study evaluator for Carbon-SLA-Net Phase 4."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from env.cloud_edge_env import CloudEdgeEnv
from evaluation.metrics import (
    AggregatedMetrics,
    EpisodeMetrics,
    aggregate_metrics,
    format_results_table,
)

# Observation layout: 11 nodes × 6 features each
# per-node feature offsets: 0=cpu_util, 1=mem_util, 2=bat_soc,
#                           3=re_avail, 4=carbon_ci, 5=tier_id
_N_NODES = 11


def _zero_carbon_features(obs: np.ndarray) -> np.ndarray:
    """Zero the carbon_ci feature (offset 4) for all 11 nodes."""
    obs = obs.copy()
    for j in range(_N_NODES):
        obs[j * 6 + 4] = 0.0
    return obs


def _zero_battery_features(obs: np.ndarray) -> np.ndarray:
    """Zero the bat_soc feature (offset 2) for all 11 nodes."""
    obs = obs.copy()
    for j in range(_N_NODES):
        obs[j * 6 + 2] = 0.0
    return obs


class AblationEvaluator:
    """Evaluates 5 ablation variants of Carbon-SLA-Net.

    Variants
    --------
    ``full``
        Full Carbon-SLA-Net with all components.
    ``no_carbon_signal``
        Carbon intensity zeroed from the observation before inference.
    ``no_battery_signal``
        Battery SoC zeroed from the observation before inference.
    ``no_bc_pretrain``
        PPO trained from scratch (random init checkpoint, no BC warm-start).
    ``no_masking``
        All action masks disabled — model sees all actions as valid.
    """

    VARIANTS: List[str] = [
        "full",
        "no_carbon_signal",
        "no_battery_signal",
        "no_bc_pretrain",
        "no_masking",
    ]

    def __init__(
        self,
        N: int = 100,
        T: int = 8,
        n_episodes: int = 50,
        seed_offset: int = 30_000,
    ) -> None:
        self.N = N
        self.T = T
        self.n_episodes = n_episodes
        self.seed_offset = seed_offset

    def run(
        self,
        model_paths: Dict[str, str],
        save_dir: str = "results/tables",
    ) -> Dict[str, AggregatedMetrics]:
        """Evaluate all variants whose checkpoint is provided in *model_paths*.

        Parameters
        ----------
        model_paths:
            Mapping ``{variant_name: checkpoint_path}``.  Variants absent
            from the dict are skipped.
        save_dir:
            Directory for CSV + table output.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        from sb3_contrib import MaskablePPO

        results: Dict[str, AggregatedMetrics] = {}

        for variant in self.VARIANTS:
            if variant not in model_paths:
                continue
            model = MaskablePPO.load(model_paths[variant])
            episodes: List[EpisodeMetrics] = []

            for ep in range(self.n_episodes):
                seed = self.seed_offset + ep
                env = CloudEdgeEnv(N=self.N, T=self.T, seed=seed)
                obs, _ = env.reset(seed=seed)
                episode_reward = 0.0
                terminal_info: dict = {}

                t0 = time.perf_counter()
                done = False
                while not done:
                    # Feature masking for ablation variants
                    if variant == "no_carbon_signal":
                        obs_in = _zero_carbon_features(obs)
                    elif variant == "no_battery_signal":
                        obs_in = _zero_battery_features(obs)
                    else:
                        obs_in = obs

                    # Action masking ablation
                    if variant == "no_masking":
                        masks = np.ones(len(env.nodes), dtype=bool)
                    else:
                        masks = env.action_masks()

                    action, _ = model.predict(
                        obs_in, action_masks=masks, deterministic=True
                    )
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    episode_reward += reward
                    terminal_info = info
                    done = terminated or truncated
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                episodes.append(EpisodeMetrics(
                    energy_wh=float(terminal_info.get("energy_wh", 0)),
                    carbon_gco2=float(terminal_info.get("carbon_gco2", 0)),
                    sla_violation=float(terminal_info.get("sla_violation", 0)),
                    re_used_wh=float(terminal_info.get("re_used_wh", 0)),
                    inference_ms=elapsed_ms,
                    n_tasks_assigned=int(terminal_info.get("task_idx", 0)),
                    reward=episode_reward,
                ))

            results[variant] = aggregate_metrics(episodes)
            agg = results[variant]
            print(
                f"  Ablation [{variant}] | "
                f"mean_E={agg.mean_energy_wh:.1f} Wh | "
                f"SLA={agg.mean_sla_violation:.3f}"
            )

        # Save
        self._save_csv(results, save_dir)
        table_str = format_results_table(results)
        (Path(save_dir) / "ablation_results_table.txt").write_text(table_str)

        return results

    def _save_csv(
        self, results: Dict[str, AggregatedMetrics], save_dir: str
    ) -> None:
        path = Path(save_dir) / "ablation_results.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "variant",
                "mean_energy_wh", "ci95_energy_wh",
                "mean_carbon_gco2", "ci95_carbon_gco2",
                "mean_sla_violation", "ci95_sla_violation",
                "mean_re_used_wh", "mean_inference_ms", "mean_reward",
            ])
            for variant, agg in results.items():
                writer.writerow([
                    variant,
                    f"{agg.mean_energy_wh:.4f}", f"{agg.ci95_energy_wh:.4f}",
                    f"{agg.mean_carbon_gco2:.4f}", f"{agg.ci95_carbon_gco2:.4f}",
                    f"{agg.mean_sla_violation:.4f}", f"{agg.ci95_sla_violation:.4f}",
                    f"{agg.mean_re_used_wh:.4f}",
                    f"{agg.mean_inference_ms:.4f}",
                    f"{agg.mean_reward:.6f}",
                ])
