"""Workload-pattern × renewable-condition generalization evaluation."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from env.cloud_edge_env import CloudEdgeEnv
from evaluation.metrics import AggregatedMetrics, EpisodeMetrics, aggregate_metrics


class GeneralizationEvaluator:
    """Evaluates Carbon-SLA-Net on all 4 × 3 = 12 pattern/condition combos.

    The model is trained on ``('uniform', 'sunny')`` by default.  Running on
    all combinations reveals how well it generalises to unseen distributions.
    """

    PATTERNS: List[str] = ["uniform", "bursty", "heavy", "light"]
    CONDITIONS: List[str] = ["sunny", "cloudy", "no_re"]

    def __init__(
        self,
        N: int = 100,
        T: int = 8,
        n_episodes: int = 50,
        seed_offset: int = 40_000,
    ) -> None:
        self.N = N
        self.T = T
        self.n_episodes = n_episodes
        self.seed_offset = seed_offset

    def run(
        self,
        drl_model_path: str,
        save_dir: str = "results/tables",
    ) -> Dict[str, Dict[str, AggregatedMetrics]]:
        """Evaluate on all pattern × condition combinations.

        Parameters
        ----------
        drl_model_path:
            Path to a trained MaskablePPO ``.zip`` checkpoint.
        save_dir:
            Directory for CSV output.

        Returns
        -------
        Nested dict ``{pattern: {condition: AggregatedMetrics}}``.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(drl_model_path)

        outer: Dict[str, Dict[str, AggregatedMetrics]] = {}

        for pattern in self.PATTERNS:
            inner: Dict[str, AggregatedMetrics] = {}
            for condition in self.CONDITIONS:
                episodes: List[EpisodeMetrics] = []

                for ep in range(self.n_episodes):
                    seed = (
                        self.seed_offset
                        + self.PATTERNS.index(pattern) * 1000
                        + self.CONDITIONS.index(condition) * 100
                        + ep
                    )
                    env = CloudEdgeEnv(
                        N=self.N, T=self.T,
                        workload_pattern=pattern,
                        renewable_condition=condition,
                        seed=seed,
                    )
                    obs, _ = env.reset(seed=seed)
                    episode_reward = 0.0
                    terminal_info: dict = {}

                    t0 = time.perf_counter()
                    done = False
                    while not done:
                        masks = env.action_masks()
                        action, _ = model.predict(
                            obs, action_masks=masks, deterministic=True
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

                agg = aggregate_metrics(episodes)
                inner[condition] = agg
                print(
                    f"  [{pattern}/{condition}] "
                    f"E={agg.mean_energy_wh:.1f} Wh | "
                    f"SLA={agg.mean_sla_violation:.3f}"
                )

            outer[pattern] = inner

        self._save_csv(outer, save_dir)
        return outer

    def _save_csv(
        self,
        results: Dict[str, Dict[str, AggregatedMetrics]],
        save_dir: str,
    ) -> None:
        path = Path(save_dir) / "generalization_results.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "pattern", "condition",
                "mean_energy_wh", "mean_carbon_gco2",
                "mean_sla_violation", "mean_re_used_wh",
            ])
            for pattern, cond_dict in results.items():
                for condition, agg in cond_dict.items():
                    writer.writerow([
                        pattern, condition,
                        f"{agg.mean_energy_wh:.4f}",
                        f"{agg.mean_carbon_gco2:.4f}",
                        f"{agg.mean_sla_violation:.4f}",
                        f"{agg.mean_re_used_wh:.4f}",
                    ])
