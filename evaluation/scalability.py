"""Scalability sweep (N=50→500) for Carbon-SLA-Net Phase 4."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

from env.cloud_edge_env import CloudEdgeEnv
from milp.scheduler_milp import solve_milp
from milp.nsga2_scheduler import NSGA2Scheduler
from evaluation.metrics import AggregatedMetrics, EpisodeMetrics, aggregate_metrics

_MILP_N_CAP = 150   # MILP skipped for N > this threshold (too slow)


class ScalabilityEvaluator:
    """Runs MILP, NSGA-II, and optionally DRL across multiple task counts.

    For N > ``_MILP_N_CAP`` MILP is evaluated but with a tighter time limit
    (the solver returns a partial/timed-out solution — the result reflects
    whatever assignments it found within the limit).
    """

    def __init__(
        self,
        N_values: Optional[List[int]] = None,
        T: int = 8,
        n_trials: int = 5,
        milp_time_limit: float = 60.0,
        seed_offset: int = 50_000,
    ) -> None:
        self.N_values = N_values if N_values is not None else [
            50, 100, 150, 200, 250, 300, 400, 500
        ]
        self.T = T
        self.n_trials = n_trials
        self.milp_time_limit = milp_time_limit
        self.seed_offset = seed_offset

    def run(
        self,
        drl_model_path: Optional[str] = None,
        save_dir: str = "results/tables",
        fast_mode: bool = False,
    ) -> Dict[int, Dict[str, AggregatedMetrics]]:
        """Run the N-sweep.

        Returns
        -------
        Nested dict ``{N: {method: AggregatedMetrics}}``.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        nsga_pop = 20 if fast_mode else 100
        nsga_gen = 10 if fast_mode else 50

        drl_model = None
        if drl_model_path is not None:
            from sb3_contrib import MaskablePPO
            drl_model = MaskablePPO.load(drl_model_path)

        outer: Dict[int, Dict[str, AggregatedMetrics]] = {}

        for N in self.N_values:
            inner: Dict[str, AggregatedMetrics] = {}
            methods = ["MILP", "NSGA-II"]
            if drl_model is not None:
                methods.append("Carbon-SLA-Net")

            for method in methods:
                episodes: List[EpisodeMetrics] = []

                for trial in range(self.n_trials):
                    seed = self.seed_offset + N * 1000 + trial
                    env = CloudEdgeEnv(N=N, T=self.T, seed=seed)
                    env.reset(seed=seed)

                    t0 = time.perf_counter()

                    if method == "MILP":
                        # Use shorter limit for large N
                        tl = self.milp_time_limit if N <= _MILP_N_CAP else 30.0
                        result = solve_milp(
                            env.tasks, env.nodes,
                            env.carbon_episode, env.re_episode,
                            T=self.T, time_limit=tl, silent=True,
                        )
                        assignments = result.assignments
                        # Replay
                        env2 = CloudEdgeEnv(N=N, T=self.T, seed=seed)
                        em = self._replay(assignments, env2, seed)

                    elif method == "NSGA-II":
                        nsga2 = NSGA2Scheduler(
                            population_size=nsga_pop,
                            n_generations=nsga_gen,
                            seed=seed,
                        )
                        out = nsga2.schedule(
                            env.tasks, env.nodes,
                            env.carbon_episode, env.re_episode, T=self.T,
                        )
                        env2 = CloudEdgeEnv(N=N, T=self.T, seed=seed)
                        em = self._replay(out["best_energy"]["assignments"], env2, seed)

                    elif method == "Carbon-SLA-Net":
                        env2 = CloudEdgeEnv(N=N, T=self.T, seed=seed)
                        env2.reset(seed=seed)
                        obs = env2.reset(seed=seed)[0]
                        ep_reward = 0.0
                        terminal_info: dict = {}
                        done = False
                        while not done:
                            masks = env2.action_masks()
                            action, _ = drl_model.predict(
                                obs, action_masks=masks, deterministic=True
                            )
                            obs, r, ter, trun, info = env2.step(int(action))
                            ep_reward += r
                            terminal_info = info
                            done = ter or trun
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        em = EpisodeMetrics(
                            energy_wh=float(terminal_info.get("energy_wh", 0)),
                            carbon_gco2=float(terminal_info.get("carbon_gco2", 0)),
                            sla_violation=float(terminal_info.get("sla_violation", 0)),
                            re_used_wh=float(terminal_info.get("re_used_wh", 0)),
                            inference_ms=elapsed_ms,
                            n_tasks_assigned=int(terminal_info.get("task_idx", 0)),
                            reward=ep_reward,
                        )
                    else:
                        continue

                    episodes.append(em)

                inner[method] = aggregate_metrics(episodes)

            outer[N] = inner
            print(
                f"  N={N:4d} | MILP={inner['MILP'].mean_energy_wh:.1f} Wh | "
                f"NSGA-II={inner['NSGA-II'].mean_energy_wh:.1f} Wh"
            )

        self._save_csv(outer, save_dir)
        return outer

    # ------------------------------------------------------------------

    def _replay(
        self,
        assignments: Dict,
        env: CloudEdgeEnv,
        seed: int,
    ) -> EpisodeMetrics:
        env.reset(seed=seed)
        J = len(env.nodes)
        episode_reward = 0.0
        terminal_info: dict = {}

        t0 = time.perf_counter()
        for _ in range(len(env.tasks)):
            task = env.tasks[env.task_idx]
            action = assignments.get(task.task_id, 0)
            action = int(action) % J
            _, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            terminal_info = info
            if terminated or truncated:
                break
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return EpisodeMetrics(
            energy_wh=float(terminal_info.get("energy_wh", 0)),
            carbon_gco2=float(terminal_info.get("carbon_gco2", 0)),
            sla_violation=float(terminal_info.get("sla_violation", 0)),
            re_used_wh=float(terminal_info.get("re_used_wh", 0)),
            inference_ms=elapsed_ms,
            n_tasks_assigned=int(terminal_info.get("task_idx", 0)),
            reward=episode_reward,
        )

    def _save_csv(
        self,
        results: Dict[int, Dict[str, AggregatedMetrics]],
        save_dir: str,
    ) -> None:
        path = Path(save_dir) / "scalability_results.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "N", "method",
                "mean_energy_wh", "ci95_energy_wh",
                "mean_sla_violation", "mean_carbon_gco2",
                "mean_inference_ms",
            ])
            for N, method_dict in results.items():
                milp_comp = (
                    method_dict["MILP"].composite_energy
                    if "MILP" in method_dict else None
                )
                for method, agg in method_dict.items():
                    writer.writerow([
                        N, method,
                        f"{agg.mean_energy_wh:.4f}",
                        f"{agg.ci95_energy_wh:.4f}",
                        f"{agg.mean_sla_violation:.4f}",
                        f"{agg.mean_carbon_gco2:.4f}",
                        f"{agg.mean_inference_ms:.4f}",
                    ])

        # MILP gap vs N
        gap_path = Path(save_dir) / "milp_gap_vs_N.csv"
        with open(gap_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["N", "method", "milp_gap_pct"])
            for N, method_dict in results.items():
                if "MILP" not in method_dict:
                    continue
                milp_comp = method_dict["MILP"].composite_energy
                for method, agg in method_dict.items():
                    if method == "MILP":
                        gap = 0.0
                    elif milp_comp and milp_comp != 0.0:
                        gap = (agg.composite_energy - milp_comp) / milp_comp * 100.0
                    else:
                        gap = float("nan")
                    writer.writerow([N, method, f"{gap:.4f}"])
