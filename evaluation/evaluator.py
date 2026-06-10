"""Main evaluation runner for Carbon-SLA-Net Phase 4.

Runs all methods on held-out episodes, collects metrics via env replay, and
saves CSV + formatted table to results/tables/.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from env.cloud_edge_env import CloudEdgeEnv
from milp.scheduler_milp import solve_milp
from milp.nsga2_scheduler import NSGA2Scheduler
from baselines.renewable_greedy import RenewableGreedyScheduler
from baselines.energy_greedy import EnergyGreedyScheduler
from baselines.sla_priority import SLAPriorityScheduler
from evaluation.metrics import (
    AggregatedMetrics,
    EpisodeMetrics,
    aggregate_metrics,
    format_results_table,
    milp_gap,
)

_DEFAULT_WEIGHTS = {"energy": 0.3, "carbon": 0.3, "sla": 0.3, "renewable": 0.1}


class Evaluator:
    """Runs all scheduling methods on held-out episodes and aggregates metrics.

    All methods are evaluated on the **same** episode seeds so comparisons are
    fair.  Metrics are collected by replaying assignments through
    :class:`~env.cloud_edge_env.CloudEdgeEnv` — the environment is the single
    source of truth for energy, carbon, SLA, and RE accounting.
    """

    def __init__(
        self,
        N: int = 100,
        T: int = 8,
        n_episodes: int = 200,
        seed_offset: int = 10_000,
        renewable_condition: str = "sunny",
        workload_pattern: str = "uniform",
    ) -> None:
        self.N = N
        self.T = T
        self.n_episodes = n_episodes
        self.seed_offset = seed_offset
        self.renewable_condition = renewable_condition
        self.workload_pattern = workload_pattern

    # ------------------------------------------------------------------
    # Core episode runners
    # ------------------------------------------------------------------

    def _make_env(self, seed: int) -> CloudEdgeEnv:
        return CloudEdgeEnv(
            N=self.N,
            T=self.T,
            renewable_condition=self.renewable_condition,
            workload_pattern=self.workload_pattern,
            seed=seed,
        )

    def _replay_assignments(
        self,
        assignments: Dict[int, int],
        env: CloudEdgeEnv,
        seed: int,
    ) -> EpisodeMetrics:
        """Reset env and step through all tasks using precomputed assignments.

        Timing covers the replay loop only (not the scheduling computation),
        giving a fair wall-clock measurement for heuristic methods.
        """
        obs, _ = env.reset(seed=seed)
        J = len(env.nodes)
        episode_reward = 0.0
        terminal_info: Dict = {}

        t0 = time.perf_counter()
        for _ in range(len(env.tasks)):
            task = env.tasks[env.task_idx]
            action = assignments.get(task.task_id)
            if action is None:
                action = 0
            action = int(action) % J
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            terminal_info = info
            if terminated or truncated:
                break
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return EpisodeMetrics(
            energy_wh=float(terminal_info.get("energy_wh", 0.0)),
            carbon_gco2=float(terminal_info.get("carbon_gco2", 0.0)),
            sla_violation=float(terminal_info.get("sla_violation", 0.0)),
            re_used_wh=float(terminal_info.get("re_used_wh", 0.0)),
            inference_ms=elapsed_ms,
            n_tasks_assigned=int(terminal_info.get("task_idx", 0)),
            reward=episode_reward,
        )

    def _run_drl_episode(self, model, env: CloudEdgeEnv, seed: int) -> EpisodeMetrics:
        """Step through one episode with a MaskablePPO model."""
        obs, _ = env.reset(seed=seed)
        episode_reward = 0.0
        terminal_info: Dict = {}

        t0 = time.perf_counter()
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            terminal_info = info
            done = terminated or truncated
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return EpisodeMetrics(
            energy_wh=float(terminal_info.get("energy_wh", 0.0)),
            carbon_gco2=float(terminal_info.get("carbon_gco2", 0.0)),
            sla_violation=float(terminal_info.get("sla_violation", 0.0)),
            re_used_wh=float(terminal_info.get("re_used_wh", 0.0)),
            inference_ms=elapsed_ms,
            n_tasks_assigned=int(terminal_info.get("task_idx", 0)),
            reward=episode_reward,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        drl_model_path: Optional[str] = None,
        milp_time_limit: float = 60.0,
        save_dir: str = "results/tables",
        fast_mode: bool = False,
    ) -> Dict[str, AggregatedMetrics]:
        """Run every method on the same held-out episodes.

        Parameters
        ----------
        drl_model_path:
            Path to a MaskablePPO ``.zip`` checkpoint.  If None,
            Carbon-SLA-Net is skipped.
        milp_time_limit:
            Per-episode Gurobi time limit in seconds.
        save_dir:
            Directory for CSV and text output files.
        fast_mode:
            If True, uses reduced NSGA-II settings
            (population=20, generations=10) for faster iteration.

        Returns
        -------
        Dict mapping method name to :class:`AggregatedMetrics`.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        nsga_pop = 20 if fast_mode else 100
        nsga_gen = 10 if fast_mode else 50

        drl_model = None
        if drl_model_path is not None:
            from sb3_contrib import MaskablePPO
            drl_model = MaskablePPO.load(drl_model_path)

        methods_to_run = [
            "MILP", "NSGA-II",
            "RenewableGreedy", "EnergyGreedy", "SLAPriority",
        ]
        if drl_model is not None:
            methods_to_run.append("Carbon-SLA-Net")

        results: Dict[str, AggregatedMetrics] = {}

        for idx, method in enumerate(methods_to_run, start=1):
            episode_metrics: List[EpisodeMetrics] = []
            t_method = time.perf_counter()

            for ep in range(self.n_episodes):
                seed = self.seed_offset + ep
                env = self._make_env(seed)
                env.reset(seed=seed)

                if method == "MILP":
                    milp_result = solve_milp(
                        env.tasks, env.nodes,
                        env.carbon_episode, env.re_episode,
                        T=self.T, time_limit=milp_time_limit, silent=True,
                    )
                    em = self._replay_assignments(milp_result.assignments, env, seed)

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
                    em = self._replay_assignments(
                        out["best_energy"]["assignments"], env, seed
                    )

                elif method == "RenewableGreedy":
                    assignments = RenewableGreedyScheduler().schedule(env)
                    em = self._replay_assignments(assignments, env, seed)

                elif method == "EnergyGreedy":
                    assignments = EnergyGreedyScheduler().schedule(env)
                    em = self._replay_assignments(assignments, env, seed)

                elif method == "SLAPriority":
                    assignments = SLAPriorityScheduler().schedule(env)
                    em = self._replay_assignments(assignments, env, seed)

                elif method == "Carbon-SLA-Net":
                    env2 = self._make_env(seed)
                    em = self._run_drl_episode(drl_model, env2, seed)

                else:
                    continue

                episode_metrics.append(em)

            agg = aggregate_metrics(episode_metrics)
            results[method] = agg
            elapsed = time.perf_counter() - t_method
            print(
                f"[{idx}/{len(methods_to_run)}] {method:<15} | "
                f"{self.n_episodes} episodes | "
                f"mean_E={agg.mean_energy_wh:.1f} Wh | "
                f"t={elapsed/self.n_episodes:.1f}s/ep | "
                f"total={elapsed/60:.1f}min"
            )

        # Persist results
        self._save_csv(results, save_dir)
        table_str = format_results_table(results)
        (Path(save_dir) / "main_results_table.txt").write_text(table_str)
        print(f"\n{table_str}")

        return results

    def run_single(
        self,
        method_name: str,
        scheduler_or_model,
        seed_offset: Optional[int] = None,
    ) -> AggregatedMetrics:
        """Run one scheduling method for ``self.n_episodes`` episodes.

        Parameters
        ----------
        method_name:
            Human-readable name (used only for printing).
        scheduler_or_model:
            One of:
            * A greedy scheduler with a ``schedule(env)`` method.
            * A loaded ``MaskablePPO`` model with a ``predict()`` method.
            * The string ``'milp'``.
        seed_offset:
            Override for the starting seed; defaults to ``self.seed_offset``.
        """
        if seed_offset is None:
            seed_offset = self.seed_offset

        episode_metrics: List[EpisodeMetrics] = []

        for ep in range(self.n_episodes):
            seed = seed_offset + ep
            env = self._make_env(seed)
            env.reset(seed=seed)

            if scheduler_or_model == "milp":
                result = solve_milp(
                    env.tasks, env.nodes,
                    env.carbon_episode, env.re_episode,
                    T=self.T, silent=True,
                )
                em = self._replay_assignments(result.assignments, env, seed)

            elif hasattr(scheduler_or_model, "schedule"):
                assignments = scheduler_or_model.schedule(env)
                em = self._replay_assignments(assignments, env, seed)

            elif hasattr(scheduler_or_model, "predict"):
                env2 = self._make_env(seed)
                em = self._run_drl_episode(scheduler_or_model, env2, seed)

            else:
                raise ValueError(
                    f"scheduler_or_model must be a scheduler, MaskablePPO, or 'milp'; "
                    f"got {type(scheduler_or_model)}"
                )

            episode_metrics.append(em)

        return aggregate_metrics(episode_metrics)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_csv(
        self, results: Dict[str, AggregatedMetrics], save_dir: str
    ) -> None:
        milp_comp = (
            results["MILP"].composite_energy if "MILP" in results else None
        )
        path = Path(save_dir) / "main_results.csv"
        fieldnames = list(asdict(next(iter(results.values()))).keys()) + [
            "method", "milp_gap_pct"
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for method, agg in results.items():
                row = asdict(agg)
                row["method"] = method
                if milp_comp is not None and milp_comp != 0.0:
                    gap = milp_gap(agg.composite_energy, milp_comp)
                else:
                    gap = float("nan")
                row["milp_gap_pct"] = gap
                writer.writerow(row)
