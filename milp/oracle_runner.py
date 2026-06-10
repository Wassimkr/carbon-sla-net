"""Behavior cloning dataset generator for Carbon-SLA-Net.

Runs the MILP oracle on many episodes, replays the optimal assignments
through the Gymnasium environment to collect (observation, action) pairs,
then serialises the dataset to disk for offline BC training.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from env.cloud_edge_env import CloudEdgeEnv
from milp.scheduler_milp import MILPResult, solve_milp


def generate_bc_dataset(
    n_episodes: int = 500,
    N: int = 100,
    T: int = 8,
    output_path: str = "data/bc_dataset/bc_500.pkl",
    time_limit: float = 60.0,
    seed_offset: int = 0,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """Generate a behavior-cloning dataset using the MILP oracle.

    For each episode a fresh environment is created, the MILP is solved on
    its tasks and environmental signals, and the optimal assignment is
    replayed step-by-step to collect ``(obs, action)`` pairs.  Episodes
    where the MILP fails or produces an action that is infeasible in the
    environment are skipped entirely.

    Parameters
    ----------
    n_episodes:
        Total number of episodes to attempt.
    N:
        Number of tasks per episode.
    T:
        Number of time slots per episode.
    output_path:
        Where to write the pickled dataset.
    time_limit:
        Per-episode Gurobi time limit (seconds).
    seed_offset:
        Added to the episode index to form the environment seed.
    verbose:
        Print progress every 50 episodes and a final summary.

    Returns
    -------
    Flat list of ``(obs, action)`` tuples across all valid episodes.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_pairs: List[Tuple[np.ndarray, int]] = []
    skip_count = 0
    t0 = time.perf_counter()

    for episode_idx in range(n_episodes):
        seed = seed_offset + episode_idx
        env = CloudEdgeEnv(N=N, T=T, seed=seed)
        obs, _ = env.reset(seed=seed)

        # Solve MILP on this episode's tasks and environment signals
        result: MILPResult = solve_milp(
            tasks=env.tasks,
            nodes=env.nodes,
            carbon_per_node=env.carbon_episode,
            re_per_node=env.re_episode,
            T=T,
            time_limit=time_limit,
            silent=True,
        )

        # Skip if MILP did not produce a complete assignment
        if result.n_assigned == 0 or result.timed_out:
            skip_count += 1
            if verbose:
                warnings.warn(
                    f"Episode {episode_idx}: MILP failed "
                    f"(n_assigned={result.n_assigned}, timed_out={result.timed_out}) — skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        # Replay MILP assignments through the environment
        episode_pairs: List[Tuple[np.ndarray, int]] = []
        valid = True
        current_obs = obs  # observation before the first step

        for _ in range(len(env.tasks)):
            task = env.tasks[env.task_idx]
            action = result.assignments.get(task.task_id)

            if action is None:
                # MILP did not assign this task — episode is incomplete
                valid = False
                break

            masks = env.action_masks()
            if not masks[action]:
                # MILP chose a node that is infeasible from the env's view
                warnings.warn(
                    f"Episode {episode_idx}: MILP action {action} is infeasible "
                    f"for task {task.task_id} — skipping episode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                valid = False
                break

            # Record the state–action pair BEFORE stepping
            episode_pairs.append((current_obs.copy(), int(action)))
            current_obs, _, terminated, _, _ = env.step(action)

            if terminated:
                break

        if valid and len(episode_pairs) == len(env.tasks):
            all_pairs.extend(episode_pairs)
        else:
            if not valid:
                skip_count += 1

        # Progress logging
        if verbose and (episode_idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"Episode {episode_idx + 1}/{n_episodes} | "
                f"collected {len(all_pairs)} pairs | "
                f"skipped {skip_count} episodes | "
                f"elapsed {elapsed:.0f}s"
            )

    # Persist dataset
    with open(output_path, "wb") as fh:
        pickle.dump(all_pairs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    n_valid = n_episodes - skip_count
    if verbose:
        print(
            f"BC dataset complete: {len(all_pairs)} pairs from {n_valid} episodes "
            f"({skip_count} skipped) → {output_path}"
        )

    return all_pairs


def bc_dataset_stats(dataset: List[Tuple[np.ndarray, int]]) -> Dict:
    """Compute summary statistics for a BC dataset.

    Parameters
    ----------
    dataset:
        List of ``(obs, action)`` tuples as returned by
        :func:`generate_bc_dataset`.

    Returns
    -------
    Dict with keys:
      * ``n_pairs`` — total number of (obs, action) pairs
      * ``action_distribution`` — ``{node_id: count}``
      * ``obs_min``, ``obs_max``, ``obs_mean`` — scalar statistics
    """
    n_pairs = len(dataset)
    action_dist: Dict[int, int] = {}
    obs_list: List[np.ndarray] = []

    for obs, action in dataset:
        action_dist[action] = action_dist.get(action, 0) + 1
        obs_list.append(obs)

    if obs_list:
        arr = np.stack(obs_list)
        obs_min = float(arr.min())
        obs_max = float(arr.max())
        obs_mean = float(arr.mean())
    else:
        obs_min = obs_max = obs_mean = 0.0

    return {
        "n_pairs": n_pairs,
        "action_distribution": action_dist,
        "obs_min": obs_min,
        "obs_max": obs_max,
        "obs_mean": obs_mean,
    }
