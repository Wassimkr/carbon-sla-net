"""Tests for Phase 2: MILP oracle, BC dataset generator, and NSGA-II baseline."""

from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np
import pytest

from env.cloud_edge_env import CloudEdgeEnv
from milp.scheduler_milp import MILPResult, solve_milp
from milp.oracle_runner import bc_dataset_stats, generate_bc_dataset
from milp.nsga2_scheduler import NSGA2Scheduler


# ---------------------------------------------------------------------------
# Gurobi availability check
# ---------------------------------------------------------------------------

def _gurobi_ok() -> bool:
    try:
        import gurobipy as gp
        m = gp.Model()
        m.Params.OutputFlag = 0
        x = m.addVar()
        m.setObjective(x)
        m.optimize()
        return True
    except Exception:
        return False


GUROBI_AVAILABLE = _gurobi_ok()
_skip_no_gurobi = pytest.mark.skipif(
    not GUROBI_AVAILABLE, reason="Gurobi not available or no valid license"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_env_solved():
    """N=10, T=8 env with a solved MILP result."""
    env = CloudEdgeEnv(N=10, T=8, seed=42)
    env.reset()
    result = solve_milp(
        env.tasks, env.nodes, env.carbon_episode, env.re_episode, T=8,
        time_limit=120.0, silent=True,
    )
    return env, result


# ===========================================================================
# MILP oracle tests (1 – 10)
# ===========================================================================

# 1. solve_milp returns a MILPResult
def test_milp_returns_result_type(small_env_solved):
    _, result = small_env_solved
    assert isinstance(result, MILPResult)


# 2. n_assigned == N for small feasible problem  [needs Gurobi]
@_skip_no_gurobi
def test_milp_assigns_all_tasks(small_env_solved):
    env, result = small_env_solved
    assert result.n_assigned == len(env.tasks), (
        f"Expected {len(env.tasks)} assignments, got {result.n_assigned}"
    )


# 3. assignments is a dict with integer keys and node-index values
def test_milp_assignments_dict_types(small_env_solved):
    env, result = small_env_solved
    J = len(env.nodes)
    for task_id, node_id in result.assignments.items():
        assert isinstance(task_id, int)
        assert isinstance(node_id, int)
        assert 0 <= node_id < J


# 4. energy_wh > 0 for any non-trivial assignment  [needs Gurobi]
@_skip_no_gurobi
def test_milp_energy_positive(small_env_solved):
    _, result = small_env_solved
    assert result.energy_wh > 0.0


# 5. timed_out == False for small problem  [needs Gurobi]
@_skip_no_gurobi
def test_milp_not_timed_out(small_env_solved):
    _, result = small_env_solved
    assert not result.timed_out


# 6. mip_gap < 0.01 for N=10  [needs Gurobi]
@_skip_no_gurobi
def test_milp_gap_optimal(small_env_solved):
    _, result = small_env_solved
    assert result.mip_gap < 0.01, f"MIP gap {result.mip_gap:.4f} too large"


# 7. All assigned node_ids are valid indices
def test_milp_valid_node_ids(small_env_solved):
    env, result = small_env_solved
    J = len(env.nodes)
    for node_id in result.assignments.values():
        assert 0 <= node_id < J


# 8. Each task appears at most once in assignments
def test_milp_unique_task_assignments(small_env_solved):
    _, result = small_env_solved
    task_ids = list(result.assignments.keys())
    assert len(task_ids) == len(set(task_ids))


# 9. CPU capacity is respected for every (node, slot) pair
def test_milp_cpu_capacity_respected(small_env_solved):
    env, result = small_env_solved
    if result.n_assigned == 0:
        pytest.skip("No MILP solution to verify")

    cpu_load: dict = defaultdict(float)
    for task in env.tasks:
        if task.task_id in result.assignments:
            j = result.assignments[task.task_id]
            cpu_load[(j, task.t_start)] += task.cpu_req

    for (j, t), total in cpu_load.items():
        cap = env.nodes[j].cpu_capacity
        assert total <= cap + 1e-6, (
            f"Node {j} slot {t}: CPU load {total:.3f} > capacity {cap:.3f}"
        )


# 10. solve_milp completes without exception for N=50
def test_milp_no_exception_n50():
    env = CloudEdgeEnv(N=50, T=8, seed=7)
    env.reset()
    result = solve_milp(
        env.tasks, env.nodes, env.carbon_episode, env.re_episode,
        T=8, time_limit=120.0, silent=True,
    )
    assert isinstance(result, MILPResult)


# ===========================================================================
# BC dataset tests (11 – 18)
# ===========================================================================

_BC_PATH = "data/bc_dataset/test_5.pkl"


@pytest.fixture(scope="module")
def bc_dataset():
    return generate_bc_dataset(
        n_episodes=5, N=10, T=8,
        output_path=_BC_PATH,
        time_limit=120.0,
        verbose=False,
    )


# 11. generate_bc_dataset returns a list
def test_bc_returns_list(bc_dataset):
    assert isinstance(bc_dataset, list)


# 12. Each element is a (ndarray, int) tuple
def test_bc_element_types(bc_dataset):
    for obs, action in bc_dataset:
        assert isinstance(obs, np.ndarray)
        assert isinstance(action, int)


# 13. obs shape == (74,)
def test_bc_obs_shape(bc_dataset):
    for obs, _ in bc_dataset:
        assert obs.shape == (74,), f"Expected (74,), got {obs.shape}"


# 14. action in [0, 10]
def test_bc_action_range(bc_dataset):
    for _, action in bc_dataset:
        assert 0 <= action <= 10, f"action={action} out of [0, 10]"


# 15. Saved pickle file exists
def test_bc_pickle_file_exists(bc_dataset):
    assert os.path.exists(_BC_PATH), f"Pickle not found at {_BC_PATH}"


# 16. Loaded dataset matches returned dataset
def test_bc_pickle_matches(bc_dataset):
    with open(_BC_PATH, "rb") as fh:
        loaded = pickle.load(fh)
    assert len(loaded) == len(bc_dataset)
    for (o1, a1), (o2, a2) in zip(loaded, bc_dataset):
        np.testing.assert_array_equal(o1, o2)
        assert a1 == a2


# 17. bc_dataset_stats returns dict with correct keys
def test_bc_stats_keys(bc_dataset):
    stats = bc_dataset_stats(bc_dataset)
    required = {"n_pairs", "action_distribution", "obs_min", "obs_max", "obs_mean"}
    assert required.issubset(stats.keys())


# 18. action_distribution values sum to n_pairs
def test_bc_stats_distribution_sum(bc_dataset):
    stats = bc_dataset_stats(bc_dataset)
    dist_sum = sum(stats["action_distribution"].values())
    assert dist_sum == stats["n_pairs"]


# ===========================================================================
# NSGA-II tests (19 – 24)
# ===========================================================================

@pytest.fixture(scope="module")
def nsga2_result():
    env = CloudEdgeEnv(N=20, T=8, seed=42)
    env.reset()
    sched = NSGA2Scheduler(population_size=20, n_generations=10, seed=42)
    return env, sched.schedule(
        env.tasks, env.nodes, env.carbon_episode, env.re_episode, T=8
    )


# 19. schedule() returns a dict with 'pareto_front'
def test_nsga2_returns_dict(nsga2_result):
    _, result = nsga2_result
    assert isinstance(result, dict)
    assert "pareto_front" in result


# 20. pareto_front is a non-empty list
def test_nsga2_pareto_nonempty(nsga2_result):
    _, result = nsga2_result
    assert isinstance(result["pareto_front"], list)
    assert len(result["pareto_front"]) > 0


# 21. Each Pareto solution has required keys
def test_nsga2_pareto_solution_keys(nsga2_result):
    _, result = nsga2_result
    required = {"assignments", "energy_wh", "sla_violation", "carbon_gco2"}
    for sol in result["pareto_front"]:
        assert required.issubset(sol.keys())


# 22. best_energy['energy_wh'] > 0
def test_nsga2_best_energy_positive(nsga2_result):
    _, result = nsga2_result
    assert result["best_energy"]["energy_wh"] > 0.0


# 23. runtime_s > 0
def test_nsga2_runtime_positive(nsga2_result):
    _, result = nsga2_result
    assert result["runtime_s"] > 0.0


# 24. NSGA-II completes without exception for N=20, T=8
def test_nsga2_no_exception():
    env = CloudEdgeEnv(N=20, T=8, seed=99)
    env.reset()
    sched = NSGA2Scheduler(population_size=10, n_generations=5, seed=99)
    result = sched.schedule(
        env.tasks, env.nodes, env.carbon_episode, env.re_episode, T=8
    )
    assert isinstance(result, dict)
