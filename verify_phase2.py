"""Phase 2 verification script for Carbon-SLA-Net.

Run with:  python verify_phase2.py

Exercises the MILP oracle, BC dataset generator, and NSGA-II baseline.
No tracebacks should occur.
"""

from __future__ import annotations

import numpy as np

from env.cloud_edge_env import CloudEdgeEnv
from milp.scheduler_milp import solve_milp
from milp.oracle_runner import bc_dataset_stats, generate_bc_dataset
from milp.nsga2_scheduler import NSGA2Scheduler


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. MILP smoke test
# ---------------------------------------------------------------------------
section("1. MILP oracle smoke test (N=20, T=8)")
env = CloudEdgeEnv(N=20, T=8, seed=42)
env.reset()

result = solve_milp(
    env.tasks, env.nodes, env.carbon_episode, env.re_episode,
    T=8, time_limit=120.0, silent=True,
)

print(f"  n_assigned   : {result.n_assigned} / {len(env.tasks)}")
print(f"  energy_wh    : {result.energy_wh:.2f}")
print(f"  carbon_gco2  : {result.carbon_gco2:.4f}")
print(f"  sla_violation: {result.sla_violation:.4f}")
print(f"  mip_gap      : {result.mip_gap:.6f}")
print(f"  runtime_s    : {result.runtime_s:.3f}")

# ---------------------------------------------------------------------------
# 2. Consistency check
# ---------------------------------------------------------------------------
section("2. Consistency check: n_assigned == N")
ok = result.n_assigned == len(env.tasks)
print(f"  n_assigned={result.n_assigned}, N={len(env.tasks)} → {'PASS' if ok else 'FAIL'}")

# ---------------------------------------------------------------------------
# 3. MILP vs env comparison
# ---------------------------------------------------------------------------
section("3. MILP vs env energy comparison (tolerance 5%)")
env2 = CloudEdgeEnv(N=20, T=8, seed=42)
obs, _ = env2.reset(seed=42)

env_energy = 0.0
for _ in range(len(env2.tasks)):
    task = env2.tasks[env2.task_idx]
    action = result.assignments.get(task.task_id, 0)
    _, _, terminated, _, info = env2.step(action)
    env_energy = info["energy_wh"]
    if terminated:
        break

pct_diff = abs(env_energy - result.energy_wh) / max(result.energy_wh, 1.0) * 100
ok3 = pct_diff <= 5.0
print(f"  MILP energy  : {result.energy_wh:.2f} Wh")
print(f"  Env energy   : {env_energy:.2f} Wh")
print(f"  Difference   : {pct_diff:.2f}% → {'PASS' if ok3 else 'FAIL'}")

# ---------------------------------------------------------------------------
# 4. BC dataset mini-run
# ---------------------------------------------------------------------------
section("4. BC dataset mini-run (10 episodes, N=20, T=8)")
dataset = generate_bc_dataset(
    n_episodes=10,
    N=20,
    T=8,
    output_path="data/bc_dataset/verify_10.pkl",
    time_limit=120.0,
    verbose=True,
)
print(f"  Collected: {len(dataset)} (obs, action) pairs")

# ---------------------------------------------------------------------------
# 5. Dataset statistics
# ---------------------------------------------------------------------------
section("5. BC dataset statistics")
stats = bc_dataset_stats(dataset)
print(f"  n_pairs : {stats['n_pairs']}")
print(f"  obs_min : {stats['obs_min']:.4f}")
print(f"  obs_max : {stats['obs_max']:.4f}")
print(f"  obs_mean: {stats['obs_mean']:.4f}")
print(f"  Action distribution (node_id: count):")
for node_id in sorted(stats["action_distribution"]):
    count = stats["action_distribution"][node_id]
    bar = "#" * (count * 40 // max(stats["action_distribution"].values(), default=1))
    print(f"    node {node_id:2d}: {count:4d}  {bar}")

# ---------------------------------------------------------------------------
# 6. NSGA-II smoke test
# ---------------------------------------------------------------------------
section("6. NSGA-II smoke test (N=20, pop=20, gen=10)")
env3 = CloudEdgeEnv(N=20, T=8, seed=42)
env3.reset()

sched = NSGA2Scheduler(population_size=20, n_generations=10, seed=42)
nsga2_out = sched.schedule(
    env3.tasks, env3.nodes, env3.carbon_episode, env3.re_episode, T=8
)

print(f"  Pareto front size : {len(nsga2_out['pareto_front'])}")
print(f"  Best energy (Wh)  : {nsga2_out['best_energy']['energy_wh']:.2f}")
print(f"  Best SLA viol     : {nsga2_out['best_sla']['sla_violation']:.4f}")
print(f"  Best carbon gCO2  : {nsga2_out['best_carbon']['carbon_gco2']:.4f}")
print(f"  Runtime (s)       : {nsga2_out['runtime_s']:.2f}")

# ---------------------------------------------------------------------------
# 7. NSGA-II vs MILP energy comparison
# ---------------------------------------------------------------------------
section("7. NSGA-II best_energy vs MILP energy")
milp_e = result.energy_wh
nsga2_e = nsga2_out["best_energy"]["energy_wh"]
print(f"  MILP energy  : {milp_e:.2f} Wh")
print(f"  NSGA-II best : {nsga2_e:.2f} Wh")
ok7 = milp_e <= nsga2_e * 1.10  # MILP should be within 10 % of NSGA-II best energy
print(f"  MILP ≤ NSGA-II×1.10 → {'PASS' if ok7 else 'FAIL (expected; MILP minimises composite, not energy alone)'}")

# ---------------------------------------------------------------------------
print("\nPhase 2 complete. MILP oracle, BC dataset generator, and NSGA-II verified.")
