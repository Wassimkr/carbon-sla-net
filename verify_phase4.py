"""Phase 4 verification: baselines, metrics, evaluator, scalability, ablation.

Must be run as a script (not imported) due to macOS 'spawn' multiprocessing.
"""

from __future__ import annotations

import sys
import numpy as np


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Baseline smoke test
    # ------------------------------------------------------------------
    section("1. Baseline smoke test (N=20)")

    from env.cloud_edge_env import CloudEdgeEnv
    from baselines.renewable_greedy import RenewableGreedyScheduler
    from baselines.energy_greedy import EnergyGreedyScheduler
    from baselines.sla_priority import SLAPriorityScheduler

    env = CloudEdgeEnv(N=20, T=8, seed=0)
    env.reset(seed=0)

    rg = RenewableGreedyScheduler().schedule(env)
    eg = EnergyGreedyScheduler().schedule(env)
    sp = SLAPriorityScheduler().schedule(env)

    task_ids = {t.task_id for t in env.tasks}
    J = len(env.nodes)
    for name, asgn in [("RenewableGreedy", rg), ("EnergyGreedy", eg), ("SLAPriority", sp)]:
        assert set(asgn.keys()) == task_ids, f"{name}: missing task_ids"
        assert all(0 <= v < J for v in asgn.values()), f"{name}: invalid node id"
        print(f"  {name}: {len(asgn)} assignments, nodes in [0, {J-1}]  OK")

    # ------------------------------------------------------------------
    # 2. Metrics smoke test
    # ------------------------------------------------------------------
    section("2. Metrics smoke test")

    from evaluation.metrics import (
        EpisodeMetrics, AggregatedMetrics, aggregate_metrics,
        milp_gap, format_results_table,
    )

    eps = [
        EpisodeMetrics(energy_wh=100.0 * (i + 1), carbon_gco2=10.0 * (i + 1),
                       sla_violation=float(i), re_used_wh=5.0 * (i + 1),
                       inference_ms=2.0, n_tasks_assigned=20, reward=-0.1 * (i + 1))
        for i in range(5)
    ]
    agg = aggregate_metrics(eps)
    assert abs(agg.mean_energy_wh - 300.0) < 1e-6, f"mean energy wrong: {agg.mean_energy_wh}"
    assert abs(agg.mean_carbon_gco2 - 30.0) < 1e-6
    assert agg.n_episodes == 5
    expected_composite = 300.0 + 0.1 * 30.0
    assert abs(agg.composite_energy - expected_composite) < 1e-6
    print(f"  mean_energy_wh={agg.mean_energy_wh:.1f}  mean_carbon={agg.mean_carbon_gco2:.1f}  composite={agg.composite_energy:.1f}  OK")

    gap = milp_gap(800.0, 700.0)
    assert abs(gap - (100.0 / 700.0 * 100.0)) < 1e-6
    assert milp_gap(700.0, 700.0) == 0.0
    assert milp_gap(500.0, 0.0) == float("inf")
    print(f"  milp_gap checks: OK")

    try:
        aggregate_metrics([])
        print("  ERROR: should have raised ValueError for empty list")
        sys.exit(1)
    except ValueError:
        print("  aggregate_metrics([]) raises ValueError  OK")

    table = format_results_table({"TestMethod": agg})
    assert "TestMethod" in table
    print(f"  format_results_table: {len(table)} chars, contains 'TestMethod'  OK")

    # ------------------------------------------------------------------
    # 3. Replay smoke test
    # ------------------------------------------------------------------
    section("3. Replay smoke test (N=20, MILP)")

    from milp.scheduler_milp import solve_milp
    from evaluation.evaluator import Evaluator

    eval20 = Evaluator(N=20, T=8, n_episodes=1, seed_offset=99_000)
    env2 = CloudEdgeEnv(N=20, T=8, seed=99_000)
    env2.reset(seed=99_000)

    milp_result = solve_milp(
        env2.tasks, env2.nodes,
        env2.carbon_episode, env2.re_episode,
        T=8, time_limit=30.0, silent=True,
    )
    em = eval20._replay_assignments(milp_result.assignments, env2, seed=99_000)
    assert em.energy_wh > 0.0, "MILP replay: energy_wh should be > 0"
    assert em.n_tasks_assigned > 0, "MILP replay: n_tasks_assigned should be > 0"
    print(f"  MILP replay: energy={em.energy_wh:.2f} Wh  n_assigned={em.n_tasks_assigned}  OK")

    # ------------------------------------------------------------------
    # 4. MILP vs greedy comparison (5 episodes, N=20)
    # ------------------------------------------------------------------
    section("4. MILP vs greedy comparison (5 episodes, N=20)")

    eval5 = Evaluator(N=20, T=8, n_episodes=5, seed_offset=98_000)

    results = {}
    for sched_name, sched in [
        ("RenewableGreedy", RenewableGreedyScheduler()),
        ("EnergyGreedy", EnergyGreedyScheduler()),
        ("SLAPriority", SLAPriorityScheduler()),
    ]:
        results[sched_name] = eval5.run_single(sched_name, sched)

    milp_agg_list = []
    for ep in range(5):
        seed = 98_000 + ep
        env_ep = CloudEdgeEnv(N=20, T=8, seed=seed)
        env_ep.reset(seed=seed)
        mr = solve_milp(
            env_ep.tasks, env_ep.nodes,
            env_ep.carbon_episode, env_ep.re_episode,
            T=8, time_limit=30.0, silent=True,
        )
        em_ep = eval5._replay_assignments(mr.assignments, env_ep, seed=seed)
        milp_agg_list.append(em_ep)
    results["MILP"] = aggregate_metrics(milp_agg_list)

    table = format_results_table(results)
    print(table)

    # ------------------------------------------------------------------
    # 5. Scalability mini-run (N=[10, 20], n_trials=2, fast_mode=True)
    # ------------------------------------------------------------------
    section("5. Scalability mini-run (N=[10,20], fast_mode=True)")

    from evaluation.scalability import ScalabilityEvaluator

    sc = ScalabilityEvaluator(N_values=[10, 20], T=8, n_trials=2, seed_offset=60_000)
    sc_results = sc.run(fast_mode=True)

    assert 10 in sc_results and 20 in sc_results
    assert "MILP" in sc_results[10] and "MILP" in sc_results[20]
    e10 = sc_results[10]["MILP"].mean_energy_wh
    e20 = sc_results[20]["MILP"].mean_energy_wh
    print(f"  N=10 MILP mean_energy={e10:.2f} Wh")
    print(f"  N=20 MILP mean_energy={e20:.2f} Wh")
    assert e20 > e10, f"Expected E(N=20) > E(N=10); got {e20:.2f} vs {e10:.2f}"
    print("  Energy increases with N  OK")

    # ------------------------------------------------------------------
    # 6. Ablation feature zeroing
    # ------------------------------------------------------------------
    section("6. Ablation feature zeroing (index check)")

    from evaluation.ablation import _zero_carbon_features, _zero_battery_features

    rng = np.random.default_rng(42)
    obs = rng.random(74).astype(np.float32)
    # Ensure none of the target indices happen to be exactly 0 already
    obs[[j * 6 + 4 for j in range(11)]] = 0.5
    obs[[j * 6 + 2 for j in range(11)]] = 0.5

    carbon_indices = {j * 6 + 4 for j in range(11)}
    zeroed_c = _zero_carbon_features(obs)
    for idx in carbon_indices:
        assert zeroed_c[idx] == 0.0, f"carbon index {idx} not zeroed"
    for i in range(74):
        if i not in carbon_indices:
            assert zeroed_c[i] == obs[i], f"index {i} incorrectly modified"
    print("  _zero_carbon_features: correct indices zeroed, others unchanged  OK")

    battery_indices = {j * 6 + 2 for j in range(11)}
    zeroed_b = _zero_battery_features(obs)
    for idx in battery_indices:
        assert zeroed_b[idx] == 0.0, f"battery index {idx} not zeroed"
    for i in range(74):
        if i not in battery_indices:
            assert zeroed_b[i] == obs[i], f"index {i} incorrectly modified"
    print("  _zero_battery_features: correct indices zeroed, others unchanged  OK")

    # ------------------------------------------------------------------
    # 7. GeneralizationEvaluator structure
    # ------------------------------------------------------------------
    section("7. GeneralizationEvaluator structure")

    from evaluation.generalization import GeneralizationEvaluator

    assert hasattr(GeneralizationEvaluator, "PATTERNS")
    assert hasattr(GeneralizationEvaluator, "CONDITIONS")
    n_combos = len(GeneralizationEvaluator.PATTERNS) * len(GeneralizationEvaluator.CONDITIONS)
    assert n_combos == 12, f"Expected 12 combos, got {n_combos}"
    print(f"  PATTERNS={GeneralizationEvaluator.PATTERNS}")
    print(f"  CONDITIONS={GeneralizationEvaluator.CONDITIONS}")
    print(f"  {n_combos} combinations  OK")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Phase 4 complete. Evaluation framework verified.")
    print("=" * 60)
