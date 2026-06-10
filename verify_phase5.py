"""Phase 5 verification: full training pipeline end-to-end check.

Must be run as a script (not imported) due to macOS 'spawn' multiprocessing
required by SubprocVecEnv inside RLTrainer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


if __name__ == "__main__":
    from experiments.configs.main_training import get_default_config, get_fast_config
    from experiments.configs.ablation_configs import get_ablation_configs
    from experiments.run_bc_training import run_bc_training
    from experiments.run_rl_training import run_rl_training
    from experiments.run_pareto_sweep import PARETO_WEIGHT_GRID, run_pareto_sweep
    from experiments.run_full_pipeline import run_full_pipeline

    fast = get_fast_config()
    default = get_default_config()

    # ------------------------------------------------------------------
    # 1. Config check
    # ------------------------------------------------------------------
    section("1. Config check")

    print(f"  {'Setting':<30} {'Default':>12}  {'Fast':>12}")
    print(f"  {'-'*56}")
    for attr in ("N", "ppo_total_timesteps", "ppo_n_envs", "bc_epochs", "bc_patience"):
        dv = getattr(default, attr)
        fv = getattr(fast, attr)
        print(f"  {attr:<30} {str(dv):>12}  {str(fv):>12}")
    print(f"  {'reward_weights':<30} {str(default.reward_weights)}")
    assert default.N == 100 and fast.N == 50
    assert fast.ppo_total_timesteps == 10_000
    print("  Config check  OK")

    # ------------------------------------------------------------------
    # 2. Ablation config check
    # ------------------------------------------------------------------
    section("2. Ablation config check")

    ablation_cfgs = get_ablation_configs()
    assert len(ablation_cfgs) == 5
    print(f"  {'Variant':<25} {'zero_carbon':>12} {'zero_bat':>9} {'no_bc':>6} {'no_mask':>8}")
    print(f"  {'-'*62}")
    for ac in ablation_cfgs:
        print(
            f"  {ac.variant_name:<25} {str(ac.zero_carbon_in_obs):>12} "
            f"{str(ac.zero_battery_in_obs):>9} {str(ac.skip_bc_pretrain):>6} "
            f"{str(ac.disable_action_masking):>8}"
        )
    full = next(c for c in ablation_cfgs if c.variant_name == "full")
    assert not any([
        full.zero_carbon_in_obs, full.zero_battery_in_obs,
        full.skip_bc_pretrain, full.disable_action_masking,
    ])
    print("  Ablation config check  OK")

    # ------------------------------------------------------------------
    # 3. BC training mini-run
    # ------------------------------------------------------------------
    section("3. BC training mini-run (fast config: 5 epochs)")

    t0 = time.perf_counter()
    bc_result = run_bc_training(fast)
    bc_elapsed = time.perf_counter() - t0

    print(f"\n  Summary:")
    print(f"    best_epoch       : {bc_result.best_epoch}/{bc_result.total_epochs_run}")
    print(f"    best_val_loss    : {bc_result.best_val_loss:.4f}")
    print(f"    best_val_acc     : {bc_result.best_val_accuracy:.3f}")
    print(f"    checkpoint_path  : {bc_result.checkpoint_path}")
    print(f"    elapsed          : {bc_elapsed:.1f}s")

    assert Path(bc_result.checkpoint_path).exists(), (
        f"BC checkpoint missing: {bc_result.checkpoint_path}"
    )
    print("  BC training  OK")

    # ------------------------------------------------------------------
    # 4. PPO training mini-run
    # ------------------------------------------------------------------
    section("4. PPO training mini-run (fast config: 10k steps)")

    t0 = time.perf_counter()
    rl_result = run_rl_training(fast, bc_result)
    rl_elapsed = time.perf_counter() - t0

    print(f"\n  Summary:")
    print(f"    total_timesteps  : {rl_result.total_timesteps:,}")
    print(f"    best_mean_reward : {rl_result.best_mean_reward:.4f}")
    print(f"    training_time    : {rl_result.training_time_s:.1f}s ({rl_elapsed:.1f}s wall)")
    print(f"    best_model       : {rl_result.best_model_path}")

    assert Path(rl_result.best_model_path).exists(), (
        f"PPO checkpoint missing: {rl_result.best_model_path}"
    )
    print("  PPO training  OK")

    # ------------------------------------------------------------------
    # 5. Evaluate trained model
    # ------------------------------------------------------------------
    section("5. Evaluate trained model (10 episodes)")

    from sb3_contrib import MaskablePPO
    from agent.rl_trainer import RLTrainer

    trainer = RLTrainer(
        N=fast.N,
        T=fast.T,
        reward_weights=fast.reward_weights,
        workload_pattern=fast.workload_pattern,
        renewable_condition=fast.renewable_condition,
        checkpoint_dir=fast.ppo_checkpoint_dir,
        seed=fast.ppo_seed,
    )
    eval_metrics = trainer.evaluate_model(
        model_path=rl_result.best_model_path,
        n_episodes=10,
        seed_offset=80_000,
    )
    print(f"  mean_reward      : {eval_metrics['mean_reward']:.4f}")
    print(f"  mean_energy_wh   : {eval_metrics['mean_energy_wh']:.2f}")
    print(f"  mean_carbon_gco2 : {eval_metrics['mean_carbon_gco2']:.2f}")
    print(f"  mean_sla_viol    : {eval_metrics['mean_sla_violation']:.3f}")
    print(f"  n_episodes       : {eval_metrics['n_episodes']}")
    assert eval_metrics["n_episodes"] == 10
    print("  Evaluation  OK")

    # ------------------------------------------------------------------
    # 6. Pareto mini-sweep (2 configs, 2048 steps each)
    # ------------------------------------------------------------------
    section("6. Pareto mini-sweep (2 configs × 2048 steps)")

    pareto_results = run_pareto_sweep(
        fast,
        bc_checkpoint_path=bc_result.checkpoint_path,
        timesteps_per_config=2_048,
        max_configs=2,
    )
    assert len(pareto_results) == 2
    for r in pareto_results:
        assert "weights" in r
        assert "mean_energy_wh" in r
        assert "mean_carbon_gco2" in r
        assert "mean_sla_violation" in r
        print(
            f"  weights={r['weights']} | "
            f"E={r['mean_energy_wh']:.1f} | "
            f"CO2={r['mean_carbon_gco2']:.1f} | "
            f"SLA={r['mean_sla_violation']:.3f}"
        )
    print("  Pareto mini-sweep  OK")

    # ------------------------------------------------------------------
    # 7. Full pipeline mini-run
    # ------------------------------------------------------------------
    section("7. Full pipeline mini-run (fast config, pareto_timesteps=2048)")

    pipeline_out = run_full_pipeline(fast, pareto_timesteps=2_048)
    assert "bc_result" in pipeline_out
    assert "rl_result" in pipeline_out
    assert "pareto_results" in pipeline_out
    assert "summary" in pipeline_out
    assert Path("results/pipeline_run_summary.json").exists()
    print(f"  Pipeline returned {len(pipeline_out)} keys  OK")
    print(f"  Summary JSON saved  OK")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Phase 5 complete. Full training pipeline verified.")
    print("=" * 60)
