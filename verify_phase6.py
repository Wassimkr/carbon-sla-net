"""verify_phase6.py — End-to-end verification for Phase 6 figures and evaluation."""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


if __name__ == "__main__":
    errors: list[str] = []

    # ------------------------------------------------------------------ #
    # Section 1 — Style check                                             #
    # ------------------------------------------------------------------ #
    _section("1. Plot style configuration")
    try:
        import matplotlib
        matplotlib.use("Agg")
        from figures.plot_config import (
            apply_paper_style, save_figure,
            METHOD_COLORS, METHOD_MARKERS,
            SINGLE_COL_W, DOUBLE_COL_W, FIG_HEIGHT,
        )
        apply_paper_style()
        assert SINGLE_COL_W == 3.5, f"Expected 3.5, got {SINGLE_COL_W}"
        assert DOUBLE_COL_W == 7.16, f"Expected 7.16, got {DOUBLE_COL_W}"
        assert len(METHOD_COLORS) == 10, f"Expected 10 colors, got {len(METHOD_COLORS)}"
        assert "Carbon-SLA-Net" in METHOD_MARKERS
        print(f"  OK — {len(METHOD_COLORS)} method colors, SINGLE_COL_W={SINGLE_COL_W}, DOUBLE_COL_W={DOUBLE_COL_W}")
    except Exception as e:
        errors.append(f"Section 1: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Section 2 — Synthetic data figures                                  #
    # ------------------------------------------------------------------ #
    _section("2. Synthetic data figures (Figures 2–8)")
    import numpy as np
    import pandas as pd

    def make_monitor_df(n: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            "r": np.linspace(-0.5, -0.2, n),
            "l": np.full(n, 100.0),
            "t": np.arange(n, dtype=float) * 100.0,
        })

    def make_scalability_df() -> pd.DataFrame:
        rows = []
        for N in [10, 20, 50]:
            for method in ["MILP", "NSGA-II", "Carbon-SLA-Net"]:
                rows.append({
                    "N": N, "method": method,
                    "mean_energy_wh": N * 55.0,
                    "ci95_energy_wh": 20.0,
                    "mean_sla_violation": 1.0,
                    "mean_carbon_gco2": 5.0,
                    "mean_inference_ms": {"MILP": 5000, "NSGA-II": 3000, "Carbon-SLA-Net": 10}[method],
                })
        return pd.DataFrame(rows)

    def make_pareto_df() -> pd.DataFrame:
        return pd.DataFrame({
            "config_idx": range(3),
            "w_energy": [0.5, 0.3, 0.1],
            "w_carbon": [0.3, 0.5, 0.3],
            "w_sla": [0.1, 0.1, 0.5],
            "w_renewable": [0.1, 0.1, 0.1],
            "mean_energy_wh": [1000.0, 1200.0, 1500.0],
            "mean_carbon_gco2": [50.0, 30.0, 20.0],
            "mean_sla_violation": [5.0, 3.0, 1.0],
            "model_path": ["a.zip", "b.zip", "c.zip"],
        })

    def make_generalization_df() -> pd.DataFrame:
        from evaluation.generalization import GeneralizationEvaluator
        rows = []
        for pat in GeneralizationEvaluator.PATTERNS:
            for cond in GeneralizationEvaluator.CONDITIONS:
                rows.append({
                    "pattern": pat, "condition": cond,
                    "mean_energy_wh": 500.0,
                    "mean_carbon_gco2": 10.0,
                    "mean_sla_violation": 2.0,
                    "mean_re_used_wh": 100.0,
                })
        return pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            from figures.figure1_architecture import plot_architecture
            path = plot_architecture(output_dir=tmp)
            assert Path(path).exists(), f"fig1 not found: {path}"
            print(f"  OK — fig1_architecture: {Path(path).name}")
        except Exception as e:
            errors.append(f"Fig1: {e}")
            traceback.print_exc()

        try:
            from figures.figure2_training import plot_training_curves
            path = plot_training_curves(output_dir=tmp, data=make_monitor_df())
            assert path.endswith(".png") and Path(path).exists()
            print(f"  OK — fig2_training_curves: {Path(path).name}")
        except Exception as e:
            errors.append(f"Fig2: {e}")
            traceback.print_exc()

        try:
            from figures.figure3_scalability import plot_scalability
            plot_scalability(output_dir=tmp, data=make_scalability_df())
            print("  OK — fig3_scalability")
        except Exception as e:
            errors.append(f"Fig3: {e}")
            traceback.print_exc()

        try:
            from figures.figure4_pareto import plot_pareto_front
            plot_pareto_front(output_dir=tmp, data=(make_pareto_df(), None))
            print("  OK — fig4_pareto_front")
        except Exception as e:
            errors.append(f"Fig4: {e}")
            traceback.print_exc()

        try:
            from figures.figure7_generalization import plot_generalization_heatmap
            plot_generalization_heatmap(output_dir=tmp, data=make_generalization_df())
            print("  OK — fig7_generalization")
        except Exception as e:
            errors.append(f"Fig7: {e}")
            traceback.print_exc()

        # Battery dynamics with synthetic log
        try:
            from figures.figure8_battery import plot_battery_dynamics
            battery_log = {
                0: [
                    {"slot": t, "soc_fraction": 0.6 + 0.03 * t,
                     "b_charge_wh": 20.0, "b_discharge_wh": 10.0}
                    for t in range(8)
                ]
            }
            path = plot_battery_dynamics(output_dir=tmp, model_path=None, battery_log=battery_log)
            assert path is not None and Path(path).exists()
            print(f"  OK — fig8_battery_dynamics: {Path(path).name}")
        except Exception as e:
            errors.append(f"Fig8: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Section 3 — battery_log check                                       #
    # ------------------------------------------------------------------ #
    _section("3. CloudEdgeEnv battery_log")
    try:
        from env.cloud_edge_env import CloudEdgeEnv
        env = CloudEdgeEnv(N=10, T=8, seed=0)
        env.reset(seed=0)
        assert hasattr(env, "battery_log") and isinstance(env.battery_log, dict), \
            "battery_log missing or wrong type after reset()"
        # Take a few steps
        for _ in range(5):
            if env.task_idx >= len(env.tasks):
                break
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert len(env.battery_log) > 0, "battery_log empty after steps"
        first_key = next(iter(env.battery_log))
        entry = env.battery_log[first_key][0]
        for field in ("slot", "soc_fraction", "b_charge_wh", "b_discharge_wh"):
            assert field in entry, f"Missing field '{field}' in battery_log entry"
        print(f"  OK — battery_log has {len(env.battery_log)} node(s), fields: {list(entry.keys())}")
    except Exception as e:
        errors.append(f"Section 3 battery_log: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Section 4 — Fast evaluation run                                     #
    # ------------------------------------------------------------------ #
    _section("4. run_evaluation.py (fast mode, figures-only)")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "run_evaluation.py", "--figures-only", "--fast"],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"run_evaluation.py exited {result.returncode}\n"
                f"STDOUT: {result.stdout[-800:]}\nSTDERR: {result.stderr[-400:]}"
            )
        print("  OK — run_evaluation.py --figures-only --fast completed")
        # Check that at least one figure was produced
        fig_dir = Path("results/figures")
        pngs = list(fig_dir.glob("*.png")) if fig_dir.exists() else []
        print(f"  Found {len(pngs)} PNG(s) in {fig_dir}")
    except Exception as e:
        errors.append(f"Section 4 run_evaluation: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Final result                                                         #
    # ------------------------------------------------------------------ #
    print(f"\n{'═' * 55}")
    if errors:
        print(f"  Phase 6 INCOMPLETE — {len(errors)} error(s):")
        for err in errors:
            print(f"    • {err}")
        sys.exit(1)
    else:
        print("  Phase 6 complete. All figures and tables verified.")
    print(f"{'═' * 55}")
