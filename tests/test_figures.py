"""Tests for Phase 6: figures, plot_config, battery_log, CloudEdgeEnv."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from figures.plot_config import (
    apply_paper_style, save_figure,
    METHOD_COLORS, METHOD_MARKERS,
    SINGLE_COL_W, DOUBLE_COL_W,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------

def _make_monitor_df(n: int = 5) -> pd.DataFrame:
    """Synthetic VecMonitor DataFrame with columns r, l, t."""
    return pd.DataFrame({
        "r": np.linspace(-0.5, -0.2, n),
        "l": np.full(n, 100.0),
        "t": np.arange(n, dtype=float) * 100.0,
    })


def _make_scalability_df() -> pd.DataFrame:
    rows = []
    for N in [10, 20]:
        for method in ["MILP", "NSGA-II", "Carbon-SLA-Net"]:
            rows.append({
                "N": N, "method": method,
                "mean_energy_wh": N * 55.0 + {"MILP": 0, "NSGA-II": 200, "Carbon-SLA-Net": 100}[method],
                "ci95_energy_wh": 20.0,
                "mean_sla_violation": 1.0,
                "mean_carbon_gco2": 5.0,
                "mean_inference_ms": {"MILP": 5000, "NSGA-II": 3000, "Carbon-SLA-Net": 10}[method],
            })
    return pd.DataFrame(rows)


def _make_pareto_df() -> pd.DataFrame:
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


def _make_generalization_df() -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# 1–7: plot_config tests
# ---------------------------------------------------------------------------

def test_apply_paper_style_no_error():
    """1. apply_paper_style() runs without error."""
    apply_paper_style()


def test_method_colors_count():
    """2. METHOD_COLORS has exactly 10 keys."""
    assert len(METHOD_COLORS) == 10


def test_method_markers_contains_drl():
    """3. METHOD_MARKERS contains key 'Carbon-SLA-Net'."""
    assert "Carbon-SLA-Net" in METHOD_MARKERS


def test_save_figure_png_exists(tmp_path):
    """4. save_figure() saves a PNG file that exists on disk."""
    apply_paper_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png_path = save_figure(fig, "test_fig", str(tmp_path))
    assert Path(png_path).exists()


def test_save_figure_pdf_exists(tmp_path):
    """5. save_figure() saves a PDF file that exists on disk."""
    apply_paper_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_figure(fig, "test_fig_pdf", str(tmp_path))
    assert (tmp_path / "test_fig_pdf.pdf").exists()


def test_single_col_width():
    """6. plot_config.SINGLE_COL_W == 3.5."""
    assert SINGLE_COL_W == 3.5


def test_double_col_width():
    """7. plot_config.DOUBLE_COL_W == 7.16."""
    assert DOUBLE_COL_W == 7.16


# ---------------------------------------------------------------------------
# 8–12: Figure function tests with synthetic data
# ---------------------------------------------------------------------------

def test_plot_training_curves_no_error(tmp_path):
    """8. plot_training_curves() with synthetic monitor df runs without error."""
    from figures.figure2_training import plot_training_curves
    path = plot_training_curves(output_dir=str(tmp_path), data=_make_monitor_df())
    assert path is not None


def test_plot_training_curves_returns_png(tmp_path):
    """9. plot_training_curves() returns a path ending with .png."""
    from figures.figure2_training import plot_training_curves
    path = plot_training_curves(output_dir=str(tmp_path), data=_make_monitor_df())
    assert path.endswith(".png")


def test_plot_scalability_no_error(tmp_path):
    """10. plot_scalability() with synthetic scalability CSV runs without error."""
    from figures.figure3_scalability import plot_scalability
    plot_scalability(output_dir=str(tmp_path), data=_make_scalability_df())


def test_plot_pareto_front_no_error(tmp_path):
    """11. plot_pareto_front() with synthetic Pareto CSV runs without error."""
    from figures.figure4_pareto import plot_pareto_front
    plot_pareto_front(output_dir=str(tmp_path), data=(_make_pareto_df(), None))


def test_plot_generalization_heatmap_no_error(tmp_path):
    """12. plot_generalization_heatmap() with synthetic generalization CSV runs without error."""
    from figures.figure7_generalization import plot_generalization_heatmap
    plot_generalization_heatmap(output_dir=str(tmp_path), data=_make_generalization_df())


# ---------------------------------------------------------------------------
# 13–15: CloudEdgeEnv battery_log tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env_episode():
    """Run one full episode and return the env."""
    from env.cloud_edge_env import CloudEdgeEnv
    env = CloudEdgeEnv(N=20, T=8, seed=42)
    env.reset(seed=42)
    for _ in range(len(env.tasks)):
        if env.task_idx >= len(env.tasks):
            break
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    return env


def test_plot_battery_dynamics_no_error(tmp_path, env_episode):
    """13. plot_battery_dynamics() runs without error after adding battery_log."""
    from figures.figure8_battery import plot_battery_dynamics
    path = plot_battery_dynamics(
        output_dir=str(tmp_path),
        model_path=None,
        battery_log=env_episode.battery_log,
    )
    assert path is not None


def test_battery_log_after_reset():
    """14. CloudEdgeEnv has a battery_log attribute after reset()."""
    from env.cloud_edge_env import CloudEdgeEnv
    env = CloudEdgeEnv(N=10, T=8, seed=0)
    env.reset(seed=0)
    assert hasattr(env, "battery_log")
    assert isinstance(env.battery_log, dict)


def test_battery_log_populated_after_step(env_episode):
    """15. battery_log contains node_id keys after at least one step()."""
    assert len(env_episode.battery_log) > 0
    # Check structure of first entry
    first_key = next(iter(env_episode.battery_log))
    entries = env_episode.battery_log[first_key]
    assert len(entries) > 0
    assert "soc_fraction" in entries[0]
    assert "b_charge_wh" in entries[0]
    assert "b_discharge_wh" in entries[0]
