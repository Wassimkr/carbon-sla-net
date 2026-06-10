"""Figure 5 — Energy and carbon under varying renewable availability."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from figures.plot_config import (
    apply_paper_style, save_figure,
    METHOD_COLORS, METHOD_MARKERS, DOUBLE_COL_W, FIG_HEIGHT,
    FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND, FONT_ANNOT,
)

_CONDITIONS_ORDER  = ["no_re", "cloudy", "sunny"]
_CONDITION_LABELS  = {"no_re": "No RE", "cloudy": "Cloudy", "sunny": "Sunny"}
_METHODS           = ["MILP", "NSGA-II", "Carbon-SLA-Net"]
_GEN_CSV           = "results/tables/generalization_results.csv"
_SCALE_CSV         = "results/tables/scalability_results.csv"
_MODEL_PATH        = "checkpoints/ppo/carbon_sla_net_best/best_model.zip"


def _run_generalization_eval(save_dir: str = "results/tables") -> pd.DataFrame:
    """Run GeneralizationEvaluator (DRL only) and return the CSV as DataFrame."""
    from evaluation.generalization import GeneralizationEvaluator
    ev = GeneralizationEvaluator(N=100, T=8, n_episodes=20, seed_offset=40_000)
    ev.run(drl_model_path=_MODEL_PATH, save_dir=save_dir)
    return pd.read_csv(Path(save_dir) / "generalization_results.csv")


def _build_table_from_csvs(
    gen_csv: str = _GEN_CSV,
    scale_csv: str = _SCALE_CSV,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build {condition: {method: {energy, carbon, sla}}} from CSVs.

    DRL values come from the generalization CSV (averaged over workload
    patterns).  MILP and NSGA-II values are proxied from the scalability
    CSV at N=100 (condition-agnostic, same for all three conditions).
    """
    gen_df = pd.read_csv(gen_csv)

    # DRL per condition: average over all workload patterns
    drl_by_cond: Dict[str, Dict[str, float]] = {}
    for cond in _CONDITIONS_ORDER:
        sub = gen_df[gen_df["condition"] == cond]
        if sub.empty:
            continue
        drl_by_cond[cond] = {
            "energy": float(sub["mean_energy_wh"].mean()),
            "carbon": float(sub["mean_carbon_gco2"].mean()),
            "sla":    float(sub["mean_sla_violation"].mean()),
        }

    # MILP / NSGA-II proxies from scalability at N=100
    proxy: Dict[str, Dict[str, float]] = {}
    if Path(scale_csv).exists():
        sc = pd.read_csv(scale_csv)
        for method in ["MILP", "NSGA-II"]:
            row = sc[(sc["method"] == method) & (sc["N"] == 100)]
            if not row.empty:
                proxy[method] = {
                    "energy": float(row["mean_energy_wh"].iloc[0]),
                    "carbon": float(row["mean_carbon_gco2"].iloc[0]),
                    "sla":    float(row["mean_sla_violation"].iloc[0]),
                }

    # Assemble final table
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond in _CONDITIONS_ORDER:
        if cond not in drl_by_cond:
            continue
        out[cond] = {"Carbon-SLA-Net": drl_by_cond[cond]}
        for method in ["MILP", "NSGA-II"]:
            if method in proxy:
                out[cond][method] = proxy[method]
    return out


def plot_renewable_sensitivity(
    results_by_condition: Optional[Dict] = None,
    output_dir: str = "results/figures",
    generalization_csv: str = _GEN_CSV,
    scalability_csv: str = _SCALE_CSV,
) -> str:
    """Grouped bar chart of composite energy by renewable condition.

    Parameters
    ----------
    results_by_condition:
        Legacy interface: ``{condition: {method: AggregatedMetrics}}``.
        Pass ``None`` or an empty dict to use the CSV-based fallback.
    output_dir:
        Output directory.
    generalization_csv:
        Path to generalization_results.csv for DRL data.
    scalability_csv:
        Path to scalability_results.csv for MILP/NSGA-II proxies.
    """
    apply_paper_style()

    # ------------------------------------------------------------------ #
    # Resolve data source                                                  #
    # ------------------------------------------------------------------ #
    use_csv = not results_by_condition  # empty dict or None → CSV path

    table: Dict[str, Dict[str, Dict[str, float]]] = {}

    if not use_csv:
        # Legacy path: results_by_condition has AggregatedMetrics objects
        for cond, method_dict in results_by_condition.items():
            table[cond] = {}
            for method, agg in method_dict.items():
                table[cond][method] = {
                    "energy": float(getattr(agg, "composite_energy",
                                            getattr(agg, "mean_energy_wh", 0))),
                    "carbon": float(getattr(agg, "mean_carbon_gco2", 0)),
                    "sla":    float(getattr(agg, "mean_sla_violation", 0)),
                }
    else:
        gen_csv_path = Path(generalization_csv)
        if not gen_csv_path.exists():
            print("  [fig5] generalization_results.csv not found — running evaluation …")
            _run_generalization_eval(save_dir=str(gen_csv_path.parent))
        table = _build_table_from_csvs(generalization_csv, scalability_csv)

    # ------------------------------------------------------------------ #
    # Plot                                                                 #
    # ------------------------------------------------------------------ #
    conditions = [c for c in _CONDITIONS_ORDER if c in table]
    methods = [m for m in _METHODS if any(m in table.get(c, {}) for c in conditions)]

    if not conditions or not methods:
        # Nothing to plot — create placeholder
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, FIG_HEIGHT))
        ax.text(0.5, 0.5, "No RE sensitivity data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_ANNOT, color="gray")
        ax.set_title("Energy and carbon under varying renewable availability",
                     fontsize=FONT_TITLE)
        return save_figure(fig, "fig5_renewable_sensitivity", output_dir)

    n_conditions = len(conditions)
    n_methods = len(methods)
    x = np.arange(n_conditions)
    bar_w = 0.22
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_w

    fig, ax1 = plt.subplots(figsize=(DOUBLE_COL_W, FIG_HEIGHT))
    ax2 = ax1.twinx()

    for k, method in enumerate(methods):
        color = METHOD_COLORS.get(method, "gray")
        marker = METHOD_MARKERS.get(method, "o")
        energies, carbons = [], []
        for cond in conditions:
            vals = table.get(cond, {}).get(method, {})
            energies.append(vals.get("energy", 0.0))
            carbons.append(vals.get("carbon", 0.0))

        ax1.bar(x + offsets[k], energies, bar_w,
                color=color, alpha=0.85, label=method)
        ax2.plot(x + offsets[k], carbons,
                 color=color, marker=marker,
                 linestyle="-", linewidth=1.0, markersize=3, alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels([_CONDITION_LABELS.get(c, c) for c in conditions],
                        fontsize=FONT_TICK)
    ax1.set_xlabel("Renewable condition", fontsize=FONT_AXIS)
    ax1.set_ylabel("Mean energy (Wh)", fontsize=FONT_AXIS)
    ax2.set_ylabel("Carbon (gCO₂)", fontsize=FONT_AXIS)
    ax1.set_title("Energy and carbon under varying renewable availability",
                  fontsize=FONT_TITLE)
    ax1.tick_params(labelsize=FONT_TICK)
    ax2.tick_params(labelsize=FONT_TICK)
    ax1.legend(fontsize=FONT_LEGEND, loc="upper right")
    fig.tight_layout()

    return save_figure(fig, "fig5_renewable_sensitivity", output_dir)
