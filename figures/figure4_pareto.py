"""Figure 4 — Energy–SLA Pareto front with carbon colorbar."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from figures.plot_config import (
    apply_paper_style, save_figure,
    METHOD_COLORS, SINGLE_COL_W, FIG_HEIGHT,
    FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND, FONT_ANNOT,
)

# Hardcoded MILP reference from evaluation
_MILP_ENERGY  = 6862.4
_MILP_SLA     = 0.0
_MILP_CARBON  = 32.5


def plot_pareto_front(
    pareto_csv: str = "results/tables/pareto_sweep_results.csv",
    milp_results_csv: str = "results/tables/main_results.csv",
    output_dir: str = "results/figures",
    data: Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = None,
) -> str:
    """Plot Energy–SLA Pareto front scatter with carbon colorbar.

    Parameters
    ----------
    pareto_csv:
        Path to pareto_sweep_results.csv.
    milp_results_csv:
        Path to main_results.csv (for MILP overlay).
    output_dir:
        Output directory.
    data:
        ``(pareto_df, milp_df)`` pre-loaded DataFrames; if provided,
        CSVs are not read.  ``milp_df`` may be None.
    """
    apply_paper_style()

    if data is not None:
        pareto_df, milp_df = data
    else:
        pareto_df = pd.read_csv(pareto_csv)
        milp_df = pd.read_csv(milp_results_csv) if Path(milp_results_csv).exists() else None

    energy = pareto_df["mean_energy_wh"].values.astype(float)
    sla    = pareto_df["mean_sla_violation"].values.astype(float)
    carbon = pareto_df["mean_carbon_gco2"].values.astype(float)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, FIG_HEIGHT * 1.4))

    # Scatter all 7 Pareto sweep points coloured by carbon
    sc = ax.scatter(energy, sla, c=carbon, cmap="RdYlGn_r",
                    s=80, zorder=4, edgecolors="white", linewidths=0.5)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Carbon (gCO₂)", fontsize=FONT_LEGEND)
    cb.ax.tick_params(labelsize=FONT_TICK)

    # Connect all 7 points sorted by energy (no Pareto filtering)
    sort_idx = np.argsort(energy)
    ax.plot(energy[sort_idx], sla[sort_idx],
            color="steelblue", linestyle="--", linewidth=1.0,
            alpha=0.7, zorder=3, label="Pareto sweep")

    # Resolve MILP reference values
    milp_e, milp_s = _MILP_ENERGY, _MILP_SLA
    if milp_df is not None and "method" in milp_df.columns:
        row = milp_df[milp_df["method"] == "MILP"]
        if not row.empty and "mean_energy_wh" in row.columns:
            milp_e = float(row["mean_energy_wh"].iloc[0])
            milp_s = float(row.get("mean_sla_violation", pd.Series([_MILP_SLA])).iloc[0])

    # MILP as gold star with arrow annotation
    ax.scatter(milp_e, milp_s, marker="*", s=200,
               color="gold", edgecolors="black", linewidths=0.8,
               zorder=5, label="MILP")

    offset_x = (energy.max() - energy.min()) * 0.12
    offset_y = (sla.max() - sla.min()) * 0.12 + 0.5
    ax.annotate(
        "MILP",
        xy=(milp_e, milp_s),
        xytext=(milp_e + offset_x, milp_s + offset_y),
        fontsize=FONT_ANNOT,
        arrowprops=dict(arrowstyle="->", color="black",
                        lw=0.7, shrinkA=4, shrinkB=4),
        ha="left", va="bottom",
    )

    # "Lower-left = better" note in bottom-right
    ax.text(0.97, 0.04, "Lower-left = better",
            transform=ax.transAxes, fontsize=FONT_ANNOT,
            color="gray", style="italic", ha="right", va="bottom")

    ax.set_xlabel("Mean energy (Wh)", fontsize=FONT_AXIS)
    ax.set_ylabel("Mean SLA violation", fontsize=FONT_AXIS)
    ax.set_title("Energy–SLA Pareto front (Carbon-SLA-Net)", fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="upper right")
    fig.tight_layout()

    return save_figure(fig, "fig4_pareto_front", output_dir)
