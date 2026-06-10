"""Figure 3 — Scalability: runtime, energy, and MILP gap vs N."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from figures.plot_config import (
    apply_paper_style, save_figure,
    METHOD_COLORS, METHOD_MARKERS, LINE_STYLES,
    DOUBLE_COL_W, FIG_HEIGHT, FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND, FONT_ANNOT,
)

_TIMEOUT_MS = 60_000.0   # MILP time-limit (ms)
_METHODS = ["MILP", "NSGA-II", "Carbon-SLA-Net"]


def plot_scalability(
    scalability_csv: str = "results/tables/scalability_results.csv",
    output_dir: str = "results/figures",
    data: Optional[pd.DataFrame] = None,
) -> str:
    """Plot 3-panel scalability figure.

    Parameters
    ----------
    scalability_csv:
        Path to scalability_results.csv.
    output_dir:
        Output directory.
    data:
        Pre-loaded DataFrame; if provided, CSV is not read.
    """
    apply_paper_style()

    df = data if data is not None else pd.read_csv(scalability_csv)

    # Load MILP gap from companion CSV
    gap_csv = Path(scalability_csv).parent / "milp_gap_vs_N.csv"
    if data is None and gap_csv.exists():
        gap_df = pd.read_csv(str(gap_csv))
    else:
        gap_df = pd.DataFrame(columns=["N", "method", "milp_gap_pct"])

    N_vals = sorted(df["N"].unique())
    methods_present = [m for m in _METHODS if m in df["method"].values]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_W, FIG_HEIGHT))

    # ------------------------------------------------------------------ #
    # Panel 1 — Runtime (log scale)                                       #
    # ------------------------------------------------------------------ #
    ax1 = axes[0]
    for method in methods_present:
        sub = df[df["method"] == method].sort_values("N")
        color = METHOD_COLORS.get(method, "gray")
        marker = METHOD_MARKERS.get(method, "o")
        ls = LINE_STYLES.get(method, "-")

        # Separate normal points from timeout points for MILP
        if method == "MILP":
            normal = sub[sub["mean_inference_ms"] < _TIMEOUT_MS]
            timeout = sub[sub["mean_inference_ms"] >= _TIMEOUT_MS]
            if not normal.empty:
                ax1.plot(normal["N"], normal["mean_inference_ms"],
                         color=color, marker=marker, linestyle=ls,
                         label=method, markersize=4, linewidth=1.2)
            if not timeout.empty:
                ax1.plot(timeout["N"], timeout["mean_inference_ms"],
                         color=color, marker=marker, linestyle=ls,
                         markersize=4, linewidth=1.2,
                         markerfacecolor="none", markeredgewidth=1.2)
                if normal.empty:
                    ax1.lines[-1].set_label(method)
        else:
            ax1.plot(sub["N"], sub["mean_inference_ms"], color=color,
                     marker=marker, linestyle=ls, label=method,
                     markersize=4, linewidth=1.2)

    ax1.set_yscale("log")
    ax1.set_ylim(0.1, 200_000)

    # Timeout zone shaded band
    ax1.axhspan(_TIMEOUT_MS, 200_000, alpha=0.15, color="red", zorder=0)
    ax1.text(N_vals[0] + 2, _TIMEOUT_MS * 1.4,
             "MILP timeout zone", fontsize=FONT_ANNOT, color="red", alpha=0.85)

    # Reference line at N*=100
    if 100 in N_vals:
        ax1.axvline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax1.text(102, 0.2, "N*=100", fontsize=FONT_ANNOT, color="gray", va="bottom")

    ax1.set_xlabel("N (tasks)", fontsize=FONT_AXIS)
    ax1.set_ylabel("Inference time (ms)", fontsize=FONT_AXIS)
    ax1.set_title("Runtime vs N", fontsize=FONT_TITLE)
    ax1.tick_params(labelsize=FONT_TICK)

    # ------------------------------------------------------------------ #
    # Panel 2 — Energy vs N with CI bands                                 #
    # ------------------------------------------------------------------ #
    ax2 = axes[1]
    for method in methods_present:
        sub = df[df["method"] == method].sort_values("N")
        color = METHOD_COLORS.get(method, "gray")
        marker = METHOD_MARKERS.get(method, "o")
        ls = LINE_STYLES.get(method, "-")
        ax2.plot(sub["N"], sub["mean_energy_wh"], color=color,
                 marker=marker, linestyle=ls, label=method,
                 markersize=4, linewidth=1.2)
        if "ci95_energy_wh" in sub.columns:
            ax2.fill_between(
                sub["N"],
                sub["mean_energy_wh"] - sub["ci95_energy_wh"],
                sub["mean_energy_wh"] + sub["ci95_energy_wh"],
                alpha=0.2, color=color,
            )

    ax2.set_xlabel("N (tasks)", fontsize=FONT_AXIS)
    ax2.set_ylabel("Composite energy (Wh)", fontsize=FONT_AXIS)
    ax2.set_title("Energy vs N", fontsize=FONT_TITLE)
    ax2.tick_params(labelsize=FONT_TICK)

    # ------------------------------------------------------------------ #
    # Panel 3 — MILP gap vs N                                             #
    # ------------------------------------------------------------------ #
    ax3 = axes[2]
    drl_gap = gap_df[gap_df["method"] == "Carbon-SLA-Net"].sort_values("N")

    if not drl_gap.empty:
        gap_vals = drl_gap["milp_gap_pct"].values.astype(float)
        N_gap = drl_gap["N"].values
        all_zero = np.all(gap_vals == 0.0)

        if all_zero:
            ax3.text(
                0.5, 0.5, "MILP optimal at all N",
                transform=ax3.transAxes, ha="center", va="center",
                fontsize=FONT_ANNOT, color="gray",
            )
            ax3.set_ylim(0, 20)
        else:
            ax3.fill_between(N_gap, 0, gap_vals,
                             alpha=0.3, color="#ADD8E6", zorder=1)
            ax3.plot(N_gap, gap_vals,
                     color=METHOD_COLORS["Carbon-SLA-Net"],
                     marker="o", linewidth=1.5, markersize=4,
                     label="Carbon-SLA-Net", zorder=2)
            y_min = min(gap_vals.min() - 5, -5)
            y_max = max(gap_vals.max() + 5, 25)
            ax3.set_ylim(y_min, y_max)
    else:
        ax3.set_ylim(0, 20)

    ax3.axhline(5.0, color="gray", linestyle="--", linewidth=0.8, zorder=3)
    ax3.text(N_vals[0] + 2, 5.8, "5% optimal",
             fontsize=FONT_ANNOT, color="gray")
    ax3.axhline(15.0, color="salmon", linestyle="--", linewidth=0.8, zorder=3)
    ax3.text(N_vals[0] + 2, 15.8, "15% target",
             fontsize=FONT_ANNOT, color="salmon")
    ax3.axhline(0.0, color="black", linewidth=0.4, alpha=0.5, zorder=3)

    ax3.set_xlabel("N (tasks)", fontsize=FONT_AXIS)
    ax3.set_ylabel("MILP gap (%)", fontsize=FONT_AXIS)
    ax3.set_title("MILP gap vs N", fontsize=FONT_TITLE)
    ax3.tick_params(labelsize=FONT_TICK)

    # Shared legend below all panels
    handles, labels = [], []
    for ax in axes[:2]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center",
               ncol=len(methods_present), fontsize=FONT_LEGEND,
               bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout(w_pad=2.0)

    return save_figure(fig, "fig3_scalability", output_dir)
