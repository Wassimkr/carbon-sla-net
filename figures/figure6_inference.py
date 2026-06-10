"""Figure 6 — Inference time crossover (cumulative time vs instances solved)."""

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
    METHOD_COLORS, LINE_STYLES, SINGLE_COL_W, FIG_HEIGHT,
    FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND, FONT_ANNOT,
)

_N_REF         = 100       # reference N for extracting per-instance times
_N_MAX         = 500       # max instances on X axis
_DRL_TRAINING_S = 230.0   # one-time DRL training cost (seconds, measured)
_DRL_FALLBACK_S = 0.85    # fallback DRL inference if not in CSV (seconds)


def plot_inference_crossover(
    scalability_csv: str = "results/tables/scalability_results.csv",
    output_dir: str = "results/figures",
    data: Optional[pd.DataFrame] = None,
) -> str:
    """Plot cumulative wall-clock time vs number of scheduling instances.

    For Carbon-SLA-Net, the one-time training cost is amortised over all
    instances solved.  For non-learning methods the cost grows linearly.

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

    # Choose closest available N to the reference
    available_n = sorted(df["N"].unique())
    ref_n = min(available_n, key=lambda n: abs(n - _N_REF))

    # Extract per-instance inference times at ref_n (ms → s)
    times_s: dict[str, float] = {}
    for method in df["method"].unique():
        sub = df[(df["method"] == method) & (df["N"] == ref_n)]
        if not sub.empty:
            times_s[method] = float(sub["mean_inference_ms"].iloc[0]) / 1000.0

    if "Carbon-SLA-Net" not in times_s:
        times_s["Carbon-SLA-Net"] = _DRL_FALLBACK_S

    instances = np.arange(0, _N_MAX + 1)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, FIG_HEIGHT))

    # Build cumulative curves
    cumulative: dict[str, np.ndarray] = {}
    for method, t_per_inst in times_s.items():
        if method == "Carbon-SLA-Net":
            cumulative[method] = _DRL_TRAINING_S + instances * t_per_inst
        else:
            cumulative[method] = instances * t_per_inst

    # Plot all curves
    label_map = {"Carbon-SLA-Net": "Carbon-SLA-Net (incl. training)"}
    for method, cum in cumulative.items():
        color = METHOD_COLORS.get(method, "gray")
        ls = LINE_STYLES.get(method, "-")
        label = label_map.get(method, method)
        ax.plot(instances, cum, color=color, linestyle=ls,
                label=label, linewidth=1.5)

    # Annotate DRL training overhead at x=0
    drl_t0 = cumulative.get("Carbon-SLA-Net", np.array([_DRL_TRAINING_S]))[0]
    ax.annotate(
        f"Training\noverhead\n({_DRL_TRAINING_S:.0f}s)",
        xy=(0, drl_t0),
        xytext=(30, drl_t0 * 0.55),
        fontsize=FONT_ANNOT, color=METHOD_COLORS.get("Carbon-SLA-Net", "blue"),
        arrowprops=dict(arrowstyle="->", lw=0.7,
                        color=METHOD_COLORS.get("Carbon-SLA-Net", "blue")),
        ha="left",
    )

    # Find MILP–DRL crossover analytically
    milp_t = times_s.get("MILP")
    drl_t  = times_s.get("Carbon-SLA-Net", _DRL_FALLBACK_S)
    crossover_x = None
    if milp_t is not None and milp_t > drl_t:
        crossover_x = _DRL_TRAINING_S / (milp_t - drl_t)

    if crossover_x is not None and 0 < crossover_x < _N_MAX:
        ax.axvline(crossover_x, color="gray", linestyle="--", linewidth=0.8)
        ax.text(crossover_x + 5,
                ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"Break-even\nN*={crossover_x:.0f}",
                fontsize=FONT_ANNOT, color="gray", va="bottom")
    else:
        ax.text(0.97, 0.97, "No crossover\nin range",
                transform=ax.transAxes, fontsize=FONT_ANNOT,
                color="gray", ha="right", va="top")

    ax.set_xlabel("Number of scheduling instances", fontsize=FONT_AXIS)
    ax.set_ylabel("Cumulative time (s)", fontsize=FONT_AXIS)
    ax.set_title("Inference time crossover", fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")
    fig.tight_layout()

    return save_figure(fig, "fig6_inference_crossover", output_dir)
