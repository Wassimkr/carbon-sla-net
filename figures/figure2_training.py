"""Figure 2 — Training reward progression."""

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
    METHOD_COLORS, SINGLE_COL_W, FIG_HEIGHT,
    FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND, FONT_ANNOT,
)

_MILP_REWARD = -0.2873   # actual MILP replay reward from evaluation
_DRL_PLATEAU = -0.30     # level where agent stabilised during training


def plot_training_curves(
    monitor_csv_path: str = "results/ppo/eval_rewards.csv",
    output_dir: str = "results/figures",
    data: Optional[pd.DataFrame] = None,
) -> str:
    """Plot rolling reward from eval_rewards.csv.

    Parameters
    ----------
    monitor_csv_path:
        Path to results/ppo/eval_rewards.csv (10 rows, one per 100 k step).
    output_dir:
        Directory for output files.
    data:
        Pre-loaded DataFrame with columns ``timestep, mean_reward``.
        If provided, CSV loading is skipped (used for testing).
    """
    apply_paper_style()

    if data is not None:
        df = data.copy()
        # Accept legacy monitor format (r, l, t columns) from tests
        if "r" in df.columns and "timestep" not in df.columns:
            lengths = df["l"].values.astype(float) if "l" in df.columns else np.ones(len(df)) * 100
            df["timestep"] = np.cumsum(lengths)
            df["mean_reward"] = df["r"].values.astype(float)
    else:
        df = pd.read_csv(monitor_csv_path)
        # Normalise column names: accept both 'timestep' and 'r'/'l'/'t' formats
        if "mean_reward" not in df.columns and "r" in df.columns:
            lengths = df["l"].values.astype(float) if "l" in df.columns else np.ones(len(df)) * 100
            df["timestep"] = np.cumsum(lengths)
            df["mean_reward"] = df["r"].values.astype(float)

    df = df.sort_values("timestep").reset_index(drop=True)
    ts_k = df["timestep"].values / 1_000.0   # convert to thousands
    rewards = df["mean_reward"].values.astype(float)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, FIG_HEIGHT))

    # Reward curve
    ax.plot(ts_k, rewards, color=METHOD_COLORS["Carbon-SLA-Net"],
            linewidth=1.5, marker="o", markersize=4,
            label="Carbon-SLA-Net", zorder=3)

    # MILP reference line (actual replay reward)
    ax.axhline(_MILP_REWARD, color=METHOD_COLORS["MILP"],
               linestyle="--", linewidth=1.0, alpha=0.9, zorder=2)
    # Annotate inside right side of plot
    ax.text(950, _MILP_REWARD + 0.004, "MILP",
            fontsize=FONT_ANNOT, color=METHOD_COLORS["MILP"],
            ha="right", va="bottom")

    # DRL plateau line
    ax.axhline(_DRL_PLATEAU, color="gray",
               linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)
    ax.text(950, _DRL_PLATEAU + 0.004, "DRL plateau",
            fontsize=FONT_ANNOT, color="gray",
            ha="right", va="bottom")

    ax.set_xlim(50, 1050)
    ax.set_ylim(-0.35, -0.10)
    ax.set_title("Training reward progression", fontsize=FONT_TITLE)
    ax.set_xlabel("Timesteps (×10³)", fontsize=FONT_AXIS)
    ax.set_ylabel("Mean episode reward", fontsize=FONT_AXIS)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="lower right")

    return save_figure(fig, "fig2_training_curves", output_dir)
