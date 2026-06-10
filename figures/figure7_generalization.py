"""Figure 7 — Generalization heatmap (workload pattern × renewable condition)."""

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
    DOUBLE_COL_W, FIG_HEIGHT, FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_ANNOT,
)

_PATTERNS   = ["uniform", "bursty", "heavy", "light"]
_CONDITIONS = ["no_re", "cloudy", "sunny"]
_MODEL_PATH = "checkpoints/ppo/carbon_sla_net_best/best_model.zip"
_GEN_CSV    = "results/tables/generalization_results.csv"


def _ensure_generalization_csv(csv_path: str) -> None:
    """Run GeneralizationEvaluator if CSV is missing."""
    p = Path(csv_path)
    if p.exists():
        return
    print("  [fig7] generalization_results.csv not found — running evaluation …")
    from evaluation.generalization import GeneralizationEvaluator
    ev = GeneralizationEvaluator(N=100, T=8, n_episodes=20, seed_offset=40_000)
    ev.run(drl_model_path=_MODEL_PATH, save_dir=str(p.parent))


def plot_generalization_heatmap(
    generalization_csv: str = _GEN_CSV,
    output_dir: str = "results/figures",
    data: Optional[pd.DataFrame] = None,
) -> str:
    """2-panel heatmap: energy and SLA violation by pattern × condition.

    Parameters
    ----------
    generalization_csv:
        Path to generalization_results.csv.  If the file does not exist,
        the evaluation is run inline to produce it.
    output_dir:
        Output directory.
    data:
        Pre-loaded DataFrame; if provided, CSV loading (and inline eval)
        is skipped.
    """
    apply_paper_style()

    if data is not None:
        df = data
    else:
        _ensure_generalization_csv(generalization_csv)
        df = pd.read_csv(generalization_csv)

    # Normalise condition column: the evaluator uses ['sunny','cloudy','no_re']
    # but the heatmap shows them in _CONDITIONS order
    patterns   = [p for p in _PATTERNS   if p in df["pattern"].values]
    conditions = [c for c in _CONDITIONS  if c in df["condition"].values]

    # Fall back to whatever is in the CSV if our lists produce nothing
    if not patterns:
        patterns = sorted(df["pattern"].unique())
    if not conditions:
        conditions = sorted(df["condition"].unique())

    def make_matrix(col: str) -> np.ndarray:
        mat = np.full((len(patterns), len(conditions)), np.nan)
        for r, pat in enumerate(patterns):
            for c, cond in enumerate(conditions):
                row = df[(df["pattern"] == pat) & (df["condition"] == cond)]
                if not row.empty:
                    mat[r, c] = float(row[col].iloc[0])
        return mat

    energy_mat = make_matrix("mean_energy_wh")
    sla_mat    = make_matrix("mean_sla_violation")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, FIG_HEIGHT))
    fig.suptitle(
        "Generalization across workload patterns and renewable conditions",
        fontsize=FONT_TITLE,
    )

    for ax, mat, title, fmt in [
        (axes[0], energy_mat, "Mean energy (Wh)", ".0f"),
        (axes[1], sla_mat,    "Mean SLA violation", ".2f"),
    ]:
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, pad=0.02)

        for r in range(len(patterns)):
            for c in range(len(conditions)):
                val = mat[r, c]
                if not np.isnan(val):
                    threshold = np.nanmax(mat) * 0.7
                    text_color = "white" if val > threshold else "black"
                    ax.text(c, r, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=FONT_ANNOT, color=text_color)

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=FONT_TICK)
        ax.set_yticks(range(len(patterns)))
        ax.set_yticklabels(patterns, fontsize=FONT_TICK)
        ax.set_xlabel("Renewable condition", fontsize=FONT_AXIS)
        ax.set_ylabel("Workload pattern", fontsize=FONT_AXIS)
        ax.set_title(title, fontsize=FONT_AXIS)

    fig.tight_layout()
    return save_figure(fig, "fig7_generalization", output_dir)
