"""Figure 1 — System architecture diagram (text-based schematic).

Produces a clean block-diagram of the Carbon-SLA-Net system using
matplotlib patches and annotations (no external diagram tools required).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from figures.plot_config import (
    apply_paper_style, save_figure,
    METHOD_COLORS, DOUBLE_COL_W, FIG_HEIGHT, FONT_TITLE, FONT_AXIS, FONT_ANNOT,
)


def plot_architecture(output_dir: str = "results/figures") -> str:
    """Draw the Carbon-SLA-Net system architecture block diagram."""
    apply_paper_style()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, FIG_HEIGHT * 1.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Carbon-SLA-Net System Architecture", fontsize=FONT_TITLE)

    def box(x, y, w, h, label, color, fs=7):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor="white",
                                       linewidth=0.8, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray",
                                   lw=0.8, connectionstyle="arc3,rad=0.0"))

    # Infrastructure layer
    box(0.2, 4.5, 1.6, 1.0, "Edge\nNodes\n(×4)", "#BA7517", fs=7)
    box(2.0, 4.5, 1.6, 1.0, "Far-Edge\nNodes\n(×3)", "#888780", fs=7)
    box(3.8, 4.5, 1.6, 1.0, "Cloud\nNodes\n(×4)", "#7F77DD", fs=7)

    # Signals layer
    box(0.2, 3.0, 1.2, 1.0, "RE /\nCarbon\nSignals", "#1D9E75", fs=6)
    box(1.6, 3.0, 1.2, 1.0, "Battery\nSoC", "#1D9E75", fs=6)
    box(3.0, 3.0, 1.4, 1.0, "Workload\nTasks", "#E24B4A", fs=6)

    # Observation builder
    box(0.2, 1.7, 3.5, 0.9, "Observation Builder  (74-dim state vector)", "#378ADD", fs=7)

    # Policy
    box(4.2, 1.7, 2.8, 0.9, "Carbon-SLA-Net Policy\n(MaskablePPO + BC warm-start)", "#378ADD", fs=7)

    # Action masking
    box(7.2, 1.7, 2.5, 0.9, "Action Mask\n(Feasibility filter)", "#F09995", fs=7)

    # Output
    box(3.0, 0.3, 4.0, 0.9, "Task Assignment  →  Env Step  →  Reward", "#1D9E75", fs=7)

    # Arrows
    for x in [1.0, 2.8, 4.6]:
        arrow(x, 4.5, x * 0.85 + 0.3, 3.9)
    arrow(0.8, 3.0, 0.8, 2.6)
    arrow(2.2, 3.0, 2.2, 2.6)
    arrow(3.7, 3.0, 3.0, 2.6)
    arrow(3.75, 2.15, 4.2, 2.15)
    arrow(7.0, 2.15, 7.2, 2.15)
    arrow(5.6, 1.7, 5.0, 1.2)
    arrow(8.4, 1.7, 6.0, 1.2)

    png_path = save_figure(fig, "fig1_architecture", output_dir)
    return png_path
