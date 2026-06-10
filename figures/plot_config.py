"""Shared matplotlib style, colors, and save helper for all Carbon-SLA-Net figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Method colors — consistent across all figures
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "MILP":            "#1D9E75",
    "NSGA-II":         "#E24B4A",
    "RenewableGreedy": "#BA7517",
    "EnergyGreedy":    "#888780",
    "SLAPriority":     "#7F77DD",
    "Carbon-SLA-Net":  "#378ADD",
    "DRL-no-carbon":   "#F09995",
    "DRL-no-battery":  "#9FE1CB",
    "DRL-no-bc":       "#C0DD97",
    "DRL-no-mask":     "#FAC775",
}

METHOD_MARKERS = {
    "MILP":            "s",
    "NSGA-II":         "^",
    "RenewableGreedy": "D",
    "EnergyGreedy":    "v",
    "SLAPriority":     "p",
    "Carbon-SLA-Net":  "o",
}

LINE_STYLES = {
    "MILP":            "--",
    "NSGA-II":         "-.",
    "RenewableGreedy": ":",
    "EnergyGreedy":    ":",
    "Carbon-SLA-Net":  "-",
}

# ---------------------------------------------------------------------------
# Figure dimensions — IEEE TPDS two-column format (inches)
# ---------------------------------------------------------------------------

SINGLE_COL_W = 3.5
DOUBLE_COL_W = 7.16
FIG_HEIGHT   = 2.8

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------

FONT_TITLE  = 10
FONT_AXIS   = 9
FONT_TICK   = 8
FONT_LEGEND = 8
FONT_ANNOT  = 7

# ---------------------------------------------------------------------------
# DPI
# ---------------------------------------------------------------------------

DPI_SCREEN = 100
DPI_PAPER  = 300


def apply_paper_style() -> None:
    """Apply IEEE TPDS-compatible rcParams globally."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          9,
        "text.usetex":        False,
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
        "lines.linewidth":    1.5,
        "lines.markersize":   5,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.8",
        "figure.dpi":         DPI_SCREEN,
        "savefig.dpi":        DPI_PAPER,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })


def save_figure(fig, name: str, output_dir: str = "results/figures") -> str:
    """Save *fig* as both PNG (300 DPI) and PDF to *output_dir*.

    Returns the PNG path.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    png_path = str(Path(output_dir) / f"{name}.png")
    pdf_path = str(Path(output_dir) / f"{name}.pdf")
    fig.savefig(png_path, dpi=DPI_PAPER)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path} and {pdf_path}")
    plt.close(fig)
    return png_path
