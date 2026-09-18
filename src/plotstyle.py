"""Shared figure style for the manuscript.

The target journal accepts TIFF or EPS at 300-600 dpi, with a figure width
between 789 and 2250 pixels, text set only in Arial, Times or Symbol at 8-12 pt,
and no author names, article title or figure number inside the image file.

Import ``apply_style()`` at the top of every figure script so that all panels
share one look and one set of sizes.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Arial is not installed on every machine; the first available family is used.
FONT_CANDIDATES = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

# Journal limits, kept here so that scripts can assert against them.
MIN_WIDTH_PX = 789
MAX_WIDTH_PX = 2250
MIN_DPI = 300
MAX_DPI = 600
MAX_FILE_MB = 10.0

BASE_FONT_PT = 9
LABEL_FONT_PT = 10
TITLE_FONT_PT = 10


def apply_style():
    """Set matplotlib defaults compatible with the journal figure rules."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_CANDIDATES,
        "font.size": BASE_FONT_PT,
        "axes.labelsize": LABEL_FONT_PT,
        "axes.titlesize": TITLE_FONT_PT,
        "xtick.labelsize": BASE_FONT_PT,
        "ytick.labelsize": BASE_FONT_PT,
        "legend.fontsize": BASE_FONT_PT,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "figure.dpi": MIN_DPI,
        "savefig.dpi": MIN_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType, avoids missing-font problems
        "ps.fonttype": 42,
    })


# A colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE = {
    "ours": "#D55E00",
    "ablation": "#0072B2",
    "oracle": "#009E73",
    "robust": "#E69F00",
    "transfer": "#CC79A7",
    "retrain": "#56B4E9",
    "const": "#999999",
    "accent": "#000000",
    "shade": "#DDDDDD",
}

MARKERS = {
    "ours": "o",
    "ablation": "s",
    "oracle": "^",
    "robust": "D",
    "transfer": "v",
    "retrain": "<",
    "const": "x",
}


def panel_label(ax, letter, dx=-0.12, dy=1.04):
    """Add a bold panel letter in the style the journal expects."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=TITLE_FONT_PT, fontweight="bold", va="bottom", ha="left")


# Figure widths in inches that stay inside the journal pixel limits at 300 dpi.
# 7.5 in -> 2250 px (full page width). 2.63 in -> 789 px (minimum).
WIDTH_SINGLE_COL = 3.40   # 1020 px
WIDTH_ONE_HALF = 5.20     # 1560 px, aligns with the text column of the PDF
WIDTH_DOUBLE_COL = 7.20   # 2160 px, safely under the 2250 px ceiling


def assert_width(fig, dpi: int = MIN_DPI):
    """Raise if the rendered figure would break the journal width limits."""
    px = int(round(fig.get_size_inches()[0] * dpi))
    if px < MIN_WIDTH_PX or px > MAX_WIDTH_PX:
        raise ValueError(
            f"figure width {px} px at {dpi} dpi is outside the allowed range "
            f"[{MIN_WIDTH_PX}, {MAX_WIDTH_PX}]. Use one of WIDTH_SINGLE_COL, "
            f"WIDTH_ONE_HALF or WIDTH_DOUBLE_COL in inches.")
    return px


def save_figure(fig, stem, figdir="figures", dpi: int = MIN_DPI):
    """Save one figure as PNG (for review) and PDF (for vector fidelity).

    Enforces the journal width limits before writing anything, so an oversized
    figure can never reach the submission package.
    """
    from pathlib import Path
    px = assert_width(fig, dpi)
    out = Path(figdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=dpi)
    fig.savefig(out / f"{stem}.pdf")
    return {"stem": stem, "width_px": px, "dpi": dpi,
            "png": str(out / f"{stem}.png"), "pdf": str(out / f"{stem}.pdf")}
