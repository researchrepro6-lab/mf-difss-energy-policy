"""Common figure style: LaTeX-like fonts, fixed sizes and journal-ready PNG output.

All matplotlib figures are drawn at the text width of the Elsevier CAS single-column
layout (16.46 cm = 6.48 in) and included at \\textwidth, so figure text appears in the
manuscript at its nominal size: 8 pt for axis labels and panel titles, 7 pt for tick
labels, legends and annotations. Text is set in Latin Modern Sans and symbols in
Computer Modern math, matching the sans-serif captions and the math of the manuscript.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "fonts"
TEXTWIDTH_IN = 6.48          # CAS single-column text width (468.33 pt)
LABEL_PT, SMALL_PT = 8, 7

# colour palette shared by all figures
C_POL, C_SHR, C_CO2 = "#3B5BA5", "#2E8B57", "#8C5A3C"     # policy, renewable share, CO2
C_MAIN, C_GREY, C_AMBER = "#C0503F", "#9AA5B1", "#D08414"


def apply() -> None:
    for f in sorted(FONT_DIR.glob("lmsans10-*.otf")):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
        "mathtext.fontset": "cm", "font.size": LABEL_PT,
        "axes.titlesize": LABEL_PT, "axes.labelsize": LABEL_PT, "axes.titleweight": "normal",
        "axes.titlelocation": "left", "axes.titlepad": 4,
        "xtick.labelsize": SMALL_PT, "ytick.labelsize": SMALL_PT,
        "legend.fontsize": SMALL_PT, "legend.title_fontsize": SMALL_PT, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.2, "lines.markersize": 4, "patch.linewidth": 0.5,
        "figure.dpi": 100, "savefig.dpi": 600, "figure.constrained_layout.use": True,
    })


def figure(height_in: float, **kw):
    """Figure of full text width and the given height (inches)."""
    return plt.subplots(figsize=(TEXTWIDTH_IN, height_in), **kw)


def save(fig, out) -> None:
    """Write a 600-dpi RGB PNG (no alpha channel) at exactly the text width."""
    from PIL import Image
    png = Path(out).with_suffix(".png")
    fig.savefig(png, dpi=600, facecolor="white")
    plt.close(fig)
    im = Image.open(png)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
        im = bg
    im.save(png, dpi=(600, 600))


apply()
