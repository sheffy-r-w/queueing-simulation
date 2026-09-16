"""Shared matplotlib style for the presentation figures.

One look for every figure in the deck: a clean sans, recessive hairline axes and
grid, thin marks, a left-aligned headline that states the finding with a muted
subtitle that states the setup, and one colour per policy everywhere:

    INK  solid   true Gittins rank;   INK dashed   optimal / genie (ratio 1.0)
    BLUE         empirical Gittins (exact), including k-updating; SEQ ramp for n or w
    ORANGE       a degraded variant: truncated empirical Gittins, or a static fit
    INK2         FCFS (dotted) and PLCFS (dash-dot)
    VIOLET       learned (NN) rank function;   AQUA   RL from scratch / other variants

The categorical slots are the validated reference palette (blue, orange, aqua,
violet: CVD-safe in that order); INK/INK2 are text tokens, never series colours.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# EGITTINS_SLIDES=1: no baked-in headline/subtitle (the slide title carries it) and
# every figure is written to a slides/ subfolder next to its usual path.
SLIDES = os.environ.get("EGITTINS_SLIDES", "0") == "1"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID, AXIS = "#ffffff", "#ebeae6", "#d9d8d3"
# Sequential blue ramps, light -> dark (n = 10, 100, 1000; or five windows / periods).
SEQ3 = ["#9cc3ee", "#4a8fdf", "#1a4f93"]
SEQ5 = ["#b7d3f2", "#7fb0e6", "#3f88d8", "#1f5aa8", "#123a6e"]


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        # a family list gives per-glyph fallback (arrows, ∧, ★ are missing from some faces)
        "font.family": ["Avenir Next", "Avenir", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.titlecolor": INK,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "xtick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.handlelength": 2.4,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.2,
        "lines.markeredgecolor": SURFACE,
        "text.color": INK,
    })


def headline(fig, title, subtitle=None, top=0.84, x=0.01):
    """Left-aligned finding above the figure, with the setup in a muted subtitle.

    A no-op under EGITTINS_SLIDES=1, where the slide title plays this role.
    """
    if SLIDES:
        return
    fig.subplots_adjust(top=top)
    fig.text(x, 0.995, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, 0.945, subtitle, ha="left", va="top", fontsize=9.2, color=INK2)


def style_box(bp, color):
    """Thin boxes: series-coloured edge, light fill, ink median, no caps."""
    for patch in bp["boxes"]:
        patch.set(facecolor=color, alpha=0.28, edgecolor=color, linewidth=1.0)
    for line in bp["whiskers"]:
        line.set(color=color, linewidth=1.0)
    for line in bp["caps"]:
        line.set(visible=False)
    for line in bp["medians"]:
        line.set(color=INK, linewidth=1.4)


def ref_line(ax, y, label, style, color=INK2, side="right", dy=0.0):
    """A horizontal reference (FCFS, PLCFS, optimal) with a small muted label at one end."""
    h = ax.axhline(y, color=color, lw=1.0, ls=style, zorder=1)
    x = 0.995 if side == "right" else 0.005
    ax.text(x, y + dy, label, transform=ax.get_yaxis_transform(), fontsize=8, color=color,
            ha="right" if side == "right" else "left", va="bottom")
    return h


def savefig(fig, path):
    if SLIDES:
        d, name = os.path.split(path)
        d = os.path.join(d, "slides")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.12)
    plt.close(fig)
    print("wrote", path)
