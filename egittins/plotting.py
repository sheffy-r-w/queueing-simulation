"""Shared matplotlib style for the presentation figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical palette (validated for colour-vision deficiency, fixed order).
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# Sequential blue ramp, light -> dark, for n = 10, 100, 1000.
SEQ3 = ["#9cc3ee", "#4a8fdf", "#1a4f93"]


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "text.color": INK,
        "axes.labelcolor": INK,
    })


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)
