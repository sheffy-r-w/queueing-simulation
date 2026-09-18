"""
Conceptual figures for the early slides (no simulation).

  s02_spectrum.png            the spectrum of information about job sizes and the policy
                              that is optimal at each point
  s04_gittins_intuition.png   the 1-6-14 density next to its Gittins rank function, annotated
  s04_gittins_empirical.png   the same from 100 past job sizes, with the true curves behind

Run:  python -m experiments.slide_figures
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from egittins.distributions import one_six_fourteen
from egittins.gittins import gittins_policy
from egittins.plotting import use_style, savefig, INK, INK2, INK3, BLUE, ORANGE, AQUA, VIOLET, GRID, AXIS, SLIDES

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

DENSITY_YMAX = 0.5      # shared y-range for the s04 density panels (true and empirical)


def spectrum():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.add_patch(FancyArrowPatch((0.3, 0.55), (9.7, 0.55), arrowstyle="-|>", mutation_scale=18,
                                 color=INK2, lw=1.4))
    ax.text(5.0, 0.18, "less information about job sizes", ha="center", va="center", fontsize=10, color=INK2)
    boxes = [
        (1.35, "exact sizes", "SRPT\n(shortest remaining\nprocessing time)", "optimal; needs the\nsize of every job", INK2),
        (3.8, "the distribution", "Gittins\n(rank function from\nthe size distribution)", "optimal among policies\nthat only see ages", INK2),
        (6.25, "a sample of sizes", "empirical Gittins\n(this work)", "plug in the empirical\ndistribution — how good is it?", BLUE),
        (8.7, "nothing", "FCFS / PS", "no size information\nused at all", INK2),
    ]
    for x, know, pol, note, c in boxes:
        ax.add_patch(FancyBboxPatch((x - 1.05, 1.15), 2.1, 1.45, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor="white", edgecolor=c if c == BLUE else AXIS, lw=2.2 if c == BLUE else 1.2))
        ax.text(x, 2.85, know, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x, 1.88, pol, ha="center", va="center", fontsize=9.5, color=INK)
        ax.text(x, 0.92, note, ha="center", va="top", fontsize=8, color=INK2)
    ax.text(5.0, 3.12, "What do you know about job sizes, and what is the best schedule?",
            ha="center", va="center", fontsize=11, color=INK)
    savefig(fig, os.path.join(FIG, "s02_spectrum.png"))


def gittins_intuition():
    F = one_six_fourteen()
    L = F.max_u + 1
    pol = gittins_policy(F, L)
    x = F.atoms
    ages = np.arange(L) * F.h
    m = np.arange(L) < F.max_u

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.6]))
    fig.subplots_adjust(hspace=0.12)
    ax1.fill_between(x, F.probs / F.h, color=BLUE, alpha=0.25, lw=0)
    ax1.plot(x, F.probs / F.h, color=BLUE, lw=1.2)
    ax1.set_ylim(0, DENSITY_YMAX)
    ax1.set_ylabel("density of job size")
    if not SLIDES:      # slides: no text overlays, the annotations are added on the slide
        ax1.set_title("1-6-14: equal-weight mixture of normals (means 1, 6, 14; sd 0.5), truncated to (0, 16]")
        for mu in (1, 6, 14):
            ax1.text(mu, ax1.get_ylim()[1] * 0.75, f"{mu}", ha="center", fontsize=10, color=INK)

    ax2.plot(ages[m], pol.rank[m], color=INK, lw=2.0)
    ax2.set_ylabel("Gittins rank  r(a)\n(lower = served first)")
    ax2.set_xlabel("age  a  (service the job has already received)")
    ax2.set_xlim(0, 16)
    notes = [   # (point on the curve, text position)
        ((0.05, 4.8), (0.15, 9.35), "age 0: could be a 1, so try it"),
        ((1.0, 3.1), (1.6, 1.5), "almost 1 and not done yet?\nabout to finish: top priority"),
        ((2.1, 7.9), (4.0, 9.2), "past 2: it is a 6 or a 14;\nlong way to go, so yield to new arrivals"),
        ((6.0, 1.85), (6.4, 0.5), "approaching 6: finish it"),
        ((7.6, 6.4), (8.4, 7.9), "past 7: it is a 14; rank is now just\nthe remaining work, an SRPT-like descent"),
    ]
    for pt, xy, txt in ([] if SLIDES else notes):
        ax2.annotate(txt, pt, xytext=xy, textcoords="data", fontsize=8.5, ha="left", va="center",
                     color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax2.set_ylim(0, 10)
    savefig(fig, os.path.join(FIG, "s04_gittins_intuition.png"))


def gittins_empirical(n=100, seed=3):
    """The same picture built from n past job sizes, styled exactly like s04_gittins_intuition:
    histogram of the samples (blue fill) with the true density as a thin grey line; the empirical
    Gittins rank function as a solid black line with the true rank behind it as a thin grey
    dashed line. Same axis limits, so the two slides line up."""
    F = one_six_fourteen()
    rng = np.random.default_rng(seed)
    samples_u = F.sample_u(rng, n)
    G = F.__class__.empirical(samples_u, F.h, name=f"empirical n={n}")
    L = F.max_u + 1
    pol = gittins_policy(G, L)
    true_pol = gittins_policy(F, L)
    ages = np.arange(L) * F.h
    m = np.arange(L) < G.max_u
    mt = np.arange(L) < F.max_u
    grey = INK3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.6]))
    fig.subplots_adjust(hspace=0.12)
    bins = np.arange(0, 16.25, 0.25)
    ax1.hist(samples_u * F.h, bins=bins, density=True, color=BLUE, alpha=0.25, lw=0)
    ax1.hist(samples_u * F.h, bins=bins, density=True, histtype="step", color=BLUE, lw=1.2)
    ax1.plot(F.atoms, F.probs / F.h, color=grey, lw=1.0, label="true")
    ax1.set_ylim(0, DENSITY_YMAX)
    ax1.set_ylabel("density of job size")
    if not SLIDES:
        ax1.set_title(f"the last {n} job sizes, drawn from 1-6-14")

    ax2.plot(ages[mt], true_pol.rank[mt], color=grey, lw=1.0, ls="--", label="true")
    ax2.plot(ages[m], pol.rank[m], color=INK, lw=2.0, label=f"from {n} samples")
    ax2.set_ylabel("Gittins rank  r(a)\n(lower = served first)")
    ax2.set_xlabel("age  a  (service the job has already received)")
    ax2.set_xlim(0, 16)
    ax2.set_ylim(0, 10)
    ax2.legend(loc="upper right", fontsize=9)
    savefig(fig, os.path.join(FIG, "s04_gittins_empirical.png"))


if __name__ == "__main__":
    use_style()
    if not SLIDES:      # the spectrum diagram is drawn on the slide itself
        spectrum()
    gittins_intuition()
    gittins_empirical()
