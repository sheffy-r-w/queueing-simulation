"""
EDA: how close is the empirical distribution to the truth, and *where*?

Figure 1 (s08_tail_ratio.png), two panels on the bounded Pareto distribution:
  (a) the true tail F̄(x) and empirical tails Ḡ(x) for n = 10, 100, 1000 samples
  (b) the ratio Ḡ(x) / F̄(x). It hugs 1 up to a threshold that grows with n, then
      breaks down: the empirical distribution has finite support, so its tail
      hits zero at the largest sample while the true tail keeps going.

This is the data-first version of Lemma 2.3 (ε-multiplicative closeness holds
only up to a threshold ℓ) and the reason the paper's analysis is about *tails*.

Figure 2 (s07_rank_functions.png): the Gittins rank functions of true Gittins and
empirical Gittins (1000 samples) — a reproduction of the paper's Fig. 1.1 —
showing that the two policies look nothing alike even though their tails are
close. This is the obstacle the paper's two new WINE identities get around.

Run:  python -m experiments.eda_tails
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from egittins.distributions import GridDistribution, bounded_pareto
from egittins.gittins import gittins_policy
from egittins.plotting import use_style, savefig, SEQ3, INK, ORANGE, BLUE

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def main(seed: int = 3):
    use_style()
    F = bounded_pareto()
    rng = np.random.default_rng(seed)
    ns = [10, 100, 1000]
    x = np.geomspace(2.0, 500.0, 2000)
    Ft = F.tail(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # (a) tails
    ax1.plot(x, Ft, color=INK, lw=2.0, label="true  F̄(x)")
    emp = {}
    for n, c in zip(ns, SEQ3):
        G = GridDistribution.empirical(F.sample_u(rng, n), F.h)
        emp[n] = G
        ax1.plot(x, np.maximum(G.tail(x), 1e-6), color=c, lw=1.4, drawstyle="steps-post",
                 label=f"empirical, n = {n}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-4, 1.5)
    ax1.set_xlim(2, 500)
    ax1.set_xlabel("job size  x")
    ax1.set_ylabel("tail  P(S > x)")
    ax1.set_title("(a) empirical vs. true tail (bounded Pareto)")
    ax1.legend(loc="lower left")

    # (b) ratio
    eps = 0.3
    ax2.axhspan(np.exp(-eps), np.exp(eps), color="#f2f1ee", zorder=0)
    ax2.axhline(1.0, color=INK, lw=1.0)
    for n, c in zip(ns, SEQ3):
        G = emp[n]
        ratio = G.tail(x) / np.maximum(Ft, 1e-12)
        ax2.plot(x, ratio, color=c, lw=1.4, drawstyle="steps-post", label=f"n = {n}")
        ax2.axvline(G.max_u * F.h, color=c, lw=0.8, ls=":")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(0.1, 5)
    ax2.set_xlim(2, 500)
    ax2.set_xlabel("job size  x")
    ax2.set_ylabel("ratio  Ḡ(x) / F̄(x)")
    ax2.set_title(f"(b) closeness holds up to a threshold (band: e^±{eps})")
    ax2.legend(loc="upper left", title="dotted = largest sample", title_fontsize=8)

    savefig(fig, os.path.join(OUT, "s08_tail_ratio.png"))

    # ---- Figure 2: rank functions (Fig 1.1 reproduction)
    L = F.max_u + 1
    true_pol = gittins_policy(F, L)
    G = GridDistribution.empirical(F.sample_u(np.random.default_rng(seed + 1), 1000), F.h)
    emp_pol = gittins_policy(G, L)
    ages = np.arange(L) * F.h

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=False)
    m = ages <= 500
    ax1.plot(ages[m], true_pol.rank[m], color=BLUE, lw=1.4)
    ax1.set_title("(a) true Gittins rank function")
    ax1.set_xlabel("age  a")
    ax1.set_ylabel("rank  r(a)   (lower = higher priority)")
    m2 = ages <= G.max_u * F.h
    ax2.plot(ages[m2], emp_pol.rank[m2], color=ORANGE, lw=0.9)
    ax2.set_title("(b) empirical Gittins rank function (n = 1000)")
    ax2.set_xlabel("age  a")
    savefig(fig, os.path.join(OUT, "s07_rank_functions.png"))


if __name__ == "__main__":
    main()
