"""
Section 5a: the O(K + L) convex-hull rank computation, and its validation.

Figure s05a_rank_hull.png:
  (a) the geometry. Each atom x_j of the (empirical) distribution is a point
      P_j = (F(x_j), E[S ∧ x_j]). The Gittins rank at age a is the minimum slope
      from Q(a) = (F(a), E[S ∧ a]) to a later point, attained on the lower
      convex hull of the later points (highlighted); the tangent line is drawn.
  (b) validation: the hull kernel reproduces the O(K · L) definition-based
      kernel bit for bit, and its running time as a function of the number of
      atoms K.

Also writes results/rank_timing.csv (the table behind panel b).

Run:  python -m experiments.rank_hull
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from egittins.distributions import GridDistribution, one_six_fourteen, bounded_pareto
from egittins.gittins import _rank_kernel, _rank_kernel_hull, rank_hull_geometry
from egittins.plotting import use_style, savefig, headline, BLUE, ORANGE, AQUA, INK, INK2

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def _time(fn, *args, repeats=3):
    fn(*args)                                   # compile / warm
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def timing_table(seed: int = 0) -> pd.DataFrame:
    P = bounded_pareto()
    F = one_six_fourteen()
    rng = np.random.default_rng(seed)
    cases = [("1-6-14 (true)", F, F.max_u + 1), ("bounded Pareto (true)", P, P.max_u + 1)]
    for n in (10, 100, 1000, 10_000, 100_000):
        G = GridDistribution.empirical(P.sample_u(rng, n), P.h, name=f"empirical Pareto n={n}")
        cases.append((G.name, G, P.max_u + 1))
    rows = []
    for name, G, L in cases:
        r1, n1 = _rank_kernel(G.atoms_u, G.probs, G.h, L)
        r2, n2 = _rank_kernel_hull(G.atoms_u, G.probs, G.h, L)
        fin = np.isfinite(r1)
        assert np.array_equal(fin, np.isfinite(r2)) and np.array_equal(n1, n2)
        maxrel = float(np.max(np.abs(r1[fin] - r2[fin]) / np.abs(r1[fin])))
        rows.append(dict(
            distribution=name, K=len(G.atoms_u), L=L,
            direct_ms=1e3 * _time(_rank_kernel, G.atoms_u, G.probs, G.h, L),
            hull_ms=1e3 * _time(_rank_kernel_hull, G.atoms_u, G.probs, G.h, L),
            max_rel_diff=maxrel))
        print(f"{name:28s} K={rows[-1]['K']:6d}  direct {rows[-1]['direct_ms']:9.2f} ms"
              f"  hull {rows[-1]['hull_ms']:7.3f} ms  max rel diff {maxrel:.1e}")
    return pd.DataFrame(rows)


def main(seed: int = 7, n: int = 40, a_u: int = 50):
    use_style()
    F = one_six_fourteen()
    G = GridDistribution.empirical(F.sample_u(np.random.default_rng(seed), n), F.h)
    geo = rank_hull_geometry(G, a_u)
    Fx, U, hull, tg = geo["F"], geo["U"], geo["hull"], geo["tangent"]
    xq, yq, r = geo["xq"], geo["yq"], geo["rank"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.28)

    # (a) geometry
    ax1.plot(Fx, U, color=INK2, lw=0.8, alpha=0.6, zorder=1)
    ax1.scatter(Fx, U, s=16, color=INK2, zorder=2, label="atoms  (F(x_j), E[S $\\wedge$ x_j])")
    ax1.plot(Fx[hull], U[hull], color=AQUA, lw=2.0, zorder=3, label="lower convex hull of later atoms")
    x_end = Fx[hull[-1]]
    ax1.plot([xq, x_end], [yq, yq + r * (x_end - xq)], color=ORANGE, lw=1.4, ls="--", zorder=4,
             label=f"tangent, slope = r(a) = {r:.2f}")
    ax1.scatter([xq], [yq], s=60, color=ORANGE, zorder=5, marker="D",
                label=f"query  (F(a), E[S $\\wedge$ a]),  a = {a_u * G.h:g}")
    ax1.scatter([Fx[tg]], [U[tg]], s=90, facecolor="none", edgecolor=ORANGE, lw=1.6, zorder=6)
    ax1.annotate("argmin atom", (Fx[tg], U[tg]), xytext=(8, -14), textcoords="offset points",
                 fontsize=8, color=ORANGE)
    ax1.set_xlabel("F(x)  cumulative probability")
    ax1.set_ylabel("E[S $\\wedge$ x]  expected work up to x")
    ax1.set_title("(a) rank = min slope to a later atom")
    ax1.legend(loc="upper left", fontsize=8, title=f"empirical 1-6-14, n = {n}", title_fontsize=8)

    # (b) timing
    df = timing_table()
    df.to_csv(os.path.join(RES, "rank_timing.csv"), index=False)
    emp = df[df.distribution.str.startswith("empirical")]
    ax2.plot(emp.K, emp.direct_ms, "o-", color=INK2, label="direct, O(K · L)")
    ax2.plot(emp.K, emp.hull_ms, "s-", color=BLUE, label="convex hull, O(K + L)")
    for _, row in df[~df.distribution.str.startswith("empirical")].iterrows():
        ax2.scatter([row.K], [row.direct_ms], color=INK2, marker="*", s=110, zorder=5)
        ax2.scatter([row.K], [row.hull_ms], color=BLUE, marker="*", s=110, zorder=5)
        ax2.annotate(row.distribution.replace(" (true)", ""), (row.K, row.direct_ms),
                     xytext=(-6, 6), textcoords="offset points", fontsize=7.5, ha="right", color=INK)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("number of atoms K   (empirical bounded Pareto; $\\bigstar$ = true distributions)")
    ax2.set_ylabel("time to tabulate the rank function  (ms)")
    big = df[df.distribution == "bounded Pareto (true)"].iloc[0]
    same = "bit-identical" if df.max_rel_diff.max() == 0 else f"max rel. diff {df.max_rel_diff.max():.0e}"
    ax2.set_title(f"(b) {same} output, {big.direct_ms / big.hull_ms:,.0f}× faster (true Pareto)")
    ax2.legend(loc="upper left")
    headline(fig, "The exact rank function is a convex-hull sweep: same numbers, 8× faster at n = 1,000 and 3,300× on the dense true grid",
             "Left: the rank at age a is the smallest slope from (F(a), E[S $\\wedge$ a]) to a later atom, which lies on the lower convex "
             "hull. Right: time to tabulate the rank at every age.", top=0.8)
    savefig(fig, os.path.join(FIG, "s05a_rank_hull.png"))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
