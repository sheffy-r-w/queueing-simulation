"""
Slide variants of existing figures (figures/slides/): no title or subtitle, larger axis and legend
text, same colours. Drawn from results/*.csv and saved parameters. The only simulation is the
small deterministic samples the original figures also draw, with the same seeds.

  s13_labelfree_stage0_a.png / _b.png   the two panels of s13_labelfree_stage0 separately; (b) with the
                                        legend off the curves and ages > 8 shaded
  s11_baseline_4panel.png, s11_baseline_rho08.png, s11_baseline_rho098.png
                                        original ρ = 0.8 cells + the 40,000-busy-period ρ = 0.98 cells
  s07_rank_functions.png, s08_tail_ratio_b.png, s13_drift_timecourse_a.png, s13_drift_sweep.png
  s13_learning_summary.png              FCFS, label-free tail-only (3 seeds), mean excess, empirical Gittins

Run:  python -m experiments.slide_extras
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from egittins.distributions import GridDistribution, one_six_fourteen, bounded_pareto
from egittins.gittins import gittins_policy
from egittins.labelfree import knot_rank, monotone_rescale
from egittins.plotting import (use_style, INK, INK2, INK3, BLUE, ORANGE, AQUA, VIOLET, SEQ3, SEQ5, SURFACE)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "slides")
os.makedirs(OUT, exist_ok=True)
NS = [10, 100, 1000]
WINDOWS = [50, 200, 500, 2000]


def big_style():
    use_style()
    mpl.rcParams.update({"font.size": 14, "axes.labelsize": 13.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
                         "legend.fontsize": 12, "axes.titlesize": 14, "lines.linewidth": 2.2})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.12)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ stage 0

def stage0():
    F = one_six_fourteen()
    L = F.max_u + 1
    truth = gittins_policy(F, L).rank
    thetas = np.load(os.path.join(RES, "s13_labelfree_stage0_params.npz"))
    curves = pd.read_csv(os.path.join(RES, "s13_labelfree_stage0_curves.csv"))
    summ = pd.read_csv(os.path.join(RES, "s13_labelfree_stage0_summary.csv")).set_index("policy")
    cols = [AQUA, VIOLET, BLUE]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for (seed, sub), c in zip(curves.dropna(subset=["val_ratio"]).groupby("seed"), cols):
        ax.plot(sub.gen, sub.val_ratio, color=c, lw=2.0, label=f"optimizer seed {seed}")
    ax.axhline(1.0, color=INK, ls="--", lw=1.2, label="true Gittins")
    ax.axhline(summ.loc["fcfs", "ratio"], color=INK2, ls=":", lw=1.2, label="FCFS")
    ax.set_xlabel("CMA-ES generation")
    ax.set_ylabel("mean response time ÷ true Gittins")
    ax.legend(loc="upper right")
    save(fig, "s13_labelfree_stage0_a.png")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ages = np.arange(L) * F.h
    ax.axvspan(8.0, 16.0, color="#f2f1ee", zorder=0)
    ax.text(12.0, 8.2, "few scheduling contests here", ha="center", va="bottom", fontsize=11.5, color=INK3)
    ax.plot(ages, truth, color=INK, lw=2.8, label="true Gittins rank")
    for (k, th), c in zip(thetas.items(), cols):
        r = monotone_rescale(knot_rank(th, L), truth)
        ax.plot(ages, r, color=c, lw=1.4, alpha=0.9,
                label=f"CMA-ES, {k.replace('seed', 'seed ')}  (ratio {summ.loc['cmaes_' + k, 'ratio']:.3f})")
    ax.set_xlabel("age  a")
    ax.set_ylabel("rank (ordering only, rescaled onto Gittins values)")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, columnspacing=1.6)
    save(fig, "s13_labelfree_stage0_b.png")


# ----------------------------------------------------------------- baseline

def _panel(ax, sub, dist_name, rho):
    from egittins.plotting import style_box, ref_line
    sub = sub.copy()
    sub["r_emp"] = sub.empirical / sub.true_paired
    sub["r_trunc"] = sub.truncated / sub.true_paired
    pos = np.arange(len(NS))
    data_e = [sub[sub.n == n].r_emp.values for n in NS]
    data_t = [sub[sub.n == n].r_trunc.values for n in NS]
    w = 0.3
    b1 = ax.boxplot(data_e, positions=pos - w / 2 - 0.02, widths=w * 0.8, patch_artist=True, showfliers=False)
    b2 = ax.boxplot(data_t, positions=pos + w / 2 + 0.02, widths=w * 0.8, patch_artist=True, showfliers=False)
    style_box(b1, BLUE)
    style_box(b2, ORANGE)
    fcfs = sub.fcfs.iloc[0] / sub.true_gittins.iloc[0]
    plcfs = sub.plcfs.iloc[0] / sub.true_gittins.iloc[0]
    allv = np.concatenate(data_e + data_t)
    ymax = max(np.percentile(allv, 95) * 1.05, plcfs * 1.08)
    ax.axhline(1.0, color=INK, lw=1.2, ls="--", zorder=1)
    ref_line(ax, plcfs, f"PLCFS  {plcfs:.2f}", "-.")
    if fcfs < ymax:
        ref_line(ax, fcfs, f"FCFS  {fcfs:.2f}", ":", side="left")
    else:
        ax.text(0.01, 0.98, f"FCFS  {fcfs:.2f}  (off scale)", transform=ax.transAxes, fontsize=11, color=INK2,
                ha="left", va="top")
    ax.set_ylim(0.97, ymax * 1.08)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("past jobs in the sample  n")
    ax.set_ylabel("mean response time ÷ optimal")
    ax.set_title(f"{dist_name},  ρ = {rho}")
    ax.grid(axis="x", visible=False)


BASE_HANDLES = [Patch(facecolor=BLUE, alpha=0.28, edgecolor=BLUE, label="empirical Gittins"),
                Patch(facecolor=ORANGE, alpha=0.28, edgecolor=ORANGE, label="truncated empirical Gittins"),
                Line2D([], [], color=INK, ls="--", lw=1.2, label="true Gittins (optimal) = 1"),
                Line2D([], [], color=INK2, ls=":", lw=1.2, label="FCFS"),
                Line2D([], [], color=INK2, ls="-.", lw=1.2, label="PLCFS")]


def baseline():
    old = pd.read_csv(os.path.join(RES, "baseline_comparison.csv"))
    new = pd.read_csv(os.path.join(RES, "baseline_comparison_long.csv"))
    df = pd.concat([old[old.rho == 0.8], new[new.rho == 0.98]])
    dists = ["1-6-14", "bounded-Pareto"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    fig.subplots_adjust(hspace=0.42, wspace=0.24, top=0.9)
    for row, rho in enumerate((0.8, 0.98)):
        for col, d in enumerate(dists):
            _panel(axes[row, col], df[(df.dist == d) & (df.rho == rho)], d, rho)
    fig.legend(handles=BASE_HANDLES, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, columnspacing=1.6)
    save(fig, "s11_baseline_4panel.png")

    for rho, name in ((0.8, "s11_baseline_rho08.png"), (0.98, "s11_baseline_rho098.png")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        fig.subplots_adjust(wspace=0.24, top=0.82)
        for ax, d in zip(axes, dists):
            _panel(ax, df[(df.dist == d) & (df.rho == rho)], d, rho)
        fig.legend(handles=BASE_HANDLES, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, columnspacing=1.6)
        save(fig, name)


# --------------------------------------------------------- s07 / s08 (eda_tails)

def eda(seed: int = 3):
    F = bounded_pareto()
    rng = np.random.default_rng(seed)
    x = np.geomspace(2.0, 500.0, 2000)
    Ft = F.tail(x)
    emp = {n: GridDistribution.empirical(F.sample_u(rng, n), F.h) for n in NS}   # same draw order as eda_tails

    eps = 0.3
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axhspan(np.exp(-eps), np.exp(eps), color="#f2f1ee", zorder=0)
    ax.axhline(1.0, color=INK, lw=1.2)
    for n, c in zip(NS, SEQ3):
        G = emp[n]
        ax.plot(x, G.tail(x) / np.maximum(Ft, 1e-12), color=c, lw=2.0, drawstyle="steps-post", label=f"n = {n}")
        ax.axvline(G.max_u * F.h, color=c, lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 5)
    ax.set_xlim(2, 500)
    ax.set_xlabel("job size  x")
    ax.set_ylabel("empirical tail ÷ true tail")
    ax.legend(loc="upper left", title="dotted = largest sample", title_fontsize=11)
    save(fig, "s08_tail_ratio_b.png")

    L = F.max_u + 1
    true_pol = gittins_policy(F, L)
    G = GridDistribution.empirical(F.sample_u(np.random.default_rng(seed + 1), 1000), F.h)
    emp_pol = gittins_policy(G, L)
    ages = np.arange(L) * F.h
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.subplots_adjust(wspace=0.24)
    m = ages <= 500
    ax1.plot(ages[m], true_pol.rank[m], color=INK, lw=2.4)
    ax1.set_title("true Gittins rank function")
    ax1.set_xlabel("age  a")
    ax1.set_ylabel("rank  r(a)   (lower = served first)")
    m2 = ages <= G.max_u * F.h
    ax2.plot(ages[m2], emp_pol.rank[m2], color=BLUE, lw=1.1)
    ax2.set_title("empirical Gittins rank function, n = 1,000")
    ax2.set_xlabel("age  a")
    save(fig, "s07_rank_functions.png")


# ---------------------------------------------------------------- drift

def drift():
    b = pd.read_csv(os.path.join(RES, "s13_headline_blocks.csv"))
    n_busy = int(pd.read_csv(os.path.join(RES, "s13_headline.csv")).n_busy.iloc[0])
    x = b.block_start + 50
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(x, b.fcfs, color=INK2, lw=1.8, ls=":", label="FCFS")
    ax.plot(x, b.static, color=ORANGE, lw=2.2, label="static empirical Gittins (fit once, w = 500)")
    for w, c, lw in zip((50, 500, 2000), (SEQ3[0], BLUE, SEQ3[2]), (1.6, 2.2, 1.6)):
        ax.plot(x, b[f"kupd_{w}"], color=c, lw=lw, label=f"k-updating empirical Gittins, w = {w}")
    ax.axhline(1.0, color=INK, lw=1.2, ls="--", label="oracle (knows the current distribution)")
    ax.set_xlabel("busy period  k   (job-size mix drifts from mostly long to mostly short)")
    ax.set_ylabel("mean response time ÷ oracle")
    ax.set_xlim(0, n_busy)
    ax.set_ylim(0.97, 1.5)
    ax.legend(loc="upper left")
    save(fig, "s13_drift_timecourse_a.png")

    summ = pd.read_csv(os.path.join(RES, "s13_sweep_summary.csv"))
    colors = {"T=100": SEQ5[0], "T=500": SEQ5[1], "T=2000": SEQ5[2], "T=8000": SEQ5[3], "stationary": INK}
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    handles, labels = [], []
    for model, c in colors.items():
        s = summ[summ.model == model].set_index("policy")
        if len(s) == 0:
            continue
        m = [s.loc[f"kupd_{w}", "mean"] for w in WINDOWS]
        ci = [s.loc[f"kupd_{w}", "ci"] for w in WINDOWS]
        label = "no drift" if model == "stationary" else f"drift period T = {int(model[2:]):,}"
        h = ax.errorbar(WINDOWS, m, yerr=ci, color=c, marker="o", ms=7, capsize=3, lw=2.2, elinewidth=1.0,
                        ecolor=c, label=label, zorder=3)
        handles.append(h)
        labels.append(label)
        ax.plot([WINDOWS[0], WINDOWS[-1]], [s.loc["static", "mean"]] * 2, color=c, lw=1.2, ls=":", zorder=2)
    ax.axhline(1.0, color=INK, lw=1.2, ls="--", zorder=1)
    handles += [Line2D([], [], color=INK2, lw=1.2, ls=":"), Line2D([], [], color=INK, lw=1.2, ls="--")]
    labels += ["dotted: static fit (w = 500), same colour", "oracle (knows the current distribution)"]
    ax.set_xscale("log")
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([str(w) for w in WINDOWS])
    ax.set_xlabel("refit window  w   (most recent completed jobs)")
    ax.set_ylabel("mean response time ÷ oracle")
    ax.legend(handles, labels, loc="upper right")
    save(fig, "s13_drift_sweep.png")


# ------------------------------------------------------------- learning slide

def learning_summary():
    summ = pd.read_csv(os.path.join(RES, "s13_labelfree_static_summary.csv"))
    rows = [("empirical_gittins", "exact empirical Gittins", BLUE),
            ("mean_excess", "rank by sample mean excess", AQUA),
            ("labelfree_tail_only|linear|seed0", "label-free search, tail-only features (3 seeds)", ORANGE),
            ("fcfs", "FCFS", INK2)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    y = np.arange(len(rows))[::-1]
    for ax, dist in zip(axes, ("1-6-14", "bounded-Pareto")):
        s = summ[summ.dist == dist].set_index("policy")
        xmax = 1.12
        for yi, (key, label, c) in zip(y, rows):
            if key.startswith("labelfree"):
                for seed in range(3):
                    r = s.loc[f"labelfree_tail_only|linear|seed{seed}"]
                    ax.errorbar(r.ratio, yi + (seed - 1) * 0.18, xerr=r.ci, fmt="o", color=c, ecolor=c, ms=7,
                                capsize=3, elinewidth=1.0, zorder=3)
            else:
                r = s.loc[key]
                if r.ratio > xmax:
                    ax.annotate("", (xmax, yi), xytext=(xmax - 0.015, yi),
                                arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
                    ax.text(xmax - 0.02, yi, f"FCFS  {r.ratio:.2f}  (off scale)", ha="right", va="center",
                            fontsize=12, color=c)
                else:
                    ax.errorbar(r.ratio, yi, xerr=r.ci, fmt="o", color=c, ecolor=c, ms=8, capsize=3,
                                elinewidth=1.0, zorder=3)
        ax.axvline(1.0, color=INK, ls="--", lw=1.2)
        ax.set_xlim(0.99, xmax)
        ax.set_title(f"{dist} (held out)")
        ax.set_xlabel("mean response time ÷ optimal   (paired, 95% CI)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[1] for r in rows])
    save(fig, "s13_learning_summary.png")


if __name__ == "__main__":
    big_style()
    stage0()
    baseline()
    eda()
    drift()
    learning_summary()
