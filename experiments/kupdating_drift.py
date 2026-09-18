"""
Section 13: static vs. k-updating vs. genie under distribution drift.

Three experiments, all on the k-updating engine (`egittins/kupdating.py`) with
the drift models of `egittins/drift.py`; every policy sees the same arrival
stream (common random numbers) and results are paired ratios to the genie
(true Gittins for the current distribution) with 95% t confidence intervals
over independent trials.

  headline  one-way weight drift on 1-6-14, (0.1,0.3,0.6) → (0.6,0.3,0.1) over
            the run (variability increases), plus a stationary control at the
            final weights (0.6,0.3,0.1), where Gittins gains most.
            -> results/s13_headline.csv, results/s13_headline_summary.csv,
               figures/s13_drift_timecourse.png
  sweep     mean-preserving weight drift (triangle wave; load pinned) with
            drift period T ∈ {100, 500, 2000, 8000} × window w ∈ {50, 200, 500, 2000}.
            -> results/s13_sweep.csv, figures/s13_drift_sweep.png
  pareto    bounded-Pareto tail drift α: 2.0 → 1.2 over the run, plus a
            stationary control at α = 2.0.
            -> results/s13_pareto.csv, results/s13_pareto_summary.csv

Run:  python -m experiments.kupdating_drift headline|sweep|pareto|all [--quick]
      python -m experiments.kupdating_drift --plot          # re-plot from CSVs
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from egittins.drift import make_drift
from egittins.kupdating import run_stream, summarize, blocks, StreamResult, DEFAULT_WINDOWS

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

RHO = 0.8
WINDOWS = DEFAULT_WINDOWS
STATIC_WINDOW = 500
SWEEP_PERIODS = [100, 500, 2000, 8000]
N_WARM = 600
BLOCK = 100


def _job(args):
    model_key, family, schedule, kw, seed, n_busy, n_warm = args
    d = make_drift(family, schedule, **kw)
    res = run_stream(d, RHO, n_busy, n_warm, seed, windows=WINDOWS, static_window=STATIC_WINDOW)
    return model_key, res


def _run_models(models, trials, n_busy, n_warm, workers):
    """models: {key: (family, schedule, kwargs)}. Returns {key: [StreamResult]}."""
    jobs = [(key, fam, sch, kw, s, n_busy, n_warm)
            for key, (fam, sch, kw) in models.items() for s in range(trials)]
    out = {key: [] for key in models}
    with Pool(workers) as pool:
        for i, (key, res) in enumerate(pool.imap_unordered(_job, jobs)):
            out[key].append(res)
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"  {i + 1}/{len(jobs)} runs done", flush=True)
    return out


def _tidy(results: dict[str, list[StreamResult]], **extra) -> pd.DataFrame:
    rows = []
    for key, lst in results.items():
        for res in lst:
            for name in res.policies:
                rows.append(dict(model=key, seed=res.seed, policy=name, mrt=res.mrt(name),
                                 ratio=res.ratio(name), n_jobs=int(res.n_jobs.sum()), **extra))
    return pd.DataFrame(rows)


def _summary(results: dict[str, list[StreamResult]]) -> pd.DataFrame:
    rows = []
    for key, lst in results.items():
        for r in summarize(lst):
            rows.append(dict(model=key, **r))
    return pd.DataFrame(rows)


def _print_table(summary: pd.DataFrame, title: str):
    print(f"\n{title}  (mean response time / genie, 95% CI over trials)")
    piv = summary.pivot(index="policy", columns="model", values=["mean", "ci"])
    order = [p for p in ["fcfs", "static"] + [f"kupd_{w}" for w in WINDOWS] if p in piv.index]
    cols = list(summary.model.unique())
    print("| policy | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for p in order:
        cells = [f"{piv.loc[p, ('mean', c)]:.3f} ± {piv.loc[p, ('ci', c)]:.3f}" for c in cols]
        print(f"| {p} | " + " | ".join(cells) + " |")


# ------------------------------------------------------------------ headline

def run_headline(trials, n_busy, workers):
    models = {
        "one-way drift": ("one-way", "ramp", dict(N=n_busy)),
        "stationary": ("one-way", "constant", dict(u0=1.0)),
    }
    print(f"headline: {trials} trials × {n_busy} busy periods (+{N_WARM} warm-up), ρ = {RHO}")
    results = _run_models(models, trials, n_busy, N_WARM, workers)
    _tidy(results, n_busy=n_busy).to_csv(os.path.join(RES, "s13_headline.csv"), index=False)
    summ = _summary(results)
    summ.to_csv(os.path.join(RES, "s13_headline_summary.csv"), index=False)
    b = blocks(results["one-way drift"], BLOCK)
    pd.DataFrame({"block_start": np.arange(len(b["genie"])) * BLOCK, **b}).to_csv(
        os.path.join(RES, "s13_headline_blocks.csv"), index=False)
    _print_table(summ, "Headline: one-way weight drift on 1-6-14")


def plot_headline():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from egittins.plotting import use_style, savefig, headline, INK, INK2, INK3, BLUE, ORANGE, SEQ3
    use_style()
    b = pd.read_csv(os.path.join(RES, "s13_headline_blocks.csv"))
    summ = pd.read_csv(os.path.join(RES, "s13_headline_summary.csv"))
    n_busy = int(pd.read_csv(os.path.join(RES, "s13_headline.csv")).n_busy.iloc[0])
    ntr = int(summ.n.iloc[0])
    x = b.block_start + BLOCK / 2

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.75, 1]))
    fig.subplots_adjust(wspace=0.24)
    ax.plot(x, b.fcfs, color=INK2, lw=1.6, ls=":", label="first-come first-served (FCFS)")
    ax.plot(x, b.static, color=ORANGE, lw=2.0, label=f"static empirical Gittins (fit once, w = {STATIC_WINDOW})")
    for w, c, lw in zip((50, 500, 2000), (SEQ3[0], BLUE, SEQ3[2]), (1.4, 2.0, 1.4)):
        ax.plot(x, b[f"kupd_{w}"], color=c, lw=lw, label=f"k-updating empirical Gittins, w = {w}")
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", label="genie: true Gittins for the current distribution")
    ax.set_xlabel("busy period  k    (job-size mix drifts linearly from mostly long to mostly short)")
    ax.set_ylabel("mean response time ÷ genie   (blocks of 100 busy periods)")
    ax.set_title("(a) over the run: the stale fit degrades like FCFS, refitting tracks the genie")
    ax.set_xlim(0, n_busy)
    ax.set_ylim(0.97, 1.5)
    ax.legend(loc="upper left")

    order = ["fcfs", "static"] + [f"kupd_{w}" for w in WINDOWS]
    labels = ["FCFS", f"static\n(w = {STATIC_WINDOW})"] + [f"k-upd\nw = {w}" for w in WINDOWS]
    colors = [INK2, ORANGE] + [BLUE] * len(WINDOWS)
    pos = np.arange(len(order))
    for off, (model, alpha) in enumerate([("stationary", 0.35), ("one-way drift", 1.0)]):
        s = summ[summ.model == model].set_index("policy").loc[order]
        ax2.bar(pos + (off - 0.5) * 0.36, s["mean"] - 1, 0.33, bottom=1, yerr=s["ci"], color=colors,
                alpha=alpha, capsize=2, error_kw=dict(ecolor=INK2, lw=0.9))
    ax2.axhline(1.0, color=INK, lw=1.0, ls="--")
    ax2.set_xticks(pos)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("mean response time ÷ genie   (whole run, 95% CI)")
    ax2.set_title("(b) whole run, with and without drift")
    ax2.grid(axis="x", visible=False)
    ax2.legend(handles=[Patch(facecolor=INK3, alpha=0.35, label="no drift"), Patch(facecolor=INK3, label="with drift")],
               loc="upper right")
    headline(fig, "A policy fitted once degrades as the workload drifts; refitting from a rolling window keeps up",
             f"1-6-14 job sizes with drifting mode weights, load ρ = {RHO}; {ntr} trials × {n_busy:,} busy periods, every "
             "policy on the same arrival stream. Genie = 1.", top=0.8)
    savefig(fig, os.path.join(FIG, "s13_drift_timecourse.png"))


# --------------------------------------------------------------------- sweep

def run_sweep(trials, n_busy, workers):
    models = {f"T={T}": ("mean-preserving", "triangle", dict(T=T)) for T in SWEEP_PERIODS}
    models["stationary"] = ("mean-preserving", "constant", dict(u0=0.5))
    print(f"sweep: {trials} trials × {n_busy} busy periods per (T, w) cell, ρ = {RHO}")
    results = _run_models(models, trials, n_busy, N_WARM, workers)
    _tidy(results, n_busy=n_busy).to_csv(os.path.join(RES, "s13_sweep.csv"), index=False)
    summ = _summary(results)
    summ.to_csv(os.path.join(RES, "s13_sweep_summary.csv"), index=False)
    _print_table(summ, "Sweep: mean-preserving triangle-wave drift, period T")


def plot_sweep():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from egittins.plotting import use_style, savefig, headline, INK, INK2, SEQ5
    use_style()
    summ = pd.read_csv(os.path.join(RES, "s13_sweep_summary.csv"))
    n_busy = int(pd.read_csv(os.path.join(RES, "s13_sweep.csv")).n_busy.iloc[0])
    ntr = int(summ.n.iloc[0])
    # every curve is k-updating empirical Gittins: lighter blue = faster drift, black = no drift
    colors = {"T=100": SEQ5[0], "T=500": SEQ5[1], "T=2000": SEQ5[2], "T=8000": SEQ5[3], "stationary": INK}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    handles, labels = [], []
    for model, c in colors.items():
        s = summ[summ.model == model].set_index("policy")
        if len(s) == 0:
            continue
        m = [s.loc[f"kupd_{w}", "mean"] for w in WINDOWS]
        ci = [s.loc[f"kupd_{w}", "ci"] for w in WINDOWS]
        label = "no drift" if model == "stationary" else f"drift period T = {int(model[2:]):,} busy periods"
        h = ax.errorbar(WINDOWS, m, yerr=ci, color=c, marker="o", ms=6, capsize=2.5, lw=2.0, elinewidth=0.9,
                        ecolor=c, label=label, zorder=3)
        handles.append(h); labels.append(label)
        ax.plot([WINDOWS[0], WINDOWS[-1]], [s.loc["static", "mean"]] * 2, color=c, lw=1.1, ls=":", zorder=2)
    ax.axhline(1.0, color=INK, lw=1, ls="--", zorder=1)
    handles += [Line2D([], [], color=INK2, lw=1.1, ls=":"), Line2D([], [], color=INK, lw=1, ls="--")]
    labels += [f"dotted: static fit (w = {STATIC_WINDOW}), same colour", "genie (true Gittins for the current distribution)"]
    ax.set_xscale("log")
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([str(w) for w in WINDOWS])
    ax.set_xlabel("refit window  w   (most recent completed jobs)")
    ax.set_ylabel("mean response time ÷ genie   (95% CI)")
    ax.legend(handles, labels, loc="upper right")
    headline(fig, "The refit window is a retraining cadence: too short is noise; the longest window lags only at T = 500",
             f"k-updating empirical Gittins under mean-preserving 1-6-14 weight drift with period T, load ρ = {RHO}; "
             f"{ntr} trials × {n_busy:,} busy periods per point.", top=0.84)
    savefig(fig, os.path.join(FIG, "s13_drift_sweep.png"))


# -------------------------------------------------------------------- pareto

def run_pareto(trials, n_busy, workers):
    models = {
        "tail drift α 2.0→1.2": ("pareto-tail", "ramp", dict(N=n_busy)),
        "stationary α=2.0": ("pareto-tail", "constant", dict(u0=0.0)),
    }
    print(f"pareto: {trials} trials × {n_busy} busy periods (+{N_WARM} warm-up), ρ = {RHO}")
    results = _run_models(models, trials, n_busy, N_WARM, workers)
    _tidy(results, n_busy=n_busy).to_csv(os.path.join(RES, "s13_pareto.csv"), index=False)
    summ = _summary(results)
    summ.to_csv(os.path.join(RES, "s13_pareto_summary.csv"), index=False)
    b = blocks(results["tail drift α 2.0→1.2"], BLOCK)
    pd.DataFrame({"block_start": np.arange(len(b["genie"])) * BLOCK, **b}).to_csv(
        os.path.join(RES, "s13_pareto_blocks.csv"), index=False)
    _print_table(summ, "Pareto: bounded-Pareto tail drift")


# --------------------------------------------------------------- drift types

TYPES = [  # (label, summary csv, model key)
    ("mode weights shift\nfrom (0.1, 0.3, 0.6) to (0.6, 0.3, 0.1)", "s13_headline_summary.csv", "one-way drift"),
    ("shape drift, load pinned\n(mean-preserving, period T = 2000)", "s13_sweep_summary.csv", "T=2000"),
    ("tail gets heavier\n(Pareto α from 2.0 to 1.2)", "s13_pareto_summary.csv", "tail drift α 2.0→1.2"),
]
TYPE_POLICIES = [("fcfs", "first-come first-served (FCFS)"), ("static", "static empirical Gittins (fit once, w = 500)"),
                 ("kupd_500", "k-updating empirical Gittins (w = 500)")]


def types_table() -> pd.DataFrame:
    rows = []
    for label, csv, model in TYPES:
        s = pd.read_csv(os.path.join(RES, csv))
        s = s[s.model == model].set_index("policy")
        for pol, _ in TYPE_POLICIES:
            rows.append(dict(drift=label.replace("\n", " "), policy=pol, mean=s.loc[pol, "mean"],
                             ci=s.loc[pol, "ci"], n=int(s.loc[pol, "n"])))
    return pd.DataFrame(rows)


def plot_types():
    """One bar chart: three kinds of drift × {FCFS, static, k-updating w = 500}, ratio to the genie."""
    import matplotlib.pyplot as plt
    from egittins.plotting import use_style, savefig, headline, INK, INK2, ORANGE, BLUE
    use_style()
    t = types_table()
    t.to_csv(os.path.join(RES, "s13_drift_types.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    pos = np.arange(len(TYPES))
    w = 0.24
    ymax = 1.45
    for j, ((pol, name), c) in enumerate(zip(TYPE_POLICIES, (INK2, ORANGE, BLUE))):
        s = t[t.policy == pol].reset_index(drop=True)
        x = pos + (j - 1) * (w + 0.02)
        m = s["mean"].values
        clipped = m > ymax
        ax.bar(x, np.where(clipped, ymax - 1, m - 1), w, bottom=1, yerr=np.where(clipped, 0, s.ci.values),
               color=c, capsize=2, error_kw=dict(ecolor=INK2, lw=0.9), label=name, zorder=3)
        for xi, mi, ci_ in zip(x, m, s.ci.values):
            txt = f"{mi:.2f}  (off scale)" if mi > ymax else f"{mi:.3f}"
            ax.text(xi, min(mi, ymax - 0.01) + (0.0 if mi > ymax else ci_) + 0.006, txt, ha="center",
                    va="bottom", fontsize=8, color=INK2)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", label="genie (true Gittins for the current distribution)", zorder=2)
    ax.set_ylim(0.97, ymax + 0.06)
    ax.set_xticks(pos)
    ax.set_xticklabels([lab for lab, _, _ in TYPES], fontsize=9)
    ax.set_ylabel("mean response time ÷ genie   (whole run, 95% CI)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    headline(fig, "Which kinds of drift hurt a policy fitted once? Mix and shape do; a heavier tail barely does",
             f"Whole-run ratio to the genie, load ρ = {RHO}. Weights: 40 trials × 4,000 busy periods; shape: 20 × 2,500; "
             "tail: 40 × 4,000.", top=0.84)
    savefig(fig, os.path.join(FIG, "s13_drift_types.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["headline", "sweep", "pareto", "all"])
    ap.add_argument("--plot", action="store_true", help="only re-plot from CSVs")
    ap.add_argument("--quick", action="store_true", help="smoke test scale")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--sweep-trials", type=int, default=20)
    ap.add_argument("--busy", type=int, default=4000)
    ap.add_argument("--sweep-busy", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    if a.quick:
        a.trials, a.sweep_trials, a.busy, a.sweep_busy = 4, 3, 600, 600
    if not a.plot:
        if a.which in ("headline", "all"):
            run_headline(a.trials, a.busy, a.workers)
        if a.which in ("sweep", "all"):
            run_sweep(a.sweep_trials, a.sweep_busy, a.workers)
        if a.which in ("pareto", "all"):
            run_pareto(a.trials, a.busy, a.workers)
    if a.which in ("headline", "all"):
        plot_headline()
    if a.which in ("sweep", "all"):
        plot_sweep()
    if all(os.path.exists(os.path.join(RES, csv)) for _, csv, _ in TYPES):
        plot_types()
