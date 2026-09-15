"""
Section 13: static vs. k-updating vs. genie under distribution drift.

Three experiments, all on the k-updating engine (`egittins/kupdating.py`) with
the drift models of `egittins/drift.py`; every policy sees the same arrival
stream (common random numbers) and results are paired ratios to the genie
(true Gittins for the current distribution) with 95% t confidence intervals
over independent trials.

  headline  one-way weight drift on 1-6-14, (0.6,0.3,0.1) → (0.1,0.3,0.6) over
            the run, plus a stationary control at the initial weights.
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
        "stationary": ("one-way", "constant", dict(u0=0.0)),
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
    from egittins.plotting import use_style, savefig, INK, INK2, BLUE, ORANGE, AQUA, VIOLET, SEQ3
    use_style()
    b = pd.read_csv(os.path.join(RES, "s13_headline_blocks.csv"))
    summ = pd.read_csv(os.path.join(RES, "s13_headline_summary.csv"))
    n_busy = int(pd.read_csv(os.path.join(RES, "s13_headline.csv")).n_busy.iloc[0])
    x = b.block_start + BLOCK / 2

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.7, 1]))
    fig.subplots_adjust(wspace=0.25)
    ax.plot(x, b.fcfs, color=INK2, lw=1.2, ls=":", label="FCFS")
    ax.plot(x, b.static, color=ORANGE, lw=1.8, label=f"static empirical Gittins (fit once, w = {STATIC_WINDOW})")
    for w, c in zip((50, 500, 2000), (SEQ3[0], BLUE, SEQ3[2])):
        ax.plot(x, b[f"kupd_{w}"], color=c, lw=1.4, label=f"k-updating, window w = {w}")
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", label="genie (true Gittins for current F)")
    ax.set_xlabel("busy period  k   (weights drift linearly (0.6, 0.3, 0.1) → (0.1, 0.3, 0.6))")
    ax.set_ylabel("mean response time / genie   (blocks of 100 busy periods)")
    ax.set_title("(a) one-way drift: a stale policy degrades, k-updating tracks")
    ax.set_xlim(0, n_busy)
    ax.legend(loc="upper left", fontsize=8)

    order = ["fcfs", "static"] + [f"kupd_{w}" for w in WINDOWS]
    labels = ["FCFS", f"static\n(w={STATIC_WINDOW})"] + [f"k-upd\nw={w}" for w in WINDOWS]
    pos = np.arange(len(order))
    for off, (model, c) in enumerate([("stationary", AQUA), ("one-way drift", VIOLET)]):
        s = summ[summ.model == model].set_index("policy").loc[order]
        ax2.bar(pos + (off - 0.5) * 0.38, s["mean"] - 1, 0.36, bottom=1, yerr=s["ci"], color=c,
                alpha=0.85, capsize=2, label=model)
    ax2.axhline(1.0, color=INK, lw=1.0, ls="--")
    ax2.set_xticks(pos)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("mean response time / genie  (95% CI)")
    ax2.set_title("(b) whole run: with and without drift")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper right", fontsize=8)
    ntr = int(summ.n.iloc[0])
    fig.suptitle(f"k-updating under drift, 1-6-14 weights, ρ = {RHO}, {ntr} trials × {n_busy} busy periods",
                 y=1.01)
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
    from egittins.plotting import use_style, savefig, INK, INK2, BLUE, ORANGE, AQUA, VIOLET
    use_style()
    summ = pd.read_csv(os.path.join(RES, "s13_sweep_summary.csv"))
    n_busy = int(pd.read_csv(os.path.join(RES, "s13_sweep.csv")).n_busy.iloc[0])
    colors = {"stationary": INK, "T=8000": BLUE, "T=2000": AQUA, "T=500": ORANGE, "T=100": VIOLET}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for model, c in colors.items():
        s = summ[summ.model == model].set_index("policy")
        if len(s) == 0:
            continue
        m = [s.loc[f"kupd_{w}", "mean"] for w in WINDOWS]
        ci = [s.loc[f"kupd_{w}", "ci"] for w in WINDOWS]
        label = "stationary" if model == "stationary" else f"drift period T = {model[2:]} busy periods"
        ax.errorbar(WINDOWS, m, yerr=ci, color=c, marker="o", ms=5, capsize=3, label=label)
        ax.plot([WINDOWS[0], WINDOWS[-1]], [s.loc["static", "mean"]] * 2, color=c, lw=1, ls=":")
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([str(w) for w in WINDOWS])
    ax.set_xlabel("k-updating window  w  (most recent completed jobs)")
    ax.set_ylabel("mean response time / genie   (95% CI)")
    ntr = int(summ.n.iloc[0])
    ax.set_title(f"Window vs. drift speed: mean-preserving 1-6-14 weight drift, ρ = {RHO}\n"
                 f"({ntr} trials × {n_busy} busy periods per point; dotted = static fit, w = {STATIC_WINDOW})",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
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
