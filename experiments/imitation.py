"""
Section 13 (imitation): a neural rank function trained to imitate empirical Gittins.

  train    build ~4000 random (distribution, sample) pairs with exact empirical
           Gittins targets and train one MLP per feature set in
           `egittins.imitation.FEATURE_SETS` (tail_only ⊂ no_hazard ⊂ full).
           -> results/imitation_<set>.pt, results/s13_imitation_training.csv
  overlay  true Gittins vs. empirical Gittins vs. the learned rank functions on
           a 500-sample window from 1-6-14 and from the bounded Pareto.
           -> figures/s13_rank_overlay.png
  static   the §7.1 protocol at n = 500: each trial draws a sample, fits every
           policy to it, and simulates 10,000 busy periods on one seed.
           -> results/s13_imitation_static*.csv
  kupd     k-updating (w = 500) under one-way weight drift, the learned nets
           refit every busy period next to empirical Gittins.
           -> results/s13_imitation_kupd*.csv

Run:  python -m experiments.imitation train|overlay|static|kupd|all [--quick]
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from egittins.distributions import GridDistribution, one_six_fourteen, bounded_pareto
from egittins.gittins import gittins_policy, fcfs_policy
from egittins.imitation import make_dataset, train, nn_policy, RankNet, FEATURE_SETS
from egittins.simulate import simulate

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

RHO = 0.8
N_SAMPLES = 500
DISTS = {"1-6-14": one_six_fourteen, "bounded-Pareto": bounded_pareto}
LABELS = {"full": "NN (full features)", "no_hazard": "NN without hazard features",
          "tail_only": "NN tail + mean excess only"}


def net_path(feature_set: str) -> str:
    return os.path.join(RES, f"imitation_{feature_set}.pt")


_NETS: dict[str, RankNet] = {}


def get_net(feature_set: str) -> RankNet:
    if feature_set not in _NETS:
        _NETS[feature_set] = RankNet.load(net_path(feature_set))
    return _NETS[feature_set]


def _worker_init():
    import torch
    torch.set_num_threads(1)


# --------------------------------------------------------------------- train

def run_train(n_dists: int, epochs: int, seed: int = 0):
    print(f"building dataset: {n_dists} random distributions", flush=True)
    X, y, g = make_dataset(n_dists, seed=seed)
    print(f"  {len(y)} rows, {X.shape[1]} features", flush=True)
    hist = []
    for fs in FEATURE_SETS:
        net, h = train(X, y, g, feature_set=fs, seed=seed, epochs=epochs)
        net.save(net_path(fs))
        hist += h
    df = pd.DataFrame(hist)
    df.to_csv(os.path.join(RES, "s13_imitation_training.csv"), index=False)
    last = df.groupby("feature_set").last()
    print("\nheld-out fit (log rank / mean excess):")
    print(last[["val_mse", "val_r2"]].to_string())


# ------------------------------------------------------------------- overlay

def plot_overlay(seed: int = 11):
    import matplotlib.pyplot as plt
    from egittins.plotting import use_style, savefig, headline, INK, INK2, BLUE, VIOLET, AQUA
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(wspace=0.24)
    for ax, (name, ctor) in zip(axes, DISTS.items()):
        F = ctor()
        L = F.max_u + 1
        s = F.sample_u(np.random.default_rng(seed), N_SAMPLES)
        ages = np.arange(L) * F.h
        truth = gittins_policy(F, L).rank
        emp = gittins_policy(GridDistribution.empirical(s, F.h), L).rank
        lim = int(s.max())
        m = np.arange(L) < lim
        ax.plot(ages[m], truth[m], color=INK, lw=2.4, label="true Gittins")
        ax.plot(ages[m], emp[m], color=BLUE, lw=1.0, alpha=0.9, label=f"empirical Gittins (n = {N_SAMPLES})")
        for fs, c, ls in (("full", VIOLET, "-"), ("no_hazard", AQUA, "--"), ("tail_only", INK2, ":")):
            r = nn_policy(get_net(fs), s, F.h, L).rank
            ax.plot(ages[m], r[m], color=c, lw=1.4 if fs == "full" else 1.1, ls=ls, label=LABELS[fs])
        ax.set_xlabel("age  a")
        ax.set_ylabel("rank  r(a)   (lower = higher priority)")
        ax.set_title(f"{name} job sizes")
        if name == "bounded-Pareto":
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(2, lim * F.h)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, columnspacing=1.8)
    headline(fig, "A small network reproduces the Gittins rank function from the sample alone",
             f"Trained on 4,000 synthetic distributions (these two held out); evaluated on the same {N_SAMPLES} samples as "
             "empirical Gittins. Ablations drop the hazard features, then the quantiles.", top=0.82)
    savefig(fig, os.path.join(FIG, "s13_rank_overlay.png"))


# -------------------------------------------------------------------- static

def _static_trial(args):
    dist_name, trial, n_busy = args
    F = DISTS[dist_name]()
    L = F.max_u + 1
    rng = np.random.default_rng(2_000_003 * trial + 17)
    s = F.sample_u(rng, N_SAMPLES)
    pols = {"true_gittins": gittins_policy(F, L),
            "empirical_gittins": gittins_policy(GridDistribution.empirical(s, F.h), L),
            "fcfs": fcfs_policy(L)}
    for fs in FEATURE_SETS:
        pols[f"nn_{fs}"] = nn_policy(get_net(fs), s, F.h, L)
    seed = 50_000 + trial
    out = dict(dist=dist_name, trial=trial)
    for name, pol in pols.items():
        r = simulate(pol, F, RHO, n_busy, seed)
        assert not r.overflow
        out[name] = r.mean_response_time
    return out


def _ci_table(df: pd.DataFrame, ref: str, key: str) -> pd.DataFrame:
    from scipy import stats
    rows = []
    for k, sub in df.groupby(key):
        n = len(sub)
        t = stats.t.ppf(0.975, n - 1)
        for col in [c for c in df.columns if c not in (key, "trial", ref)]:
            r = sub[col] / sub[ref]
            rows.append({key: k, "policy": col, "mean": r.mean(), "ci": t * r.std(ddof=1) / np.sqrt(n), "n": n})
    return pd.DataFrame(rows)


def run_static(trials: int, n_busy: int, workers: int):
    jobs = [(d, t, n_busy) for d in DISTS for t in range(trials)]
    print(f"static: {trials} trials × {n_busy} busy periods, n = {N_SAMPLES}, ρ = {RHO}", flush=True)
    rows = []
    with Pool(workers, initializer=_worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_static_trial, jobs)):
            rows.append(r)
            if (i + 1) % max(1, len(jobs) // 8) == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "s13_imitation_static.csv"), index=False)
    summ = _ci_table(df, ref="true_gittins", key="dist")
    summ.to_csv(os.path.join(RES, "s13_imitation_static_summary.csv"), index=False)
    print("\nStatic (mean response time / true Gittins, 95% CI):")
    print(summ.pivot(index="policy", columns="dist", values="mean").round(3).to_string())


# ---------------------------------------------------------------- k-updating

def _kupd_trial(args):
    from egittins.drift import make_drift
    from egittins.kupdating import run_stream
    seed, n_busy, n_warm = args
    d = make_drift("one-way", "ramp", N=n_busy)
    fitters = {f"nn_{fs}_500": (500, (lambda fs_: lambda su, h, L: nn_policy(get_net(fs_), su, h, L))(fs))
               for fs in FEATURE_SETS}
    return run_stream(d, RHO, n_busy, n_warm, seed, windows=(500,), static_window=500, fitters=fitters)


def run_kupd(trials: int, n_busy: int, workers: int, n_warm: int = 600):
    from egittins.kupdating import summarize
    print(f"k-updating: {trials} trials × {n_busy} busy periods, one-way drift, ρ = {RHO}", flush=True)
    results = []
    with Pool(workers, initializer=_worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_kupd_trial, [(s, n_busy, n_warm) for s in range(trials)])):
            results.append(r)
            if (i + 1) % max(1, trials // 8) == 0:
                print(f"  {i + 1}/{trials}", flush=True)
    rows = [dict(seed=r.seed, policy=p, mrt=r.mrt(p), ratio=r.ratio(p)) for r in results for p in r.policies]
    pd.DataFrame(rows).to_csv(os.path.join(RES, "s13_imitation_kupd.csv"), index=False)
    summ = pd.DataFrame(summarize(results))
    summ.to_csv(os.path.join(RES, "s13_imitation_kupd_summary.csv"), index=False)
    print("\nk-updating under one-way drift (mean response time / genie, 95% CI):")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["train", "overlay", "static", "kupd", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dists", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--busy", type=int, default=10_000, help="busy periods per static trial")
    ap.add_argument("--kupd-busy", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    if a.quick:
        a.dists, a.epochs, a.trials, a.busy, a.kupd_busy = 300, 8, 4, 1000, 500
    if a.which in ("train", "all"):
        run_train(a.dists, a.epochs)
    if a.which in ("overlay", "all"):
        plot_overlay()
    if a.which in ("static", "all"):
        run_static(a.trials, a.busy, a.workers)
    if a.which in ("kupd", "all"):
        run_kupd(a.trials, a.kupd_busy, a.workers)
