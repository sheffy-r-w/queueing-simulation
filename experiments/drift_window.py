"""
Window size for the k-updating scheduler under distribution drift (Section 8.5).

At the start of each busy period an empirical Gittins policy is built from the sizes of the
most recent P completed jobs and run for that busy period. P trades variance (small P) against
bias (large P, pre-drift data). Refits happen at busy-period boundaries because within a busy
period the completed jobs are a biased sample of the arrivals.

Drift model: F_k = (1 - theta_k) F_A + theta_k F_B on the common grid, theta_k a triangle wave
of period T busy periods; T = inf is the stationary case. F_A is the paper's 1-6-14
distribution, F_B a 3-8-14 mixture.

Measured quantity: mean response time over the measurement horizon divided by that of an
oracle using true Gittins for the current F_k. All policies see the same arrival/size stream.

Run:  python -m experiments.drift_window --quick     # smoke test
      python -m experiments.drift_window             # full run
      python -m experiments.drift_window --plot      # re-plot from CSV
"""

import argparse
import os
from collections import deque
from itertools import product
from multiprocessing import Pool

import numpy as np

from egittins.distributions import GridDistribution, one_six_fourteen
from egittins.gittins import gittins_policy, fcfs_policy, plcfs_policy
from egittins.simulate import simulate

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

RHO = 0.8            # overridden by --rho
FB_MEANS = [3.0, 8.0, 14.0]   # overridden by --fb
WINDOWS = [10, 30, 100, 300, 1000, 3000]
PERIODS = [np.inf, 1000, 300, 100]         # drift period T in busy periods (∞ = stationary)
N_THETA = 41                                 # oracle policies precomputed on a θ grid
TAG = ""


def make_endpoints(h=0.01):
    """F_A = the paper's 1-6-14 mixture; F_B = a 3-8-14 mixture (same sd 0.5, equal weights)."""
    from scipy.stats import norm
    sd = 0.5

    def mix_cdf(means):
        means = np.asarray(means, dtype=np.float64)
        def cdf(x):
            x = np.asarray(x, dtype=np.float64)[..., None]
            return norm.cdf((x - means) / sd).mean(axis=-1)
        return cdf

    FA = GridDistribution.from_continuous_cdf(mix_cdf([1.0, 6.0, 14.0]), h, 16.0, h, "A")
    FB = GridDistribution.from_continuous_cdf(mix_cdf(FB_MEANS), h, 16.0, h, "B")
    return FA, FB


def interpolate(FA: GridDistribution, FB: GridDistribution, theta: float) -> GridDistribution:
    """Mixture interpolation on the union grid."""
    atoms = np.union1d(FA.atoms_u, FB.atoms_u)
    pa = np.zeros(len(atoms)); pa[np.searchsorted(atoms, FA.atoms_u)] = FA.probs
    pb = np.zeros(len(atoms)); pb[np.searchsorted(atoms, FB.atoms_u)] = FB.probs
    p = (1 - theta) * pa + theta * pb
    keep = p > 0
    return GridDistribution(atoms[keep], p[keep], FA.h, f"mix(θ={theta:.2f})")


def theta_at(k: int, T: float) -> float:
    if not np.isfinite(T):
        return 0.0
    phase = (k % T) / T           # in [0,1)
    return 2 * phase if phase < 0.5 else 2 * (1 - phase)


def run_one(args):
    """One full run for a given drift period T and seed. Returns dict of ratios."""
    T, seed, n_busy, n_warm = args
    FA, FB = make_endpoints()
    L = max(FA.max_u, FB.max_u) + 1
    thetas = np.linspace(0, 1, N_THETA)
    oracle_pols = [gittins_policy(interpolate(FA, FB, th), L) for th in thetas]
    fcfs = fcfs_policy(L)

    # history buffers (one per window, plus "all history")
    hist = {P: deque(maxlen=P) for P in WINDOWS}
    hist_all = []
    tot_resp = {("window", P): 0.0 for P in WINDOWS}
    tot_resp[("all", 0)] = 0.0
    tot_resp[("oracle", 0)] = 0.0
    tot_resp[("fcfs", 0)] = 0.0
    n_jobs = 0
    rng_seed = 100_000 * seed

    for k in range(n_warm + n_busy):
        th = theta_at(k, T)
        Fk = interpolate(FA, FB, th)
        oracle = oracle_pols[int(round(th * (N_THETA - 1)))]
        measuring = k >= n_warm
        bp_seed = rng_seed + k

        # oracle and fcfs (also used to record the size stream for the history)
        r_or = simulate(oracle, Fk, RHO, 1, bp_seed)
        r_fc = simulate(fcfs, Fk, RHO, 1, bp_seed)
        assert not r_or.overflow
        if measuring:
            tot_resp[("oracle", 0)] += r_or.resp.sum()
            tot_resp[("fcfs", 0)] += r_fc.resp.sum()
            n_jobs += len(r_or.resp)

        # windowed empirical Gittins policies, fitted from history *before* this busy period
        for P in WINDOWS:
            if len(hist[P]) >= 2:
                pol = gittins_policy(GridDistribution.empirical(np.fromiter(hist[P], np.int64), FA.h), L)
            else:
                pol = plcfs_policy(L)   # cold start
            r = simulate(pol, Fk, RHO, 1, bp_seed)
            if measuring:
                tot_resp[("window", P)] += r.resp.sum()
        if len(hist_all) >= 2:
            pol = gittins_policy(GridDistribution.empirical(np.asarray(hist_all, np.int64), FA.h), L)
        else:
            pol = plcfs_policy(L)
        r = simulate(pol, Fk, RHO, 1, bp_seed)
        if measuring:
            tot_resp[("all", 0)] += r.resp.sum()

        # update histories with this busy period's completed sizes (identical across policies)
        sizes = r_or.sizes_u
        for P in WINDOWS:
            hist[P].extend(sizes.tolist())
        hist_all.extend(sizes.tolist())

    out = dict(T=T, seed=seed, n_jobs=n_jobs)
    oracle_mean = tot_resp[("oracle", 0)] / n_jobs
    out["oracle_mrt"] = oracle_mean
    out["fcfs"] = tot_resp[("fcfs", 0)] / n_jobs / oracle_mean
    out["all"] = tot_resp[("all", 0)] / n_jobs / oracle_mean
    for P in WINDOWS:
        out[f"P{P}"] = tot_resp[("window", P)] / n_jobs / oracle_mean
    return out


def run(n_busy, n_warm, seeds, workers):
    import pandas as pd
    jobs = [(T, s, n_busy, n_warm) for T in PERIODS for s in range(seeds)]
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            rows.append(r)
            print(f"done T={r['T']} seed={r['seed']}  all={r['all']:.3f} "
                  + " ".join(f"P{P}={r[f'P{P}']:.3f}" for P in WINDOWS), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"drift_window{TAG}.csv"), index=False)
    return pd.DataFrame(rows)


def plot():
    import pandas as pd
    import matplotlib.pyplot as plt
    from egittins.plotting import use_style, savefig, INK, INK2, BLUE, ORANGE, AQUA, VIOLET

    use_style()
    df = pd.read_csv(os.path.join(RES, f"drift_window{TAG}.csv"))
    colors = [INK, BLUE, ORANGE, VIOLET]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for T, c in zip(PERIODS, colors):
        sub = df[np.isinf(df["T"])] if not np.isfinite(T) else df[df["T"] == T]
        if len(sub) == 0:
            continue
        m = np.array([sub[f"P{P}"].mean() for P in WINDOWS])
        se = np.array([sub[f"P{P}"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0 for P in WINDOWS])
        label = "stationary (no drift)" if not np.isfinite(T) else f"drift period T = {int(T)} busy periods"
        ax.errorbar(WINDOWS, m, yerr=se, color=c, marker="o", ms=5, capsize=3, label=label)
        ax.plot([WINDOWS[0], WINDOWS[-1]], [sub["all"].mean()] * 2, color=c, lw=1, ls=":")
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("window size  P  (most recent completed jobs used to fit empirical Gittins)")
    ax.set_ylabel("mean response time / oracle (true Gittins for the current F)")
    ax.set_title(f"Gray-box empirical Gittins under drift: choosing the window  (ρ = {RHO})")
    ax.legend(loc="upper right", fontsize=8, title="dotted = use all history", title_fontsize=8)
    savefig(fig, os.path.join(FIG, f"drift_window{TAG}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--busy", type=int, default=6000, help="measured busy periods per run")
    ap.add_argument("--warm", type=int, default=600, help="warm-up busy periods (history fills)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--fb", type=str, default="3,8,14", help="component means of F_B")
    ap.add_argument("--tag", type=str, default="", help="suffix for results/figure filenames")
    a = ap.parse_args()
    RHO = a.rho
    FB_MEANS = [float(v) for v in a.fb.split(",")]
    TAG = a.tag
    if a.quick:
        a.busy, a.warm, a.seeds = 600, 200, 2
    if not a.plot:
        run(a.busy, a.warm, a.seeds, a.workers)
    plot()
