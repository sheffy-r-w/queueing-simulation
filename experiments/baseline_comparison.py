"""
Baseline comparison — the protocol of Section 7.1, reproduced.

For each (distribution, load ρ, number of samples n) and each of `trials`
independent trials:
  1. draw n samples from F and form the empirical distribution G;
  2. build empirical Gittins γ(G) and truncated empirical Gittins γ(G_ℓ), with
     ℓ chosen so that Ḡ(ℓ) = n^{-1/3} (1-ρ)^{2/3} (the paper's rule, Thm 6.1 with α→∞);
  3. simulate `n_busy` busy periods of the M/G/1 under each policy and record
     the mean response time.

The reported quantity is the ratio of a trial's mean response time to the
mean response time of *true* Gittins γ(F) simulated on the SAME arrival/size
stream (common random numbers: the same seed and number of busy periods), so
every ratio is paired and the simulation noise of a single trial cancels.
A long unpaired true-Gittins reference (4 × 100k busy periods) is kept only to
place the FCFS and PLCFS closed forms (Pollaczek-Khinchine and E[S]/(1-ρ)) on
the ratio scale.

Run:   python -m experiments.baseline_comparison            # full run, writes results/*.csv
       python -m experiments.baseline_comparison --plot     # only re-plot from CSV
"""

import argparse
import os
import sys
from itertools import product
from multiprocessing import Pool

import numpy as np

from egittins.distributions import GridDistribution, one_six_fourteen, bounded_pareto
from egittins.gittins import gittins_policy
from egittins.simulate import simulate, fcfs_mean_response_time, plcfs_mean_response_time

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

DISTS = {"1-6-14": one_six_fourteen, "bounded-Pareto": bounded_pareto}
LOADS = [0.8, 0.98]
NS = [10, 100, 1000]


def truncation_level_u(G: GridDistribution, n: int, rho: float) -> int:
    target_tail = n ** (-1 / 3) * (1 - rho) ** (2 / 3)
    return G.quantile_u(1.0 - target_tail)


def one_trial(args):
    dist_name, rho, n, trial, n_busy = args
    F = DISTS[dist_name]()
    L = F.max_u + 1
    rng = np.random.default_rng(1_000_003 * trial + 7919 * n + int(rho * 100))
    G = GridDistribution.empirical(F.sample_u(rng, n), F.h)
    ell = truncation_level_u(G, n, rho)
    pol_e = gittins_policy(G, L)
    pol_t = gittins_policy(G.truncate(ell), L)
    pol_true = gittins_policy(F, L)
    seed = 10_000 * trial + n                      # one arrival stream per trial, shared by all three policies
    r_e = simulate(pol_e, F, rho, n_busy, seed)
    r_t = simulate(pol_t, F, rho, n_busy, seed)
    r_o = simulate(pol_true, F, rho, n_busy, seed)
    assert not (r_e.overflow or r_t.overflow or r_o.overflow), "job buffer overflow; raise cap/max_total"
    return dict(dist=dist_name, rho=rho, n=n, trial=trial, ell=ell * F.h,
                empirical=r_e.mean_response_time, truncated=r_t.mean_response_time,
                true_paired=r_o.mean_response_time)


def true_gittins_reference(dist_name, rho, n_busy_ref, seeds=4):
    F = DISTS[dist_name]()
    pol = gittins_policy(F, F.max_u + 1)
    vals = [simulate(pol, F, rho, n_busy_ref, seed=555 + s).mean_response_time for s in range(seeds)]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(seeds))


def run(trials, n_busy_08, n_busy_098, n_busy_ref, workers, out="baseline_comparison", loads=LOADS):
    """Resumable: (distribution, load) configs already complete in the CSV are skipped."""
    import pandas as pd
    csv = os.path.join(RES, f"{out}.csv")
    rows = []
    done = set()
    if os.path.exists(csv):
        prev = pd.read_csv(csv)
        counts = prev.groupby(["dist", "rho"]).size()
        done = {k for k, v in counts.items() if v >= trials * len(NS)}
        prev = prev[[ (d, r) in done for d, r in zip(prev.dist, prev.rho)]]
        rows = prev.to_dict("records")
        print(f"resuming: {len(rows)} rows kept for completed configs {sorted(done)}", flush=True)
    for dist_name, rho in product(DISTS, loads):
        if (dist_name, rho) in done:
            continue
        mrt_ref, se_ref = true_gittins_reference(dist_name, rho, n_busy_ref)
        F = DISTS[dist_name]()
        print(f"[{dist_name} rho={rho}] true Gittins E[T] = {mrt_ref:.3f} ± {se_ref:.3f}  "
              f"FCFS {fcfs_mean_response_time(F, rho):.2f}  PLCFS {plcfs_mean_response_time(F, rho):.2f}",
              flush=True)
        n_busy = n_busy_08 if rho < 0.9 else n_busy_098
        jobs = [(dist_name, rho, n, t, n_busy) for n in NS for t in range(trials)]
        with Pool(workers) as pool:
            for r in pool.imap_unordered(one_trial, jobs, chunksize=4):
                r["true_gittins"] = mrt_ref
                r["fcfs"] = fcfs_mean_response_time(F, rho)
                r["plcfs"] = plcfs_mean_response_time(F, rho)
                rows.append(r)
        print(f"[{dist_name} rho={rho}] done {len(jobs)} trials", flush=True)
        pd.DataFrame(rows).to_csv(csv, index=False)
    return pd.DataFrame(rows)


def plot(out="baseline_comparison"):
    """out = "baseline_comparison": writes s11_baseline.png and s05b_baseline_teaser.png.
    Any other stem: reads results/<out>.csv, draws the (distribution, load) cells present, and
    writes figures/s11_<out>.png."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from egittins.plotting import use_style, savefig, headline, style_box, ref_line, BLUE, ORANGE, INK, INK2, SLIDES

    use_style()
    df = pd.read_csv(os.path.join(RES, f"{out}.csv"))
    df["r_emp"] = df.empirical / df.true_paired        # paired: same seed as the trial
    df["r_trunc"] = df.truncated / df.true_paired

    def panel(ax, dist_name, rho):
        sub = df[(df.dist == dist_name) & (df.rho == rho)]
        pos = np.arange(len(NS))
        data_e = [sub[sub.n == n].r_emp.values for n in NS]
        data_t = [sub[sub.n == n].r_trunc.values for n in NS]
        w = 0.3
        b1 = ax.boxplot(data_e, positions=pos - w / 2 - 0.02, widths=w * 0.8, patch_artist=True, showfliers=False)
        b2 = ax.boxplot(data_t, positions=pos + w / 2 + 0.02, widths=w * 0.8, patch_artist=True, showfliers=False)
        style_box(b1, BLUE)
        style_box(b2, ORANGE)
        fcfs = sub.fcfs.iloc[0] / sub.true_gittins.iloc[0]      # closed forms vs the long reference
        plcfs = sub.plcfs.iloc[0] / sub.true_gittins.iloc[0]
        allv = np.concatenate(data_e + data_t)
        ymax = max(np.percentile(allv, 95) * 1.05, plcfs * 1.08)
        ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=1)
        ref_line(ax, plcfs, f"PLCFS  {plcfs:.2f}", "-.")
        if fcfs < ymax:
            ref_line(ax, fcfs, f"FCFS  {fcfs:.2f}", ":", side="left")
        else:
            ax.text(0.005, 0.98, f"FCFS  {fcfs:.2f}  (off scale)", transform=ax.transAxes, fontsize=8,
                    color=INK2, ha="left", va="top")
        ax.set_ylim(0.97, ymax * 1.08)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("number of past jobs in the sample  n")
        ax.set_ylabel("mean response time ÷ optimal")
        ax.set_title(f"{dist_name} job sizes,  load ρ = {rho}")
        ax.grid(axis="x", visible=False)

    handles = [Patch(facecolor=BLUE, alpha=0.28, edgecolor=BLUE, label="empirical Gittins"),
               Patch(facecolor=ORANGE, alpha=0.28, edgecolor=ORANGE, label="truncated empirical Gittins"),
               Line2D([], [], color=INK, ls="--", lw=1, label="true Gittins (optimal) = 1"),
               Line2D([], [], color=INK2, ls=":", lw=1, label="first-come first-served (FCFS)"),
               Line2D([], [], color=INK2, ls="-.", lw=1, label="preemptive last-come first-served (PLCFS)")]
    ntr = int(df.groupby(["dist", "rho", "n"]).size().min())

    if out != "baseline_comparison":
        cells = sorted({(d, r) for d, r in zip(df.dist, df.rho)}, key=lambda c: (c[1], c[0]))
        fig, axes = plt.subplots(1, len(cells), figsize=(5.6 * len(cells), 4.6), squeeze=False)
        fig.subplots_adjust(wspace=0.22)
        for ax, (dist_name, rho) in zip(axes.ravel(), cells):
            panel(ax, dist_name, rho)
        headline(fig, "Baseline comparison, longer runs",
                 f"Mean response time relative to true Gittins. {ntr} trials per box, each paired with true Gittins "
                 "on the same arrival stream.\nBox = quartiles, line = median, whiskers = 1.5 × IQR.", top=0.72)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.83), ncol=5, columnspacing=1.8)
        savefig(fig, os.path.join(FIG, f"s11_{out}.png"))
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    fig.subplots_adjust(hspace=0.45, wspace=0.22)
    for ax, (dist_name, rho) in zip(axes.ravel(), product(DISTS, LOADS)):
        panel(ax, dist_name, rho)
    headline(fig, "Mean response time relative to the optimal schedule, by sample size, load and workload",
             f"Mean response time relative to true Gittins. {ntr} trials per box, each paired with true Gittins on the same "
             "arrival stream.\nBox = quartiles, line = median, whiskers = 1.5 × IQR.", top=0.825)
    if SLIDES:                                    # no headline: give the legend its own band above the panels
        fig.subplots_adjust(top=0.9)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.99), ncol=5, columnspacing=1.8)
    else:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.9), ncol=5, columnspacing=1.8)
    savefig(fig, os.path.join(FIG, "s11_baseline.png"))

    # teaser for the EDA slide: one panel
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    panel(ax, "1-6-14", 0.8)
    ax.set_title("")
    headline(fig, "A hundred past jobs gets within 5% of optimal at load 0.8",
             f"1-6-14 job sizes, load ρ = 0.8. {ntr} trials per box, each paired with true Gittins on the same arrival stream.\n"
             "Box = quartiles, line = median, whiskers = 1.5 × IQR.", top=0.82)
    ax.legend(handles=handles[:3], loc="upper right")
    savefig(fig, os.path.join(FIG, "s05b_baseline_teaser.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="only re-plot from results CSV")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--busy08", type=int, default=10_000, help="busy periods per trial at rho=0.8")
    ap.add_argument("--busy098", type=int, default=4_000, help="busy periods per trial at rho=0.98")
    ap.add_argument("--busyref", type=int, default=100_000, help="busy periods per seed for the true-Gittins reference")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="baseline_comparison",
                    help="stem for results/<out>.csv and figures/s11_<out>.png (default: the main figures)")
    ap.add_argument("--loads", default=",".join(str(r) for r in LOADS), help="comma-separated loads to run")
    a = ap.parse_args()
    if not a.plot:
        run(a.trials, a.busy08, a.busy098, a.busyref, a.workers, out=a.out,
            loads=[float(r) for r in a.loads.split(",")])
    plot(a.out)
