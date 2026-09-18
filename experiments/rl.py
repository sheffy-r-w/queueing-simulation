"""
Section 13 (optional): REINFORCE on the rank function, from scratch and from imitation.

Setting: 1-6-14 at ρ = 0.8, one fixed window of 500 samples (the same one for
every arm and seed). Episodes are busy periods; reward = −(sum of response
times); the stochastic policy draws the job to serve ∝ exp(−rank/τ) at every
arrival and completion. Evaluation is deterministic in the SOAP simulator,
paired against true Gittins on a fixed seed.

Arms (3 seeds each):
  scratch    log-rank table over ages with 32 interpolated knots, constant init (= FCFS)
  finetune   the imitation net (`results/imitation_full.pt`) as the initial policy

-> results/s13_rl_curves.csv, figures/s13_rl.png

Run:  python -m experiments.rl [--quick]
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from egittins.distributions import GridDistribution, one_six_fourteen
from egittins.gittins import gittins_policy, fcfs_policy
from egittins.imitation import RankNet
from egittins.rl import train_rl, RLConfig
from egittins.simulate import simulate

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
RHO = 0.8
N_SAMPLES = 500
SAMPLE_SEED = 2
ARMS = {
    "scratch": dict(init=None, lr=0.03, tau=2.0, knots=32),
    "finetune": dict(init="full", lr=3e-5, tau=0.3, knots=0),
}


def _run(args):
    arm, seed, iters, episodes = args
    F = one_six_fourteen()
    s = F.sample_u(np.random.default_rng(SAMPLE_SEED), N_SAMPLES)
    spec = ARMS[arm]
    init = RankNet.load(os.path.join(RES, f"imitation_{spec['init']}.pt")) if spec["init"] else None
    cfg = RLConfig(tau=spec["tau"], lr=spec["lr"], knots=spec["knots"], iters=iters,
                   episodes_per_iter=episodes, eval_every=max(1, iters // 24), eval_busy=3000)
    _, curve, pol = train_rl(F, s, RHO, init, "full", seed=seed, cfg=cfg, verbose=False)
    # final paired evaluation on a longer, separate seed
    L = F.max_u + 1
    ref = simulate(gittins_policy(F, L), F, RHO, 10_000, seed=4242).mean_response_time
    final = simulate(pol, F, RHO, 10_000, seed=4242).mean_response_time / ref
    for row in curve:
        row.update(arm=arm)
    return arm, seed, curve, final, pol.rank


def main(iters: int, episodes: int, seeds: int, workers: int):
    jobs = [(arm, s, iters, episodes) for arm in ARMS for s in range(seeds)]
    curves, finals, ranks = [], [], {}
    with Pool(workers) as pool:
        for arm, seed, curve, final, rank in pool.imap_unordered(_run, jobs):
            print(f"  {arm} seed {seed}: final MRT/true Gittins = {final:.4f}", flush=True)
            curves += curve
            finals.append(dict(arm=arm, seed=seed, final_ratio=final))
            ranks[(arm, seed)] = rank
    pd.DataFrame(curves).to_csv(os.path.join(RES, "s13_rl_curves.csv"), index=False)
    fin = pd.DataFrame(finals)
    fin.to_csv(os.path.join(RES, "s13_rl_final.csv"), index=False)
    np.savez(os.path.join(RES, "s13_rl_ranks.npz"), **{f"{a}_{s}": r for (a, s), r in ranks.items()})
    print(fin.groupby("arm").final_ratio.agg(["mean", "min", "max"]).round(4))
    plot()


def plot():
    import matplotlib.pyplot as plt
    from egittins.plotting import use_style, savefig, headline, INK, INK2, BLUE, AQUA, VIOLET
    use_style()
    F = one_six_fourteen()
    L = F.max_u + 1
    s = F.sample_u(np.random.default_rng(SAMPLE_SEED), N_SAMPLES)
    ref = simulate(gittins_policy(F, L), F, RHO, 3000, seed=777).mean_response_time
    fcfs = simulate(fcfs_policy(L), F, RHO, 3000, seed=777).mean_response_time / ref
    emp = simulate(gittins_policy(GridDistribution.empirical(s, F.h), L), F, RHO, 3000, seed=777).mean_response_time / ref
    df = pd.read_csv(os.path.join(RES, "s13_rl_curves.csv"))
    ranks = np.load(os.path.join(RES, "s13_rl_ranks.npz"))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.subplots_adjust(wspace=0.24)
    for arm, c in (("scratch", AQUA), ("finetune", VIOLET)):
        for seed, sub in df[df.arm == arm].groupby("seed"):
            ax.plot(sub["iter"], sub.ratio, color=c, lw=1.2, alpha=0.8,
                    label=f"{'from scratch, 3 seeds (32-knot rank table, starts at FCFS)' if arm == 'scratch' else 'fine-tune the imitation net, 3 seeds'}"
                    if seed == 0 else None)
    ax.axhline(1.0, color=INK, ls="--", lw=1, label="true Gittins")
    ax.axhline(emp, color=BLUE, ls="-.", lw=1, label=f"empirical Gittins, same sample ({emp:.3f})")
    ax.axhline(fcfs, color=INK2, ls=":", lw=1, label=f"first-come first-served (FCFS), {fcfs:.3f}")
    ax.set_xlabel("REINFORCE iteration  (1024 busy-period episodes each)")
    ax.set_ylabel("mean response time / true Gittins")
    ax.set_title("(a) learning curves, 3 seeds per arm")
    ax.set_ylim(0.95, 2.35)
    ax.legend(loc="upper right")

    ages = np.arange(L) * F.h
    m = np.arange(L) < s.max()
    ax2.plot(ages[m], gittins_policy(F, L).rank[m], color=INK, lw=2, label="true Gittins")
    for arm, c in (("scratch", AQUA), ("finetune", VIOLET)):
        r = ranks[f"{arm}_0"]
        ax2.plot(ages[m], r[m], color=c, lw=1.0, label=f"{arm}, seed 0 (final)")
    ax2.set_ylim(0, 24)
    ax2.set_xlabel("age  a")
    ax2.set_ylabel("rank  r(a)")
    ax2.set_title("(b) learned rank functions, seed 0")
    ax2.legend()
    headline(fig, "Policy gradient holds the imitation optimum but cannot discover the rank structure from scratch",
             "1-6-14 job sizes, load ρ = 0.8, one window of 500 samples; REINFORCE on busy-period episodes; "
             "ratios paired against true Gittins.", top=0.82)
    savefig(fig, os.path.join(FIG, "s13_rl.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.quick:
        a.iters, a.episodes, a.seeds = 60, 128, 2
    if a.plot:
        plot()
    else:
        main(a.iters, a.episodes, a.seeds, a.workers)
