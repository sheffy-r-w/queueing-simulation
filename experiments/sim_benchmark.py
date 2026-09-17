"""
Simulator throughput: seconds and jobs/second per `egittins.simulate.simulate` call for
FCFS, empirical Gittins (n = 1000) and true Gittins on both distributions, at ρ = 0.8
(10,000 busy periods) and ρ = 0.98 (4,000), i.e. the calls the baseline grid makes.
From these, an estimate of the full baseline grid (100 trials × 3 policies per cell,
plus the 4 × 100,000-busy-period true-Gittins reference per (distribution, load))
on all cores, in core-seconds, wall-clock and total simulated jobs.

Run:  python -m experiments.sim_benchmark [--repeats 3] [--workers N]
      -> results/sim_benchmark.csv
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from egittins.distributions import GridDistribution, one_six_fourteen, bounded_pareto
from egittins.gittins import gittins_policy, fcfs_policy
from egittins.simulate import simulate

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
DISTS = {"1-6-14": one_six_fourteen, "bounded-Pareto": bounded_pareto}
CELLS = [(0.8, 10_000), (0.98, 4_000)]
TRIALS, NS, POLICIES_PER_TRIAL, REF_SEEDS, REF_BUSY = 100, 3, 3, 4, 100_000


def main(repeats: int, workers: int):
    rows = []
    for dist_name, ctor in DISTS.items():
        F = ctor()
        L = F.max_u + 1
        G = GridDistribution.empirical(F.sample_u(np.random.default_rng(0), 1000), F.h)
        pols = {"fcfs": fcfs_policy(L), "empirical_gittins_n1000": gittins_policy(G, L),
                "true_gittins": gittins_policy(F, L)}
        for rho, n_busy in CELLS:
            for name, pol in pols.items():
                simulate(pol, F, rho, 200, seed=0)                    # JIT warm-up, excluded
                secs, jobs = [], []
                for r in range(repeats):
                    t = time.perf_counter()
                    res = simulate(pol, F, rho, n_busy, seed=100 + r)
                    secs.append(time.perf_counter() - t)
                    jobs.append(len(res.resp))
                    assert not res.overflow
                rows.append(dict(dist=dist_name, rho=rho, policy=name, n_busy=n_busy,
                                 jobs=float(np.mean(jobs)), seconds=float(np.mean(secs)),
                                 jobs_per_second=float(np.mean(jobs) / np.mean(secs)),
                                 seconds_sd=float(np.std(secs)), repeats=repeats))
                print(f"{dist_name:15s} ρ={rho:<5} {name:24s} {np.mean(jobs):9.0f} jobs  "
                      f"{np.mean(secs):7.3f} s  {np.mean(jobs) / np.mean(secs):10.0f} jobs/s", flush=True)
    df = pd.DataFrame(rows)

    # full-grid estimate: per (dist, rho) cell the grid runs TRIALS × NS trials, each simulating
    # the empirical, truncated-empirical and true-Gittins policies on n_busy busy periods; the
    # truncated policy is costed at the empirical policy's rate. Plus the reference runs.
    est = []
    for (d, rho), sub in df.groupby(["dist", "rho"]):
        s = sub.set_index("policy")
        per_trial_s = 2 * s.loc["empirical_gittins_n1000", "seconds"] + s.loc["true_gittins", "seconds"]
        per_trial_jobs = 3 * s.loc["true_gittins", "jobs"]
        n_trials = TRIALS * NS
        ref_s = REF_SEEDS * s.loc["true_gittins", "seconds"] * REF_BUSY / s.loc["true_gittins", "n_busy"]
        ref_jobs = REF_SEEDS * s.loc["true_gittins", "jobs"] * REF_BUSY / s.loc["true_gittins", "n_busy"]
        est.append(dict(dist=d, rho=rho, trials=n_trials, core_seconds=n_trials * per_trial_s + ref_s,
                        jobs=n_trials * per_trial_jobs + ref_jobs))
    est = pd.DataFrame(est)
    tot_s, tot_jobs = est.core_seconds.sum(), est.jobs.sum()
    print(f"\nfull baseline grid estimate: {tot_s:.0f} core-seconds, ~{tot_s / workers / 60:.1f} min wall on "
          f"{workers} workers (perfect scaling), {tot_jobs / 1e6:.1f} M jobs simulated")
    print(est.round(1).to_string(index=False))
    df["grid_core_seconds_total"] = tot_s
    df["grid_wall_minutes_on_workers"] = tot_s / workers / 60
    df["grid_workers"] = workers
    df["grid_jobs_total"] = tot_jobs
    df.to_csv(os.path.join(RES, "sim_benchmark.csv"), index=False)
    return df, est


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    main(a.repeats, a.workers)
