"""
Reference vs skip-ahead simulator kernel on the benchmark cells: seconds per call and the
speed-up, after asserting bit-identical per-job response times on each call.

Run:  python -m experiments.kernel_speedup [--repeats 3]  -> results/kernel_speedup.csv
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


def main(repeats: int):
    rows = []
    for dist_name, ctor in DISTS.items():
        F = ctor()
        L = F.max_u + 1
        G = GridDistribution.empirical(F.sample_u(np.random.default_rng(0), 1000), F.h)
        pols = {"fcfs": fcfs_policy(L), "empirical_gittins_n1000": gittins_policy(G, L),
                "true_gittins": gittins_policy(F, L)}
        for rho, n_busy in CELLS:
            for name, pol in pols.items():
                for k in ("reference", "skip"):
                    simulate(pol, F, rho, 200, seed=0, kernel=k)          # warm-up
                t_ref, t_skip = [], []
                for r in range(repeats):
                    t = time.perf_counter()
                    a = simulate(pol, F, rho, n_busy, seed=100 + r, kernel="reference")
                    t_ref.append(time.perf_counter() - t)
                    t = time.perf_counter()
                    b = simulate(pol, F, rho, n_busy, seed=100 + r, kernel="skip")
                    t_skip.append(time.perf_counter() - t)
                    assert np.array_equal(a.resp, b.resp) and np.array_equal(a.sizes_u, b.sizes_u)
                rows.append(dict(dist=dist_name, rho=rho, policy=name, n_busy=n_busy, jobs=len(a.resp),
                                 reference_s=np.mean(t_ref), skip_s=np.mean(t_skip),
                                 speedup=np.mean(t_ref) / np.mean(t_skip), repeats=repeats, bit_identical=True))
                print(f"{dist_name:15s} ρ={rho:<5} {name:24s} reference {np.mean(t_ref):7.3f} s  "
                      f"skip {np.mean(t_skip):7.3f} s  speed-up {np.mean(t_ref) / np.mean(t_skip):5.1f}×", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "kernel_speedup.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    main(ap.parse_args().repeats)
