"""
kupdating.py: the k-updating scheduler of Section 8.5 under distribution drift.

A stream of busy periods k = 0, 1, ... is simulated with distribution F_k = drift.dist(k - n_warm).
Every policy sees the same arrival/size stream in every busy period (the busy period's seed
is shared and the arrival process does not depend on the scheduler), so comparisons are paired.

Policies:
  * genie     true Gittins for the current F_k
  * fcfs
  * static    empirical Gittins fitted once, at the first measured busy period, from the
              last `static_window` completed sizes
  * kupd_w    empirical Gittins refitted at the start of every busy period from the last w
              completed sizes, for each w in `windows`

Refits happen only at busy-period boundaries. Within a busy period the set of completed jobs
depends on the policy and is a biased sample; at a boundary every arrival has completed.

`run_stream` returns per-busy-period sums of response times for every policy. `summarize`
turns a list of trial results into mean ratios to the genie with 95% t confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .distributions import GridDistribution
from .drift import DriftModel
from .gittins import Policy, gittins_policy, fcfs_policy, plcfs_policy
from .simulate import simulate

DEFAULT_WINDOWS = (50, 200, 500, 2000)


@dataclass
class StreamResult:
    policies: list[str]
    resp_sum: np.ndarray      # [n_policies, n_busy] total response time per measured busy period
    n_jobs: np.ndarray        # [n_busy] jobs completed per measured busy period
    seed: int

    def mrt(self, name: str) -> float:
        return float(self.resp_sum[self.policies.index(name)].sum() / self.n_jobs.sum())

    def ratio(self, name: str, ref: str = "genie") -> float:
        return self.mrt(name) / self.mrt(ref)


def _fit(sizes_u: np.ndarray, h: float, L: int) -> Policy:
    if len(sizes_u) == 0:
        return plcfs_policy(L)                       # cold start, never measured
    return gittins_policy(GridDistribution.empirical(sizes_u, h), L)


def run_stream(drift: DriftModel, rho: float, n_busy: int, n_warm: int, seed: int,
               windows=DEFAULT_WINDOWS, static_window: int = 500,
               L: int | None = None, fitters: dict | None = None) -> StreamResult:
    """Simulate n_warm + n_busy busy periods; only the last n_busy are measured.

    The history of completed sizes starts empty and fills during warm-up (the
    drift parameter is held at its k = 0 value there). Busy period k uses seed
    100_000 * seed + k for every policy.

    `fitters` adds further k-updating policies: {name: (window, fit)} with
    fit(sizes_u, h, L) -> Policy, refit every busy period like the Gittins ones.
    """
    windows = tuple(int(w) for w in windows)
    if L is None:
        L = drift.max_u() + 1
    h = drift.dist(0).h
    specs = [(f"kupd_{w}", w, _fit) for w in windows]
    for name, (w, fn) in (fitters or {}).items():
        specs.append((name, int(w), fn))
    names = ["genie", "fcfs", "static"] + [s[0] for s in specs]
    max_window = max([static_window] + [s[1] for s in specs])
    genie_cache: dict[int, Policy] = {}
    fcfs = fcfs_policy(L)
    hist: list[int] = []
    resp_sum = np.zeros((len(names), n_busy))
    n_jobs = np.zeros(n_busy, dtype=np.int64)
    static: Policy | None = None
    base = 100_000 * seed

    for k in range(n_warm + n_busy):
        m = k - n_warm
        idx = drift.grid_index(m)
        Fk = drift.dist_at(idx)
        if idx not in genie_cache:
            genie_cache[idx] = gittins_policy(Fk, L)
        genie = genie_cache[idx]
        bp_seed = base + k

        r = simulate(genie, Fk, rho, 1, bp_seed)
        assert not r.overflow
        sizes = r.sizes_u
        if m >= 0:
            if static is None:
                static = _fit(np.asarray(hist[-static_window:], np.int64), h, L)
            resp_sum[0, m] = r.resp.sum()
            n_jobs[m] = len(r.resp)
            resp_sum[1, m] = simulate(fcfs, Fk, rho, 1, bp_seed).resp.sum()
            resp_sum[2, m] = simulate(static, Fk, rho, 1, bp_seed).resp.sum()
            for j, (_, w, fit) in enumerate(specs):
                pol = fit(np.asarray(hist[-w:], np.int64), h, L)
                r_j = simulate(pol, Fk, rho, 1, bp_seed)
                assert not r_j.overflow
                resp_sum[3 + j, m] = r_j.resp.sum()
        hist.extend(sizes.tolist())
        if len(hist) > 2 * max_window:
            del hist[: len(hist) - max_window]

    return StreamResult(names, resp_sum, n_jobs, seed)


def summarize(results: list[StreamResult], ref: str = "genie", conf: float = 0.95):
    """Mean paired ratio MRT(policy)/MRT(ref) over trials with a t confidence interval.

    Returns a list of dicts: policy, mean, ci (half-width), sd, n.
    """
    out = []
    n = len(results)
    tcrit = stats.t.ppf(0.5 + conf / 2, n - 1) if n > 1 else np.nan
    for name in results[0].policies:
        r = np.array([res.ratio(name, ref) for res in results])
        sd = r.std(ddof=1) if n > 1 else 0.0
        out.append(dict(policy=name, mean=float(r.mean()), ci=float(tcrit * sd / np.sqrt(n)) if n > 1 else np.nan,
                        sd=float(sd), n=n))
    return out


def blocks(results: list[StreamResult], block: int) -> dict[str, np.ndarray]:
    """Pooled ratio to the genie per block of `block` consecutive busy periods.

    Returns {policy: array over blocks}; ratios pool response-time sums over
    all trials in the block, i.e. a ratio of means, paired by construction.
    """
    P = len(results[0].policies)
    n_busy = results[0].resp_sum.shape[1]
    nb = n_busy // block
    tot = np.zeros((P, nb))
    for res in results:
        tot += res.resp_sum[:, : nb * block].reshape(P, nb, block).sum(axis=2)
    g = results[0].policies.index("genie")
    return {name: tot[i] / tot[g] for i, name in enumerate(results[0].policies)}
