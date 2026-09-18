"""
simulate.py: event-driven simulation of a preemptive M/G/1 under a SOAP policy.

Model (Section 2 of the paper): Poisson arrivals at rate lambda = rho / E[S], i.i.d. job
sizes from a distribution F, one server, fully preemptive, no overhead. The scheduler
serves the job of minimal rank, where rank is a function of the job's age (attained
service). Ties at a finite rank are broken FCFS; ties at rank inf are broken
preemptive-LCFS.

Time is discretized in quanta of length h (the grid step of the size distributions).
Inter-arrival times are rounded to the nearest quantum (minimum 1), which keeps the mean
inter-arrival time and the load unbiased.

Between atoms of the policy's distribution the Gittins rank is non-increasing in age
(Observation A.1), so the job in service can only be overtaken at an arrival or when it
reaches its next atom. Scheduling decisions are therefore made only at arrivals,
completions and atom crossings. FCFS and PLCFS have no atoms.

Two kernels, both compiled with numba:
  * _simulate_kernel       re-selects the served job at every atom crossing (reference)
  * _simulate_kernel_skip  same schedule, fewer re-selections (default)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .distributions import GridDistribution
from .gittins import Policy, SENTINEL


@njit(cache=True)
def _simulate_kernel(rank, next_atom, lam, h, atoms_u, cdf, n_busy, seed,
                     max_total, cap):
    """
    Returns (resp, sizes_u, n_done, n_bp_done, overflow).

    resp[k]   : response time (real units) of the k-th completed job
    sizes_u[k]: its size in units
    overflow  : 1 if the in-system job capacity `cap` or `max_total` was hit
    """
    np.random.seed(seed)
    L = rank.shape[0]
    K = atoms_u.shape[0]

    arr = np.empty(cap, dtype=np.int64)
    size = np.empty(cap, dtype=np.int64)
    att = np.empty(cap, dtype=np.int64)
    seq = np.empty(cap, dtype=np.int64)

    resp = np.empty(max_total)
    sizes_out = np.empty(max_total, dtype=np.int64)
    n_done = 0
    overflow = 0
    t = np.int64(0)
    next_seq = np.int64(0)
    n_bp_done = 0

    for bp in range(n_busy):
        # --- busy period starts with a single arrival at time t
        m = 0
        u = np.random.random()
        idx = np.searchsorted(cdf, u, side='right')
        if idx >= K:
            idx = K - 1
        arr[0] = t
        size[0] = atoms_u[idx]
        att[0] = 0
        seq[0] = next_seq
        next_seq += 1
        m = 1
        x = np.random.exponential(1.0 / lam)
        next_arr = t + max(np.int64(1), np.int64(np.floor(x / h + 0.5)))

        while m > 0:
            # --- select job of minimal rank
            best = 0
            best_rank = np.inf
            for k in range(m):
                a = att[k]
                r = rank[a] if a < L else np.inf
                if r < best_rank:
                    best = k
                    best_rank = r
                elif r == best_rank:
                    if r == np.inf:
                        if seq[k] > seq[best]:       # PLCFS: most recent wins
                            best = k
                    else:
                        if seq[k] < seq[best]:       # FCFS: earliest wins
                            best = k
            s = best
            # --- run until completion / next atom / next arrival
            dt = size[s] - att[s]
            a = att[s]
            if a < L:
                na = next_atom[a]
                if na - a < dt:
                    dt = na - a
            if next_arr - t < dt:
                dt = next_arr - t
            t += dt
            att[s] += dt
            if att[s] == size[s]:
                if n_done < max_total:
                    resp[n_done] = (t - arr[s]) * h
                    sizes_out[n_done] = size[s]
                    n_done += 1
                else:
                    overflow = 1
                # remove s by swapping in the last active job
                m -= 1
                if s != m:
                    arr[s] = arr[m]
                    size[s] = size[m]
                    att[s] = att[m]
                    seq[s] = seq[m]
            while next_arr == t:
                if m >= cap:
                    overflow = 1
                    break
                u = np.random.random()
                idx = np.searchsorted(cdf, u, side='right')
                if idx >= K:
                    idx = K - 1
                arr[m] = t
                size[m] = atoms_u[idx]
                att[m] = 0
                seq[m] = next_seq
                next_seq += 1
                m += 1
                x = np.random.exponential(1.0 / lam)
                next_arr = t + max(np.int64(1), np.int64(np.floor(x / h + 0.5)))
            if overflow == 1:
                break
        n_bp_done += 1
        if overflow == 1:
            break
        # idle period: the next busy period starts at the next arrival
        t = next_arr

    return resp[:n_done], sizes_out[:n_done], n_done, n_bp_done, overflow


@njit(cache=True)
def _simulate_kernel_skip(rank, next_atom, lam, h, atoms_u, cdf, n_busy, seed,
                          max_total, cap):
    """Same schedule as `_simulate_kernel` with fewer re-selections; per-job output is identical.

    The reference kernel re-selects the minimum-rank job at every atom crossing of the served
    job. Between arrivals and completions the served job can only lose the server to the best
    waiting job, whose rank does not change. This kernel walks the served job's atom chain with
    a constant-time comparison against that job (rank, then the FCFS / PLCFS tie-break on
    sequence number) and stops where the reference kernel would switch jobs.
    """
    np.random.seed(seed)
    L = rank.shape[0]
    K = atoms_u.shape[0]

    arr = np.empty(cap, dtype=np.int64)
    size = np.empty(cap, dtype=np.int64)
    att = np.empty(cap, dtype=np.int64)
    seq = np.empty(cap, dtype=np.int64)

    resp = np.empty(max_total)
    sizes_out = np.empty(max_total, dtype=np.int64)
    n_done = 0
    overflow = 0
    t = np.int64(0)
    next_seq = np.int64(0)
    n_bp_done = 0

    for bp in range(n_busy):
        u = np.random.random()
        idx = np.searchsorted(cdf, u, side='right')
        if idx >= K:
            idx = K - 1
        arr[0] = t
        size[0] = atoms_u[idx]
        att[0] = 0
        seq[0] = next_seq
        next_seq += 1
        m = 1
        x = np.random.exponential(1.0 / lam)
        next_arr = t + max(np.int64(1), np.int64(np.floor(x / h + 0.5)))

        while m > 0:
            # --- select job of minimal rank (identical rule to the reference kernel)
            best = 0
            best_rank = np.inf
            for k in range(m):
                a = att[k]
                r = rank[a] if a < L else np.inf
                if r < best_rank:
                    best = k
                    best_rank = r
                elif r == best_rank:
                    if r == np.inf:
                        if seq[k] > seq[best]:
                            best = k
                    else:
                        if seq[k] < seq[best]:
                            best = k
            s = best
            # --- run until completion / next arrival / first atom at which s would lose
            a = att[s]
            dt = size[s] - a
            if next_arr - t < dt:
                dt = next_arr - t
            if m > 1 and a < L and next_atom[a] - a < dt:
                # best waiting job: the only one that can take the server before the next event
                w_rank = np.inf
                w_seq = np.int64(-1)
                have_w = False
                for k in range(m):
                    if k == s:
                        continue
                    a_k = att[k]
                    r = rank[a_k] if a_k < L else np.inf
                    if not have_w:
                        w_rank = r
                        w_seq = seq[k]
                        have_w = True
                    elif r < w_rank:
                        w_rank = r
                        w_seq = seq[k]
                    elif r == w_rank:
                        if r == np.inf:
                            if seq[k] > w_seq:
                                w_seq = seq[k]
                        else:
                            if seq[k] < w_seq:
                                w_seq = seq[k]
                na = next_atom[a]
                while na - a < dt:
                    r_s = rank[na] if na < L else np.inf
                    lose = False
                    if r_s > w_rank:
                        lose = True
                    elif r_s == w_rank:
                        if r_s == np.inf:
                            lose = seq[s] < w_seq          # PLCFS tie: most recent wins
                        else:
                            lose = seq[s] > w_seq          # FCFS tie: earliest wins
                    if lose:
                        dt = na - a
                        break
                    if na >= L:
                        break                              # no further decision points beyond L
                    na = next_atom[na]
            t += dt
            att[s] += dt
            if att[s] == size[s]:
                if n_done < max_total:
                    resp[n_done] = (t - arr[s]) * h
                    sizes_out[n_done] = size[s]
                    n_done += 1
                else:
                    overflow = 1
                m -= 1
                if s != m:
                    arr[s] = arr[m]
                    size[s] = size[m]
                    att[s] = att[m]
                    seq[s] = seq[m]
            while next_arr == t:
                if m >= cap:
                    overflow = 1
                    break
                u = np.random.random()
                idx = np.searchsorted(cdf, u, side='right')
                if idx >= K:
                    idx = K - 1
                arr[m] = t
                size[m] = atoms_u[idx]
                att[m] = 0
                seq[m] = next_seq
                next_seq += 1
                m += 1
                x = np.random.exponential(1.0 / lam)
                next_arr = t + max(np.int64(1), np.int64(np.floor(x / h + 0.5)))
            if overflow == 1:
                break
        n_bp_done += 1
        if overflow == 1:
            break
        t = next_arr

    return resp[:n_done], sizes_out[:n_done], n_done, n_bp_done, overflow


@dataclass
class SimResult:
    resp: np.ndarray       # response times of completed jobs (real units)
    sizes_u: np.ndarray    # their sizes (units)
    n_busy: int
    overflow: bool

    @property
    def mean_response_time(self) -> float:
        return float(self.resp.mean())


def simulate(policy: Policy, F: GridDistribution, rho: float, n_busy: int,
             seed: int, max_total: int | None = None, cap: int = 20000,
             kernel: str = "skip") -> SimResult:
    """Simulate `n_busy` busy periods of an M/G/1 with true size distribution F
    at load ρ under `policy`.

    kernel = "skip" (default) or "reference". The two kernels give identical per-job response
    times (tests/test_simulator.py)."""
    if not (0 < rho < 1):
        raise ValueError("load must be in (0, 1)")
    lam = rho / F.mean()
    if max_total is None:
        max_total = int(6 * n_busy / (1.0 - rho)) + 10_000
    kern = _simulate_kernel_skip if kernel == "skip" else _simulate_kernel
    resp, sizes_u, n_done, n_bp, overflow = kern(
        policy.rank, policy.next_atom, float(lam), float(F.h),
        F.atoms_u, F.cdf, int(n_busy), int(seed), int(max_total), int(cap))
    return SimResult(resp, sizes_u, n_bp, bool(overflow))


def mean_response_time(policy: Policy, F: GridDistribution, rho: float,
                       n_busy: int, seed: int) -> float:
    return simulate(policy, F, rho, n_busy, seed).mean_response_time


# ---------------------------------------------------------------------------
# Closed forms used for validation (Kleinrock / Harchol-Balter)
# ---------------------------------------------------------------------------

def fcfs_mean_response_time(F: GridDistribution, rho: float) -> float:
    """Pollaczek-Khinchine: E[T] = E[S] + λ E[S²] / (2 (1 − ρ))."""
    lam = rho / F.mean()
    return F.mean() + lam * F.second_moment() / (2.0 * (1.0 - rho))


def plcfs_mean_response_time(F: GridDistribution, rho: float) -> float:
    """Preemptive LCFS (and PS): E[T] = E[S] / (1 − ρ)."""
    return F.mean() / (1.0 - rho)
