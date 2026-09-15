"""
gittins.py — the Gittins rank function for a discrete job size distribution.

Definition 2.5 of the paper: for a job size distribution G,

    r_{γ(G)}(a) = inf_{b > a}  E_{S~G}[(S ∧ b) − a | S > a]
                              ───────────────────────────────      if Ḡ(a) > 0,
                                 P_{S~G}[S ≤ b | S > a]

and r_{γ(G)}(a) = ∞ otherwise (the "PLCFS fallback").

For a discrete G with atoms G_1 < ... < G_K and masses p_1..p_K, the infimum is
attained at atoms (Observation A.1), so for an age a with G_{i-1} ≤ a < G_i,
writing T_j = P(S > G_j), C_j = Σ_{k≤j} G_k p_k, T_{i-1} := P(S > a):

    r(a) = min_{j ≥ i}  [ C_j − C_{i−1} + G_j T_j − a T_{i−1} ] / [ T_{i−1} − T_j ].

We evaluate this at every integer age (in units of the grid step h) below the
last atom. On each interval between atoms the rank is a minimum of decreasing
linear functions of a, hence decreasing (Observation A.1); it can only *jump
up* at an atom. The simulator uses this: a job in service can only lose
priority to a waiting job at an arrival or when it reaches its next atom.

A `Policy` bundles the rank array with `next_atom[a]` (the first atom > a,
or SENTINEL if none) so the simulator can skip ahead between events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .distributions import GridDistribution

SENTINEL = np.int64(2**62)


@njit(cache=True)
def _rank_kernel(atoms_u, probs, h, L):
    """
    Compute rank[a] and next_atom[a] for integer ages a in [0, L).
    Ages at/after the last atom get rank = inf and next_atom = SENTINEL.
    """
    K = atoms_u.shape[0]
    C = np.empty(K)
    T = np.empty(K)
    acc_c = 0.0
    for j in range(K):
        acc_c += atoms_u[j] * h * probs[j]
        C[j] = acc_c
    # tails by reverse cumulative sum: exactly 0 at the last atom, and no
    # cancellation for tiny masses
    acc_t = 0.0
    T[K - 1] = 0.0
    for j in range(K - 2, -1, -1):
        acc_t += probs[j + 1]
        T[j] = acc_t
    U = np.empty(K)
    for j in range(K):
        U[j] = C[j] + atoms_u[j] * h * T[j]

    rank = np.full(L, np.inf)
    next_atom = np.full(L, SENTINEL, dtype=np.int64)

    i = 0
    last = atoms_u[K - 1]
    a_max = min(L, last)
    for a in range(a_max):
        while atoms_u[i] <= a:
            i += 1
        if i == 0:
            Tprev = 1.0
            Cprev = 0.0
        else:
            Tprev = T[i - 1]
            Cprev = C[i - 1]
        ar = a * h
        best = np.inf
        base = Cprev + ar * Tprev
        for j in range(i, K):
            den = Tprev - T[j]
            if den <= 0.0:
                continue
            val = (U[j] - base) / den
            if val < best:
                best = val
        rank[a] = best
        next_atom[a] = atoms_u[i]
    return rank, next_atom


@dataclass(frozen=True)
class Policy:
    """A SOAP policy given by its rank function on integer ages (units of h)."""
    rank: np.ndarray        # float64[L]; inf = PLCFS fallback
    next_atom: np.ndarray   # int64[L]; first age > a where rank may jump up
    name: str

    @property
    def L(self) -> int:
        return len(self.rank)


def gittins_policy(G: GridDistribution, L: int, name: str | None = None) -> Policy:
    """γ(G): the Gittins policy constructed from distribution G (Section 2.3).

    `L` is the number of integer ages to tabulate; pass the maximum job size
    (in units) of the *true* distribution plus one, so every job age the
    simulator can encounter has a defined rank.
    """
    rank, nxt = _rank_kernel(G.atoms_u, G.probs, G.h, int(L))
    return Policy(rank, nxt, name or f"Gittins({G.name})")


def fcfs_policy(L: int) -> Policy:
    """FCFS: constant finite rank, ties broken first-come first-served."""
    return Policy(np.zeros(L), np.full(L, SENTINEL, dtype=np.int64), "FCFS")


def plcfs_policy(L: int) -> Policy:
    """Preemptive LCFS: rank ∞ everywhere, so the PLCFS tie-break always applies."""
    return Policy(np.full(L, np.inf), np.full(L, SENTINEL, dtype=np.int64), "PLCFS")


def gittins_rank_bruteforce(G: GridDistribution, a_u: int) -> float:
    """Definition 2.5 evaluated literally (for tests). a_u in units."""
    a = a_u * G.h
    S = G.atoms
    p = G.probs
    above = S > a
    if not above.any():
        return np.inf
    pa = p[above].sum()
    best = np.inf
    for b in S[above]:
        num = np.sum((np.minimum(S[above], b) - a) * p[above]) / pa
        den = np.sum(p[above & (S <= b)]) / pa
        if den > 0:
            best = min(best, num / den)
    return float(best)
