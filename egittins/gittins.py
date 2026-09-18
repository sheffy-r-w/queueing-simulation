"""
gittins.py — the Gittins rank function for a discrete job size distribution.

Definition 2.5 of the paper: for a job size distribution G,

    r_{γ(G)}(a) = inf_{b > a}  E_{S~G}[(S ∧ b) − a | S > a]
                              ───────────────────────────────      if Ḡ(a) > 0,
                                 P_{S~G}[S ≤ b | S > a]

and r_{γ(G)}(a) = ∞ otherwise (the "PLCFS fallback").

For a discrete G with atoms G_1 < ... < G_K and masses p_1..p_K, the infimum is
attained at atoms (Observation A.1), so for an age a with G_{i-1} ≤ a < G_i
(G_0 := 0), writing T_j = P(S > G_j), V_j = E[(S − G_j)⁺] (the mean excess),
and T(a) = T_{i-1}, V(a) = V_{i-1} − (a − G_{i-1}) T_{i-1}:

    r(a) = min_{j ≥ i}  [ V(a) − V_j ] / [ T(a) − T_j ].

Numerically this is the ratio of two *tail* quantities, both accumulated from
the right, so it stays accurate at large ages where the remaining mass is
tiny (a prefix-sum formulation E[S ∧ b] − E[S ∧ a] cancels catastrophically
there). Since V_j = E[S] − E[S ∧ G_j] and T_j = 1 − F(G_j), the points
(T_j, V_j) are a point reflection of (F(G_j), E[S ∧ G_j]); either picture
gives the same slopes.

We evaluate r at every integer age (in units of the grid step h) below the
last atom. On each interval between atoms the rank is a minimum of decreasing
linear functions of a, hence decreasing (Observation A.1); it can only *jump
up* at an atom. The simulator uses this: a job in service can only lose
priority to a waiting job at an arrival or when it reaches its next atom.

A `Policy` bundles the rank array with `next_atom[a]` (the first atom > a,
or SENTINEL if none) so the simulator can skip ahead between events.

Two kernels compute the same table:

  * `_rank_kernel`      — the definition, O(K · L): every age tries every later atom.
  * `_rank_kernel_hull` — O(K + L). The rank at age a is the minimum slope
    from the query point Q(a) = (T(a), V(a)) to the points P_j = (T_j, V_j)
    to its left, which is attained on the convex hull of those points.
    Sweeping atoms right-to-left with a monotone stack builds the hull of
    each suffix in amortized O(1). Within an interval between atoms, Q(a)
    moves vertically, so the tangent point only moves toward nearer atoms —
    and it moves exactly over the vertices the stack pops when the next atom
    is pushed. Every age and every atom is touched O(1) times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .distributions import GridDistribution

SENTINEL = np.int64(2**62)


@njit(cache=True)
def _tail_tables(atoms_u, probs, h):
    """(G, T, V) indexed j = 0..K with G_0 = 0, T_0 = 1, V_0 = E[S].

    T_j = P(S > G_j) and V_j = E[(S − G_j)⁺] = Σ_{k>j} (G_k − G_{k−1}) h T_{k−1},
    both by reverse cumulative sums so that T_K = V_K = 0 exactly and small
    tails carry no cancellation error.
    """
    K = atoms_u.shape[0]
    G = np.empty(K + 1, dtype=np.int64)
    T = np.empty(K + 1)
    V = np.empty(K + 1)
    G[0] = 0
    for j in range(1, K + 1):
        G[j] = atoms_u[j - 1]
    T[K] = 0.0
    acc = 0.0
    for j in range(K - 1, -1, -1):
        acc += probs[j]
        T[j] = acc
    V[K] = 0.0
    acc = 0.0
    for j in range(K - 1, -1, -1):
        acc += (G[j + 1] - G[j]) * h * T[j]
        V[j] = acc
    return G, T, V


@njit(cache=True)
def _rank_kernel(atoms_u, probs, h, L):
    """
    Compute rank[a] and next_atom[a] for integer ages a in [0, L) directly
    from the definition, O(K · L). Ages at/after the last atom get rank = inf
    and next_atom = SENTINEL.
    """
    K = atoms_u.shape[0]
    G, T, V = _tail_tables(atoms_u, probs, h)

    rank = np.full(L, np.inf)
    next_atom = np.full(L, SENTINEL, dtype=np.int64)

    i = 1
    a_max = min(L, G[K])
    for a in range(a_max):
        while G[i] <= a:
            i += 1
        Tq = T[i - 1]
        yq = V[i - 1] - (a - G[i - 1]) * h * Tq
        best = np.inf
        for j in range(i, K + 1):
            den = Tq - T[j]
            if den <= 0.0:
                continue
            val = (yq - V[j]) / den
            if val < best:
                best = val
        rank[a] = best
        next_atom[a] = G[i]
    return rank, next_atom


@njit(cache=True)
def _rank_kernel_hull(atoms_u, probs, h, L):
    """Same output as `_rank_kernel`, in O(K + L) via the convex hull."""
    K = atoms_u.shape[0]
    G, T, V = _tail_tables(atoms_u, probs, h)

    rank = np.full(L, np.inf)
    next_atom = np.full(L, SENTINEL, dtype=np.int64)

    # st[0..m-1]: hull vertices of the current suffix of atoms, farthest (P_K)
    # at the bottom, nearest at the top.
    st = np.empty(K + 1, dtype=np.int64)
    st[0] = K
    m = 1
    for i in range(K, 0, -1):
        # ages [G_{i-1}, G_i); the stack holds the hull of P_i..P_K
        q = i - 1
        Tq = T[q]

        # tangent from P_q: pop while the farther vertex has the smaller slope
        yq = V[q]
        t0 = m - 1
        while t0 >= 1:
            dn = Tq - T[st[t0]]
            df = Tq - T[st[t0 - 1]]
            sn = (yq - V[st[t0]]) / dn if dn > 0.0 else np.inf
            sf = (yq - V[st[t0 - 1]]) / df if df > 0.0 else np.inf
            if sn >= sf:
                t0 -= 1
            else:
                break

        lo = G[q]
        hi = G[i] if G[i] < L else L
        t = t0
        for a in range(lo, hi):
            yq = V[q] - (a - G[q]) * h * Tq
            # Q(a) descends with a: the tangent moves toward nearer vertices only
            while t < m - 1:
                dn = Tq - T[st[t + 1]]
                dc = Tq - T[st[t]]
                sn = (yq - V[st[t + 1]]) / dn if dn > 0.0 else np.inf
                sc = (yq - V[st[t]]) / dc if dc > 0.0 else np.inf
                if sn <= sc:
                    t += 1
                else:
                    break
            dc = Tq - T[st[t]]
            rank[a] = (yq - V[st[t]]) / dc if dc > 0.0 else np.inf
            next_atom[a] = G[i]

        m = t0 + 1
        st[m] = q
        m += 1
    return rank, next_atom


def rank_hull_geometry(G: GridDistribution, a_u: int) -> dict:
    """The picture behind `_rank_kernel_hull` at age a (units), for plotting,
    in the (F(x), E[S ∧ x]) coordinates of the paper.

    Returns the points F_j, U_j (j = 0..K, P_0 = (0, 0)), the query point
    (F(a), E[S ∧ a]), the indices of the lower convex hull of the atoms above
    a, and the tangent index (the argmin of the rank) with the rank itself.
    """
    Gu, T, V = _tail_tables(G.atoms_u, G.probs, G.h)
    F = 1.0 - T
    U = V[0] - V
    q = int(np.searchsorted(Gu, a_u, side="right")) - 1   # G_q <= a < G_{q+1}
    xq = F[q]
    yq = U[q] + (a_u - Gu[q]) * G.h * T[q]
    idx = np.arange(q + 1, len(Gu))
    if len(idx) == 0:
        return dict(F=F, U=U, xq=xq, yq=yq, hull=idx, tangent=None, rank=np.inf)
    hull: list[int] = []
    for j in idx:
        while len(hull) >= 2:
            a1, a2 = hull[-2], hull[-1]
            cross = (F[a2] - F[a1]) * (U[j] - U[a1]) - (U[a2] - U[a1]) * (F[j] - F[a1])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(int(j))
    hull_arr = np.array(hull)
    # slopes from the tail tables, the same arithmetic as the kernels
    vq = V[q] - (a_u - Gu[q]) * G.h * T[q]
    slopes = (vq - V[hull_arr]) / (T[q] - T[hull_arr])
    k = int(np.argmin(slopes))
    return dict(F=F, U=U, xq=xq, yq=yq, hull=hull_arr, tangent=int(hull_arr[k]),
                rank=float(slopes[k]))


@dataclass(frozen=True)
class Policy:
    """A SOAP policy given by its rank function on integer ages (units of h)."""
    rank: np.ndarray        # float64[L]; inf = PLCFS fallback
    next_atom: np.ndarray   # int64[L]; first age > a where rank may jump up
    name: str

    @property
    def L(self) -> int:
        return len(self.rank)


def gittins_policy(G: GridDistribution, L: int, name: str | None = None,
                   method: str = "hull") -> Policy:
    """γ(G): the Gittins policy constructed from distribution G (Section 2.3).

    `L` is the number of integer ages to tabulate; pass the maximum job size
    (in units) of the *true* distribution plus one, so every job age the
    simulator can encounter has a defined rank.

    `method` is "hull" (O(K + L), default) or "direct" (O(K · L), the definition).
    """
    if method == "hull":
        rank, nxt = _rank_kernel_hull(G.atoms_u, G.probs, G.h, int(L))
    elif method == "direct":
        rank, nxt = _rank_kernel(G.atoms_u, G.probs, G.h, int(L))
    else:
        raise ValueError(f"unknown method {method!r}")
    return Policy(rank, nxt, name or f"Gittins({G.name})")


def policy_from_rank(rank: np.ndarray, name: str) -> Policy:
    """Wrap an arbitrary rank table as a Policy.

    The simulator re-decides only at arrivals, completions and `next_atom`
    crossings, which is exact as long as the rank is non-increasing in between.
    Here next_atom[a] is the first age a' > a at which the rank jumps up
    (rank[a'] > rank[a'-1]); for a Gittins table these are (a subset of) the
    atoms, for a learned rank they are wherever the table happens to rise.
    """
    rank = np.asarray(rank, dtype=np.float64)
    L = len(rank)
    up = np.zeros(L, dtype=bool)
    up[1:] = rank[1:] > rank[:-1]
    nxt = np.full(L, SENTINEL, dtype=np.int64)
    pending = SENTINEL
    for a in range(L - 1, -1, -1):
        nxt[a] = pending
        if up[a]:
            pending = a
    return Policy(rank, nxt, name)


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
