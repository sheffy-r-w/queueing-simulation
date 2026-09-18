"""
drift.py: non-stationary job size distributions for the drift experiments (Section 8.5).

A `DriftModel` maps the index k of a measured busy period to the job size distribution F_k in
force during that busy period. It composes a schedule u(k) in [0, 1] with a one-parameter
family of distributions D(u). D is evaluated on a grid of `n_grid` values of u and cached, so
the genie policy for F_k is computed once per grid point.

Schedules:
  * triangle(T)   u goes 0 -> 1 -> 0 with period T busy periods
  * ramp(N)       u goes 0 -> 1 linearly over N busy periods
  * constant(u)   stationary control

Families (grid h = 0.01):
  * mean_preserving_weights   1-6-14 with component weights moved along the line where E[S]
                              is fixed, so the load is fixed and only the shape changes
  * one_way_weights           weights (0.1, 0.3, 0.6) -> (0.6, 0.3, 0.1)
  * mode_shift                (1 - u) F(1-6-14) + u F(3-8-14), as in experiments/drift_window.py
  * pareto_tail               bounded Pareto with alpha 2.0 -> 1.2

The load rho is held fixed and the arrival rate is lambda_k = rho / E[F_k].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import norm

from .distributions import GridDistribution, bounded_pareto


# ----------------------------------------------------------------- schedules

def triangle(T: float) -> Callable[[int], float]:
    def u(k: int) -> float:
        phase = (k % T) / T
        return 2 * phase if phase < 0.5 else 2 * (1 - phase)
    return u


def ramp(N: int) -> Callable[[int], float]:
    def u(k: int) -> float:
        return min(k / max(N - 1, 1), 1.0)
    return u


def constant(u0: float = 0.0) -> Callable[[int], float]:
    def u(k: int) -> float:
        return u0
    return u


# ------------------------------------------------------------------ families

def gaussian_mixture(means, weights, sd: float = 0.5, hi: float = 16.0,
                     h: float = 0.01, name: str = "") -> GridDistribution:
    """Mixture of normals with the given component means and weights, bounded to (0, hi]."""
    means = np.asarray(means, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    def cdf(x):
        x = np.asarray(x, dtype=np.float64)[..., None]
        return (norm.cdf((x - means) / sd) * weights).sum(axis=-1)

    return GridDistribution.from_continuous_cdf(cdf, lo=h, hi=hi, h=h, name=name)


def interpolate(FA: GridDistribution, FB: GridDistribution, u: float, name: str = "") -> GridDistribution:
    """(1−u)·F_A + u·F_B on the union grid."""
    atoms = np.union1d(FA.atoms_u, FB.atoms_u)
    pa = np.zeros(len(atoms))
    pa[np.searchsorted(atoms, FA.atoms_u)] = FA.probs
    pb = np.zeros(len(atoms))
    pb[np.searchsorted(atoms, FB.atoms_u)] = FB.probs
    p = (1 - u) * pa + u * pb
    keep = p > 0
    return GridDistribution(atoms[keep], p[keep], FA.h, name or f"mix(u={u:.2f})")


MP_T_RANGE = (0.10, 0.70)     # weight on the 6-mode at u = 0 and u = 1


def mean_preserving_weights(u: float) -> GridDistribution:
    """1-6-14 with weights (w1, w6, w14) on the line w1 + 6 w6 + 14 w14 = 7, Σw = 1.

    Parametrized by w6 = t: w14 = (6 − 5t)/13, w1 = 1 − t − w14. At t = 1/3 this is
    the paper's equal-weight 1-6-14. t ranges over [t_lo, t_hi] = MP_T_RANGE.
    """
    t = MP_T_RANGE[0] + u * (MP_T_RANGE[1] - MP_T_RANGE[0])
    w14 = (6 - 5 * t) / 13
    w1 = 1 - t - w14
    return gaussian_mixture([1.0, 6.0, 14.0], [w1, t, w14], name=f"1-6-14 w=({w1:.2f},{t:.2f},{w14:.2f})")


ONE_WAY_W0 = np.array([0.1, 0.3, 0.6])
ONE_WAY_W1 = np.array([0.6, 0.3, 0.1])


def one_way_weights(u: float) -> GridDistribution:
    w = (1 - u) * ONE_WAY_W0 + u * ONE_WAY_W1
    return gaussian_mixture([1.0, 6.0, 14.0], w, name=f"1-6-14 w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})")


_MODE_SHIFT_ENDPOINTS: tuple[GridDistribution, GridDistribution] | None = None


def mode_shift(u: float) -> GridDistribution:
    global _MODE_SHIFT_ENDPOINTS
    if _MODE_SHIFT_ENDPOINTS is None:
        _MODE_SHIFT_ENDPOINTS = (gaussian_mixture([1.0, 6.0, 14.0], [1, 1, 1], name="1-6-14"),
                                 gaussian_mixture([3.0, 8.0, 14.0], [1, 1, 1], name="3-8-14"))
    return interpolate(*_MODE_SHIFT_ENDPOINTS, u, name=f"1-6-14→3-8-14 u={u:.2f}")


PARETO_ALPHA_RANGE = (2.0, 1.2)


def pareto_tail(u: float) -> GridDistribution:
    alpha = PARETO_ALPHA_RANGE[0] + u * (PARETO_ALPHA_RANGE[1] - PARETO_ALPHA_RANGE[0])
    G = bounded_pareto(alpha=alpha)
    return GridDistribution(G.atoms_u, G.probs, G.h, f"bounded-Pareto α={alpha:.2f}")


FAMILIES = {
    "mean-preserving": mean_preserving_weights,
    "one-way": one_way_weights,
    "mode-shift": mode_shift,
    "pareto-tail": pareto_tail,
}


# --------------------------------------------------------------------- model

@dataclass
class DriftModel:
    name: str
    family: Callable[[float], GridDistribution]
    schedule: Callable[[int], float]
    n_grid: int = 101
    _cache: dict = field(default_factory=dict, repr=False)

    def grid_index(self, k: int) -> int:
        u = float(self.schedule(max(int(k), 0)))
        return int(round(u * (self.n_grid - 1)))

    def dist_at(self, idx: int) -> GridDistribution:
        if idx not in self._cache:
            self._cache[idx] = self.family(idx / (self.n_grid - 1))
        return self._cache[idx]

    def dist(self, k: int) -> GridDistribution:
        return self.dist_at(self.grid_index(k))

    def max_u(self) -> int:
        return max(self.dist_at(i).max_u for i in range(self.n_grid))


def make_drift(family: str, schedule: str, T: float | None = None, N: int | None = None,
               u0: float = 0.0, n_grid: int = 101) -> DriftModel:
    """Convenience constructor: family in FAMILIES; schedule in {triangle, ramp, constant}."""
    fam = FAMILIES[family]
    if schedule == "triangle":
        sched, tag = triangle(T), f"triangle T={T:g}"
    elif schedule == "ramp":
        sched, tag = ramp(N), f"ramp over {N}"
    elif schedule == "constant":
        sched, tag = constant(u0), f"stationary u={u0:g}"
    else:
        raise ValueError(schedule)
    return DriftModel(f"{family} ({tag})", fam, sched, n_grid)
