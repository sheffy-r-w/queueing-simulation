"""
distributions.py — job size distributions on a grid.

A `GridDistribution` is a discrete distribution supported on positive
multiples of a step `h`. Atoms are stored as integer multiples of `h`
("units") together with their probabilities.

The two "true" distributions from Section 7.1 of the paper are provided:

  * `one_six_fourteen()` — Gaussian mixture with component means 1, 6, 14,
    standard deviation 0.5, equal weights, bounded to (0, 16], step 0.01.
  * `bounded_pareto()`   — Pareto with scale x_m = 2 and shape α = 1.2,
    bounded above at 500, step 0.01.

`GridDistribution.empirical(samples)` is Definition 2.1 (the empirical CDF
puts mass 1/n on each sample; ties are merged into one atom with summed mass,
which is the same distribution).

`GridDistribution.truncate(ell)` is Definition 2.4: all mass at or above ℓ is
moved to a single atom at ℓ, so the tail is Ḡ(x) for x < ℓ and 0 for x ≥ ℓ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class GridDistribution:
    atoms_u: np.ndarray   # int64, sorted, strictly increasing, all > 0
    probs: np.ndarray     # float64, positive, sums to 1
    h: float              # grid step (job size = atoms_u * h)
    name: str = ""

    # ----------------------------------------------------------------- basics
    def __post_init__(self):
        a = np.asarray(self.atoms_u, dtype=np.int64)
        p = np.asarray(self.probs, dtype=np.float64)
        if a.ndim != 1 or p.shape != a.shape:
            raise ValueError("atoms_u and probs must be 1-D arrays of equal length")
        if len(a) == 0:
            raise ValueError("distribution must have at least one atom")
        if np.any(np.diff(a) <= 0):
            raise ValueError("atoms_u must be strictly increasing")
        if a[0] <= 0:
            raise ValueError("job sizes must be strictly positive")
        if np.any(p <= 0):
            raise ValueError("all probabilities must be positive")
        p = p / p.sum()
        object.__setattr__(self, "atoms_u", a)
        object.__setattr__(self, "probs", p)

    @property
    def atoms(self) -> np.ndarray:
        """Atom locations in real units."""
        return self.atoms_u * self.h

    @property
    def max_u(self) -> int:
        return int(self.atoms_u[-1])

    @property
    def cdf(self) -> np.ndarray:
        return np.cumsum(self.probs)

    def mean(self) -> float:
        return float(np.sum(self.atoms * self.probs))

    def second_moment(self) -> float:
        return float(np.sum(self.atoms**2 * self.probs))

    def variance(self) -> float:
        return self.second_moment() - self.mean() ** 2

    def tail(self, x) -> np.ndarray:
        """Tail function F̄(x) = P(S > x), vectorized over x (real units)."""
        x = np.asarray(x, dtype=np.float64)
        idx = np.searchsorted(self.atoms, x, side="right")  # number of atoms <= x
        # tail[k] = P(S > k-th atom) computed as a reverse cumulative sum so the
        # value at and beyond the last atom is exactly 0.
        rev = np.concatenate([np.cumsum(self.probs[::-1])[::-1][1:], [0.0]])
        t = np.concatenate([[1.0], rev])
        return t[idx]

    def quantile_u(self, q: float) -> int:
        """Smallest atom (in units) x with P(S <= x) >= q."""
        idx = int(np.searchsorted(self.cdf, q, side="left"))
        idx = min(idx, len(self.atoms_u) - 1)
        return int(self.atoms_u[idx])

    # --------------------------------------------------------------- sampling
    def sample_u(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw n i.i.d. sizes, returned in units (int64)."""
        u = rng.random(n)
        idx = np.searchsorted(self.cdf, u, side="right")
        idx = np.minimum(idx, len(self.atoms_u) - 1)
        return self.atoms_u[idx]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.sample_u(rng, n) * self.h

    # ----------------------------------------------------------- constructors
    @classmethod
    def empirical(cls, samples_u, h: float, name: str = "empirical") -> "GridDistribution":
        """Definition 2.1: empirical distribution of the given samples (units)."""
        s = np.asarray(samples_u, dtype=np.int64)
        if s.size == 0:
            raise ValueError("need at least one sample")
        atoms, counts = np.unique(s, return_counts=True)
        return cls(atoms, counts / counts.sum(), h, name)

    def truncate(self, ell_u: int, name: str | None = None) -> "GridDistribution":
        """Definition 2.4: ℓ-truncated distribution (mass at/above ℓ moved to ℓ)."""
        ell_u = int(ell_u)
        if ell_u <= 0:
            raise ValueError("truncation level must be positive")
        below = self.atoms_u < ell_u
        atoms = np.concatenate([self.atoms_u[below], [ell_u]])
        mass_above = self.probs[~below].sum()
        probs = np.concatenate([self.probs[below], [mass_above]])
        keep = probs > 0
        return GridDistribution(atoms[keep], probs[keep], self.h,
                                name or f"{self.name}|trunc@{ell_u * self.h:g}")

    @classmethod
    def from_continuous_cdf(cls, cdf, lo: float, hi: float, h: float, name: str) -> "GridDistribution":
        """
        Discretize a continuous distribution with CDF `cdf` onto the grid
        {k*h : lo <= k*h <= hi}, giving grid point x the mass of the bin
        [x - h/2, x + h/2), then renormalizing to the bounded support.
        """
        k_lo = max(1, int(np.ceil(lo / h - 1e-9)))
        k_hi = int(np.floor(hi / h + 1e-9))
        k = np.arange(k_lo, k_hi + 1, dtype=np.int64)
        x = k * h
        p = cdf(x + h / 2) - cdf(x - h / 2)
        p = np.clip(p, 0.0, None)
        keep = p > 0
        return cls(k[keep], p[keep], h, name)


# ---------------------------------------------------------------------------
# The two distributions from Section 7.1
# ---------------------------------------------------------------------------

def one_six_fourteen(h: float = 0.01) -> GridDistribution:
    """Gaussian mixture, means 1/6/14, sd 0.5, equal weights, bounded to (0, 16]."""
    means = np.array([1.0, 6.0, 14.0])
    sd = 0.5

    def cdf(x):
        x = np.asarray(x, dtype=np.float64)[..., None]
        return norm.cdf((x - means) / sd).mean(axis=-1)

    return GridDistribution.from_continuous_cdf(cdf, lo=h, hi=16.0, h=h, name="1-6-14")


def bounded_pareto(h: float = 0.01, x_m: float = 2.0, alpha: float = 1.2,
                   upper: float = 500.0) -> GridDistribution:
    """Pareto(x_m, α) bounded above at `upper`."""

    def cdf(x):
        x = np.asarray(x, dtype=np.float64)
        return np.where(x < x_m, 0.0, 1.0 - (x_m / np.maximum(x, x_m)) ** alpha)

    return GridDistribution.from_continuous_cdf(cdf, lo=x_m, hi=upper, h=h, name="bounded-Pareto")
