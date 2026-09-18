"""
distribution.py — Empirical distribution over job size samples.

Stores sorted sample values (atoms) and answers queries needed by
scheduling policies: atom locations and the next atom above a given age.

Initialized with seed samples (observed job sizes). New samples are
added incrementally as jobs complete during the simulation.
"""

import numpy as np
from typing import Optional


class EmpiricalDistribution:
    """
    Empirical distribution over observed job sizes.

    Places mass 1/n at each sample (ties allowed — treated as
    separate atoms at the same location).

    Parameters
    ----------
    seed_samples : array-like
        Initial positive job size observations. Must be non-empty.
    """

    def __init__(self, seed_samples):
        arr = np.asarray(seed_samples, dtype=float)
        if arr.size == 0:
            raise ValueError("Must provide at least one seed sample.")
        if np.any(arr <= 0):
            raise ValueError("All job sizes must be strictly positive.")
        self._atoms = np.sort(arr)

    @property
    def n(self) -> int:
        """Number of samples."""
        return len(self._atoms)

    @property
    def atoms(self) -> np.ndarray:
        """Sorted array of sample values."""
        return self._atoms

    def add_sample(self, size: float) -> None:
        """
        Insert a completed job's size into the sorted atom array.

        Parameters
        ----------
        size : float
            Observed job size (must be strictly positive).
        """
        if size <= 0:
            raise ValueError(f"Job size must be strictly positive, got {size}.")
        idx = np.searchsorted(self._atoms, size)
        self._atoms = np.insert(self._atoms, idx, size)

    def next_atom_above(self, age: float) -> Optional[float]:
        """
        Return the smallest atom strictly greater than `age`,
        or None if no such atom exists.
        """
        idx = np.searchsorted(self._atoms, age, side="right")
        if idx >= len(self._atoms):
            return None
        return float(self._atoms[idx])

    def __repr__(self) -> str:
        return (
            f"EmpiricalDistribution(n={self.n}, "
            f"min={self._atoms[0]:.4f}, max={self._atoms[-1]:.4f})"
        )