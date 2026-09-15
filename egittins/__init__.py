"""
egittins — a small, self-contained simulator for empirical Gittins scheduling
in the M/G/1 queue.

Everything here follows the notation of

    Ramakrishna, Harlev, Scully. "Empirical Gittins for Data-Driven M/G/1
    Scheduling with Arbitrary Job Size Distributions." POMACS 10(1), 2026.

Design choices (see README):
  * Job sizes live on a grid with step `h` (default 0.01), as in the paper's
    Section 7.1, so every distribution is discrete and the Gittins rank
    function can be computed exactly (Definition 2.5, Observation A.1).
  * Time is simulated in quanta of length `h`; inter-arrival times are rounded
    to the nearest quantum. With mean job sizes of ~7-10 this is a <0.2% effect.
  * A SOAP policy is a rank array over integer ages (Section 2.1). Rank ∞
    means "PLCFS fallback" (Sections 2.1, 2.3).

This package is independent of the in-progress `sim/` package.
"""

from .distributions import GridDistribution
from .gittins import gittins_policy, fcfs_policy, plcfs_policy
from .simulate import simulate, mean_response_time

__all__ = [
    "GridDistribution",
    "gittins_policy",
    "fcfs_policy",
    "plcfs_policy",
    "simulate",
    "mean_response_time",
]
