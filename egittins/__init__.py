"""
egittins: simulator and rank computations for empirical Gittins scheduling in the M/G/1 queue.

Notation follows Ramakrishna, Harlev, Scully, "Empirical Gittins for Data-Driven M/G/1
Scheduling with Arbitrary Job Size Distributions", POMACS 10(1), 2026.

Conventions:
  * Job sizes are on a grid with step h (default 0.01), so every distribution is discrete
    and the Gittins rank function is computed exactly (Definition 2.5, Observation A.1).
  * Time is simulated in quanta of length h. Inter-arrival times are rounded to the
    nearest quantum.
  * A SOAP policy is a rank array over integer ages (Section 2.1). Rank inf means the
    PLCFS fallback (Sections 2.1, 2.3).

Independent of the sim/ package.
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
