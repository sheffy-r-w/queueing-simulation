"""The stochastic episode kernel behind REINFORCE."""

import numpy as np
import pytest

from egittins.distributions import one_six_fourteen
from egittins.rl import _episodes_kernel
from egittins.simulate import fcfs_mean_response_time


def test_size_blind_policy_matches_pollaczek_khinchine():
    # Constant rank => uniform random choice at every event: size- and age-blind, so
    # the conservation law gives the FCFS mean response time.
    F = one_six_fourteen()
    lam = 0.8 / F.mean()
    rank = np.zeros(F.max_u + 1)
    out = _episodes_kernel(rank, lam, F.h, F.atoms_u, F.cdf, 4000, 3, 1.0, 20000, 400_000, 4_000_000)
    seg_ptr, flat, chosen, rtg, area_total, n_jobs, D, M, overflow = out
    assert not overflow
    assert abs(area_total.sum() / n_jobs - fcfs_mean_response_time(F, 0.8)) / fcfs_mean_response_time(F, 0.8) < 0.05
    # bookkeeping: one chosen index per decision, inside its segment; return-to-go is
    # minus the remaining area, so the first decision of an episode carries the whole area
    assert D == len(chosen) == len(rtg) and seg_ptr[-1] == M == len(flat)
    seg_sizes = np.diff(seg_ptr)
    assert np.all(chosen < seg_sizes) and np.all(seg_sizes >= 1)
    assert rtg[0] == pytest.approx(-area_total[0])
    assert np.all(rtg <= 0)


def test_low_temperature_prefers_low_rank():
    # SRPT-like rank (remaining size of a deterministic job) with tiny tau: the job in
    # service is never preempted by an equal-size arrival, so no decision ever picks a
    # job with age 0 while another job has positive age.
    F = one_six_fourteen()
    lam = 0.8 / F.mean()
    rank = np.linspace(10.0, 0.0, F.max_u + 1)
    out = _episodes_kernel(rank, lam, F.h, F.atoms_u, F.cdf, 500, 1, 1e-6, 20000, 400_000, 4_000_000)
    seg_ptr, flat, chosen, *_ = out
    for d in range(len(chosen)):
        seg = flat[seg_ptr[d]: seg_ptr[d + 1]]
        assert seg[chosen[d]] == seg.max()
