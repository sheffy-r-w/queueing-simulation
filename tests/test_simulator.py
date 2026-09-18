"""Validate the M/G/1 simulator against closed-form mean response times."""

import numpy as np
import pytest

from egittins.distributions import GridDistribution, one_six_fourteen, bounded_pareto
from egittins.gittins import gittins_policy, fcfs_policy, plcfs_policy
from egittins.simulate import (simulate, fcfs_mean_response_time,
                               plcfs_mean_response_time)


def _rel_err(x, y):
    return abs(x - y) / y


@pytest.mark.parametrize("F", [one_six_fourteen(), bounded_pareto()])
@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_fcfs_pollaczek_khinchine(F, rho):
    res = simulate(fcfs_policy(F.max_u + 1), F, rho, n_busy=20_000, seed=1)
    assert not res.overflow
    # Compare against P-K evaluated at the *realized* size moments: for the
    # heavy-tailed bounded Pareto, E[S^2] is dominated by rare huge jobs, so
    # the sampling noise in E[S^2] (not the simulator) dominates the error.
    S = res.sizes_u * F.h
    lam = rho / F.mean()
    rho_hat = lam * S.mean()
    pk_hat = S.mean() + lam * np.mean(S**2) / (2.0 * (1.0 - rho_hat))
    tol = 0.03 if F.name == "1-6-14" else 0.05
    assert _rel_err(res.mean_response_time, pk_hat) < tol
    assert _rel_err(res.mean_response_time, fcfs_mean_response_time(F, rho)) < 0.15


@pytest.mark.parametrize("F", [one_six_fourteen(), bounded_pareto()])
@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_plcfs_closed_form(F, rho):
    res = simulate(plcfs_policy(F.max_u + 1), F, rho, n_busy=40_000, seed=2)
    assert not res.overflow
    tol = 0.03 if F.name == "1-6-14" else 0.06   # heavy tail => slower convergence
    assert _rel_err(res.mean_response_time, plcfs_mean_response_time(F, rho)) < tol


def test_srpt_special_case_beats_fcfs():
    # Deterministic sizes: Gittins == SRPT == FCFS (no preemption benefit).
    G = GridDistribution(np.array([300]), np.array([1.0]), h=0.01)
    g = simulate(gittins_policy(G, G.max_u + 1), G, 0.7, 5_000, seed=3).mean_response_time
    f = simulate(fcfs_policy(G.max_u + 1), G, 0.7, 5_000, seed=3).mean_response_time
    assert g == pytest.approx(f)


def test_true_gittins_is_best():
    F = one_six_fourteen()
    rho = 0.8
    L = F.max_u + 1
    g = simulate(gittins_policy(F, L), F, rho, 5_000, seed=4).mean_response_time
    f = simulate(fcfs_policy(L), F, rho, 5_000, seed=4).mean_response_time
    p = simulate(plcfs_policy(L), F, rho, 5_000, seed=4).mean_response_time
    assert g < f and g < p


def test_skip_kernel_is_bit_identical_to_reference():
    """The skip-ahead kernel must give the same per-job response times and sizes as the
    reference kernel for every policy shape the repo uses: FCFS, PLCFS, empirical Gittins
    (n = 30, 1000), true Gittins, truncated, and arbitrary wiggly tables with ∞ regions."""
    from egittins.gittins import policy_from_rank
    rng = np.random.default_rng(5)
    for F in (one_six_fourteen(), bounded_pareto()):
        L = F.max_u + 1
        pols = [fcfs_policy(L), plcfs_policy(L), gittins_policy(F, L)]
        for n in (30, 1000):
            G = GridDistribution.empirical(F.sample_u(rng, n), F.h)
            pols.append(gittins_policy(G, L))
            pols.append(gittins_policy(G.truncate(G.quantile_u(0.9)), L))
        wig = rng.normal(size=L)
        wig[L // 2:] = np.inf
        pols.append(policy_from_rank(wig, "wiggly-inf"))
        pols.append(policy_from_rank(rng.normal(size=L), "noise"))
        for rho, n_busy in ((0.8, 1500), (0.98, 300)):
            for pol in pols:
                a = simulate(pol, F, rho, n_busy, seed=11, kernel="reference")
                b = simulate(pol, F, rho, n_busy, seed=11, kernel="skip")
                assert not a.overflow and not b.overflow
                assert np.array_equal(a.resp, b.resp), (F.name, pol.name, rho)
                assert np.array_equal(a.sizes_u, b.sizes_u), (F.name, pol.name, rho)
