"""Drift models and the k-updating stream engine."""

import numpy as np
import pytest

from egittins.drift import (make_drift, triangle, ramp, constant, mean_preserving_weights,
                            one_way_weights, gaussian_mixture, one_way_weights, ONE_WAY_W0)
from egittins.distributions import one_six_fourteen
from egittins.gittins import fcfs_policy
from egittins.kupdating import run_stream, summarize, blocks
from egittins.simulate import simulate


def test_schedules():
    u = triangle(100)
    assert u(0) == 0 and u(50) == 1 and u(25) == 0.5 and u(75) == 0.5 and u(100) == 0
    r = ramp(11)
    assert r(0) == 0 and r(5) == 0.5 and r(10) == 1 and r(999) == 1
    assert constant(0.3)(7) == 0.3


def test_mean_preserving_family_pins_the_mean():
    means = [mean_preserving_weights(u).mean() for u in np.linspace(0, 1, 7)]
    assert max(means) - min(means) < 0.07          # discretization only (mass below 0 dropped)
    # u where t = 1/3 is the paper's 1-6-14
    t_lo, t_hi = 0.10, 0.70
    u_star = (1 / 3 - t_lo) / (t_hi - t_lo)
    F = one_six_fourteen()
    G = mean_preserving_weights(u_star)
    x = np.array([0.5, 1.0, 2.0, 6.0, 10.0, 14.0, 15.0])
    assert np.allclose(G.tail(x), F.tail(x), atol=2e-3)


def test_one_way_endpoints():
    G0 = one_way_weights(0.0)
    ref = gaussian_mixture([1, 6, 14], ONE_WAY_W0)
    assert np.array_equal(G0.atoms_u, ref.atoms_u) and np.allclose(G0.probs, ref.probs)
    assert one_way_weights(1.0).mean() < 5 < G0.mean()


def test_drift_model_caches_on_grid():
    d = make_drift("one-way", "ramp", N=1000, n_grid=11)
    assert d.grid_index(-5) == 0 and d.grid_index(0) == 0 and d.grid_index(999) == 10
    assert d.dist(100) is d.dist(105)                 # same grid point, same object
    assert d.max_u() == 1600


def test_stream_is_paired_with_direct_simulation():
    d = make_drift("one-way", "constant", u0=0.0)
    res = run_stream(d, 0.8, n_busy=40, n_warm=10, seed=3, windows=(50,))
    F = d.dist(0)
    fcfs = fcfs_policy(F.max_u + 1)
    for m in range(40):
        k = m + 10
        direct = simulate(fcfs, F, 0.8, 1, 100_000 * 3 + k)
        assert res.resp_sum[res.policies.index("fcfs"), m] == pytest.approx(direct.resp.sum())
        assert res.n_jobs[m] == len(direct.resp)


def test_stationary_large_window_tracks_genie():
    d = make_drift("mean-preserving", "constant", u0=0.5)
    results = [run_stream(d, 0.8, n_busy=800, n_warm=600, seed=s, windows=(2000,)) for s in range(3)]
    summ = {r["policy"]: r["mean"] for r in summarize(results)}
    assert summ["genie"] == 1.0
    assert summ["kupd_2000"] < 1.03
    assert summ["static"] < 1.03
    assert summ["fcfs"] > 1.03
    b = blocks(results, 200)
    assert b["genie"].shape == (4,) and np.allclose(b["genie"], 1.0)
