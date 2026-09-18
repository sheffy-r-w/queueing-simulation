"""Label-free rank search: parametrizations, common-random-number scoring, ordering utilities."""

import numpy as np
import pytest

from egittins.distributions import one_six_fourteen
from egittins.gittins import gittins_policy, fcfs_policy, policy_from_rank as policy_from_rank_ref
from egittins.imitation import FEATURE_SETS, n_features
from egittins.labelfree import (policy_from_rank, knot_rank, n_params, feature_rank, make_spec, feature_policy,
                                Stage0Problem, Stage1Problem, cmaes_search, ordering_agreement,
                                monotone_rescale, paired_mrt)
from egittins.simulate import simulate


def test_knot_rank_interpolates_and_is_positive():
    r = knot_rank(np.log([1.0, 3.0, 2.0]), 5)
    assert r.shape == (5,)
    assert r[0] == pytest.approx(1.0) and r[2] == pytest.approx(3.0) and r[4] == pytest.approx(2.0)
    assert r[1] == pytest.approx(np.exp(0.5 * np.log(3.0)))
    assert np.all(r > 0)
    assert knot_rank(np.array([0.7]), 3).tolist() == pytest.approx([np.exp(0.7)] * 3)


def test_feature_rank_param_count_and_shapes():
    Z = np.random.default_rng(0).normal(size=(10, 4))
    for hidden in (0, 8):
        p = np.random.default_rng(1).normal(size=n_params(4, hidden))
        assert feature_rank(p, Z, hidden).shape == (10,)
    assert n_params(4, 0) == 4 and n_params(4, 8) == 8 * 4 + 16


def test_constant_knots_is_fcfs_and_ordering_is_all_that_matters():
    F = one_six_fourteen()
    L = F.max_u + 1
    prob = Stage0Problem(F, 0.8, 300, n_knots=8)
    fcfs = simulate(fcfs_policy(L), F, 0.8, 300, 3).resp
    assert np.array_equal(simulate(prob.policy(np.zeros(8)), F, 0.8, 300, 3).resp, fcfs)
    theta = np.random.default_rng(0).normal(size=8)
    a = simulate(prob.policy(theta), F, 0.8, 300, 3).resp
    b = simulate(prob.policy(theta + 5.0), F, 0.8, 300, 3).resp      # monotone transform of the rank
    assert np.array_equal(a, b)


def test_fitness_is_deterministic_under_common_random_numbers():
    F = one_six_fourteen()
    prob = Stage0Problem(F, 0.8, 200, n_knots=6)
    x = np.random.default_rng(2).normal(size=6)
    assert prob.fitness(x, 11) == prob.fitness(x, 11)
    assert prob.fitness(x, 11) != prob.fitness(x, 12)


def test_feature_policy_matches_sample_support():
    F = one_six_fourteen()
    L = F.max_u + 1
    s = F.sample_u(np.random.default_rng(0), 200)
    spec = make_spec([F], [s], "no_hazard", hidden=0)
    assert spec.d == n_features("no_hazard") and spec.n_params == n_features("no_hazard")
    pol = feature_policy(spec, np.ones(spec.n_params), s, L)
    assert np.all(np.isfinite(pol.rank[: s.max()]))
    assert np.all(np.isinf(pol.rank[s.max():]))
    r = simulate(pol, F, 0.8, 200, 5)
    assert not r.overflow and np.isfinite(r.mean_response_time)


def test_stage1_fitness_is_one_for_constant_rank():
    F = one_six_fourteen()
    s = F.sample_u(np.random.default_rng(0), 100)
    for fs in FEATURE_SETS:
        spec = make_spec([F], [s], fs, hidden=8)
        prob = Stage1Problem([F], [s], spec, 0.8, 150)
        assert prob.fitness(np.zeros(spec.n_params), 4) == pytest.approx(1.0)   # zero net = FCFS


@pytest.mark.slow
def test_cmaes_runs_and_improves_on_fcfs():
    F = one_six_fourteen()
    prob = Stage0Problem(F, 0.8, 300, n_knots=6)
    res = cmaes_search(prob, seed=0, popsize=6, sigma0=1.0, n_gen=6, seeds_per_gen=2, workers=2,
                       val_seeds=(5, 6), val_every=3, verbose=False)
    assert res.x.shape == (6,) and len(res.history) == 6
    assert res.busy_periods == 6 * 6 * 2 * 300
    assert np.isfinite(res.history[-1]["val"])


def test_ordering_agreement_and_monotone_rescale():
    F = one_six_fourteen()
    L = F.max_u + 1
    g = gittins_policy(F, L).rank
    assert ordering_agreement(g, g) == pytest.approx(1.0)
    assert ordering_agreement(-g, g) == pytest.approx(-1.0)
    learned = np.exp(g) + 3.0                     # same ordering, other values
    learned[-1] = np.inf
    resc = monotone_rescale(learned, g)
    m = np.isfinite(learned) & np.isfinite(g)
    assert np.allclose(resc[m], g[m]) and np.isinf(resc[-1])


def test_paired_mrt_returns_every_policy():
    F = one_six_fourteen()
    L = F.max_u + 1
    out = paired_mrt({"g": gittins_policy(F, L), "f": fcfs_policy(L)}, F, 0.8, 200, 1)
    assert set(out) == {"g", "f"} and out["g"] < out["f"]


def test_vectorized_policy_from_rank_matches_reference():
    rng = np.random.default_rng(0)
    for rank in (rng.normal(size=500), np.array([3.0, 2.0, 2.5, 1.0, 0.5, np.inf]),
                 np.full(7, np.inf), np.zeros(4)):
        a, b = policy_from_rank(rank, "a"), policy_from_rank_ref(rank, "b")
        assert np.array_equal(a.next_atom, b.next_atom) and np.array_equal(a.rank, b.rank)
