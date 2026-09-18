"""Features, dataset and policy wrapping for the imitation-learned rank."""

import numpy as np
import pytest

from egittins.distributions import GridDistribution, one_six_fourteen
from egittins.gittins import gittins_policy, policy_from_rank, SENTINEL
from egittins.imitation import features, make_dataset, FEATURE_NAMES, FEATURE_SETS, feature_columns
from egittins.simulate import simulate


def test_features_by_hand():
    s = np.array([2, 5, 5, 9])            # sorted sample in units
    X, m, valid = features(s, np.array([3, 9]))
    assert valid.tolist() == [True, False]
    f = dict(zip(FEATURE_NAMES, X[0]))
    # above age 3: {5, 5, 9}; excesses {2, 2, 6}, mean excess 10/3
    assert m[0] == pytest.approx(10 / 3)
    assert f["log_tail"] == pytest.approx(np.log(3 / 4))
    assert f["log_count"] == pytest.approx(np.log(3))
    assert f["log_m"] == pytest.approx(np.log(10 / 3))
    assert f["log_q50_over_m"] == pytest.approx(np.log(2 / (10 / 3)))
    assert f["log_max_over_m"] == pytest.approx(np.log(6 / (10 / 3)))
    assert f["log_gap_over_m"] == pytest.approx(np.log(2 / (10 / 3)))     # next atom 5, gap 2
    assert f["log_next_mass"] == pytest.approx(np.log(2 / 3))              # two 5s among three
    assert np.all(X[1] == 0)


def test_feature_sets_nested():
    cols = [feature_columns(fs) for fs in ("tail_only", "no_hazard", "full")]
    assert cols[0] == cols[1][: len(cols[0])] and cols[1] == cols[2][: len(cols[1])]
    assert len(cols[2]) == len(FEATURE_NAMES)


def test_dataset_targets_are_bounded():
    X, y, g = make_dataset(20, seed=1, ages_per_dist=16)
    assert X.shape[1] == len(FEATURE_NAMES) and len(y) == len(g) == len(X)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(y))
    assert y.max() <= 1e-9                    # rank <= mean excess


def test_policy_from_rank_reproduces_gittins_simulation():
    F = one_six_fourteen()
    L = F.max_u + 1
    G = GridDistribution.empirical(F.sample_u(np.random.default_rng(0), 300), F.h)
    pol = gittins_policy(G, L)
    wrapped = policy_from_rank(pol.rank, "wrapped")
    assert np.all((wrapped.next_atom >= pol.next_atom))       # only true up-jumps are kept
    a = simulate(pol, F, 0.8, 1500, 9).resp
    b = simulate(wrapped, F, 0.8, 1500, 9).resp
    assert np.array_equal(a, b)


def test_policy_from_rank_next_up_jump():
    rank = np.array([3.0, 2.0, 2.5, 1.0, 0.5, np.inf])
    pol = policy_from_rank(rank, "x")
    assert pol.next_atom.tolist() == [2, 2, 5, 5, 5, SENTINEL]


@pytest.mark.slow
def test_tiny_training_and_policy():
    torch = pytest.importorskip("torch")
    from egittins.imitation import train, nn_policy
    X, y, g = make_dataset(60, seed=2, ages_per_dist=32)
    net, hist = train(X, y, g, feature_set="full", epochs=3, batch=256, verbose=False)
    assert hist[-1]["val_mse"] < hist[0]["val_mse"] * 1.5
    F = one_six_fourteen()
    L = F.max_u + 1
    s = F.sample_u(np.random.default_rng(3), 100)
    pol = nn_policy(net, s, F.h, L)
    assert pol.L == L
    assert np.all(np.isfinite(pol.rank[: s.max()])) and np.all(np.isinf(pol.rank[s.max():]))
    assert np.all(pol.rank[: s.max()] > 0)
    r = simulate(pol, F, 0.8, 200, 1)
    assert not r.overflow
