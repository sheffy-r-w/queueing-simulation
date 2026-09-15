"""Checks of the Gittins rank computation against Definition 2.5 directly."""

import numpy as np
import pytest

from egittins.distributions import GridDistribution, one_six_fourteen, bounded_pareto
from egittins.gittins import (gittins_policy, gittins_rank_bruteforce, rank_hull_geometry,
                              SENTINEL)


def _random_pmf(seed, h=0.1):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 30))
    atoms = np.sort(rng.choice(np.arange(1, 80), size=k, replace=False))
    probs = rng.random(k) ** rng.uniform(1, 6)      # spans many orders of magnitude
    return GridDistribution(atoms, probs, h=h)


def _assert_same_policy(G, L):
    hull = gittins_policy(G, L, method="hull")
    direct = gittins_policy(G, L, method="direct")
    assert np.array_equal(hull.next_atom, direct.next_atom)
    fin = np.isfinite(direct.rank)
    assert np.array_equal(fin, np.isfinite(hull.rank))
    np.testing.assert_allclose(hull.rank[fin], direct.rank[fin], rtol=1e-12, atol=0)


@pytest.mark.parametrize("seed", range(10))
def test_hull_matches_direct_random_pmf(seed):
    G = _random_pmf(seed)
    rng = np.random.default_rng(seed + 100)
    _assert_same_policy(G, G.max_u + int(rng.integers(0, 5)))
    _assert_same_policy(G, 3)                       # table shorter than the support


def test_hull_matches_direct_paper_distributions():
    F = one_six_fourteen()
    P = bounded_pareto()
    rng = np.random.default_rng(11)
    _assert_same_policy(F, F.max_u + 1)
    _assert_same_policy(P, P.max_u + 1)
    for n in (10, 100, 1000):
        _assert_same_policy(GridDistribution.empirical(P.sample_u(rng, n), P.h), P.max_u + 1)
        _assert_same_policy(GridDistribution.empirical(F.sample_u(rng, n), F.h), F.max_u + 1)


@pytest.mark.parametrize("seed", [2, 5, 8])
def test_hull_matches_bruteforce_with_tiny_masses(seed):
    G = _random_pmf(seed)
    pol = gittins_policy(G, L=G.max_u + 1)
    for a_u in range(G.max_u):
        assert pol.rank[a_u] == pytest.approx(gittins_rank_bruteforce(G, a_u), rel=1e-10)


def test_hull_geometry_agrees_with_kernel():
    G = _random_pmf(3)
    pol = gittins_policy(G, L=G.max_u + 1)
    for a_u in (0, 5, 17, G.max_u - 1):
        geo = rank_hull_geometry(G, a_u)
        assert geo["rank"] == pytest.approx(pol.rank[a_u], rel=1e-10)
        assert geo["tangent"] in geo["hull"]
        assert np.all(np.diff(geo["F"][geo["hull"]]) > 0)


def test_deterministic_size_is_srpt():
    # Single atom at 5.0: Gittins rank = remaining size (SRPT).
    G = GridDistribution(np.array([500]), np.array([1.0]), h=0.01)
    pol = gittins_policy(G, L=501)
    ages = np.arange(500)
    assert np.allclose(pol.rank[:500], 5.0 - ages * 0.01)
    assert np.isinf(pol.rank[500])
    assert np.all(pol.next_atom[:500] == 500)
    assert pol.next_atom[500] == SENTINEL


def test_two_atoms_matches_closed_form():
    # Atoms 1 (p=0.7) and 3 (p=0.3), h=0.5 => units 2 and 6.
    G = GridDistribution(np.array([2, 6]), np.array([0.7, 0.3]), h=0.5)
    pol = gittins_policy(G, L=7)
    for a_u in range(2):          # ages 0, 0.5 (below first atom)
        a = a_u * 0.5
        expected = min((1.0 - a) / 0.7, (0.7 * 1 + 0.3 * 3) - a)
        assert pol.rank[a_u] == pytest.approx(expected)
    for a_u in range(2, 6):       # ages between atoms: remaining to 3
        assert pol.rank[a_u] == pytest.approx(3.0 - a_u * 0.5)
    assert np.isinf(pol.rank[6])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_discrete_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    atoms = np.sort(rng.choice(np.arange(1, 60), size=8, replace=False))
    probs = rng.random(8)
    G = GridDistribution(atoms, probs, h=0.1)
    pol = gittins_policy(G, L=int(atoms[-1]) + 1)
    for a_u in range(int(atoms[-1])):
        assert pol.rank[a_u] == pytest.approx(gittins_rank_bruteforce(G, a_u), rel=1e-10)


def test_rank_decreasing_between_atoms_empirical():
    F = bounded_pareto()
    rng = np.random.default_rng(7)
    G = GridDistribution.empirical(F.sample_u(rng, 200), F.h)
    pol = gittins_policy(G, L=F.max_u + 1)
    atoms = set(G.atoms_u.tolist())
    for a in range(G.max_u - 1):
        if (a + 1) not in atoms:
            assert pol.rank[a + 1] <= pol.rank[a] + 1e-12


def test_true_distributions_sane():
    for F in (one_six_fourteen(), bounded_pareto()):
        assert F.probs.sum() == pytest.approx(1.0)
        assert F.mean() > 0
        pol = gittins_policy(F, L=F.max_u + 1)
        assert np.all(np.isfinite(pol.rank[: F.max_u]))
        assert np.isinf(pol.rank[F.max_u])
    assert one_six_fourteen().mean() == pytest.approx(7.0, abs=0.1)  # mass below 0 is dropped, lifting the mean ~0.05


def test_truncation_definition():
    F = bounded_pareto()
    G = F.truncate(1000)                        # ℓ = 10.0
    assert G.max_u == 1000
    assert G.tail(9.99) == pytest.approx(F.tail(9.99))
    assert G.tail(10.0) == 0.0
    assert G.probs.sum() == pytest.approx(1.0)
