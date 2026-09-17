"""
labelfree.py — a rank function found by derivative-free search, without Gittins labels.

The imitation policy (`imitation.py`) regresses onto exact Gittins ranks, so it
cannot say what a scheduler needs to *know*: the answer is supplied. Here the
only training signal is the simulated mean response time. A candidate rank
function is tabulated over integer ages, dropped into the SOAP simulator, and
scored; CMA-ES (a derivative-free evolution strategy) proposes the next
generation. Every candidate in a generation is scored on the SAME simulation
seeds (common random numbers), so candidates are compared on identical arrival
and size streams and the ranking within a generation is nearly noise-free;
each generation draws fresh seeds so no candidate can overfit one stream.
Only the ORDERING of ranks across ages affects the policy, so all rank
parametrizations here are read modulo monotone transforms.

Two policy classes:

  * `knot_rank`      Stage 0 (positive control): log-rank at K knots equally
                     spaced over age, linear interpolation. One distribution.
  * `feature_rank`   Stage 1: a linear map or a one-hidden-layer (SiLU) net from
                     the standardized feature tiers of `imitation.FEATURE_SETS`
                     (tail_only / no_hazard / full) to a rank, tabulated over
                     integer ages from a sample window exactly as `nn_policy`.
                     Trained across many random distributions so the features
                     must generalize; loss = mean over distributions of
                     MRT / MRT(FCFS) on the same seed. Gittins is never used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np

from .distributions import GridDistribution
from .gittins import Policy, fcfs_policy, plcfs_policy, SENTINEL
from .imitation import features, feature_columns, n_features
from .simulate import simulate


# --------------------------------------------------------------- parametrizations

def policy_from_rank(rank: np.ndarray, name: str) -> Policy:
    """Vectorized twin of `gittins.policy_from_rank` (identical output): next_atom[a] is
    the first age a' > a at which the table rises, or SENTINEL."""
    rank = np.asarray(rank, dtype=np.float64)
    L = len(rank)
    ups = np.flatnonzero(rank[1:] > rank[:-1]) + 1
    padded = np.concatenate([ups, [SENTINEL]])
    nxt = padded[np.searchsorted(ups, np.arange(L), side="right")]
    return Policy(rank, nxt.astype(np.int64), name)


def knot_rank(theta: np.ndarray, L: int) -> np.ndarray:
    """Rank over ages [0, L): exp of the linear interpolation of log-rank knots
    `theta` placed at equally spaced ages 0, ..., L-1 (same class as the RL scratch arm)."""
    theta = np.asarray(theta, dtype=np.float64)
    n = len(theta)
    if n == 1:
        return np.full(L, np.exp(theta[0]))
    pos = np.linspace(0.0, n - 1, L)
    lo = np.minimum(np.floor(pos).astype(np.int64), n - 2)
    fr = pos - lo
    return np.exp(theta[lo] * (1.0 - fr) + theta[lo + 1] * fr)


def n_params(d: int, hidden: int) -> int:
    """Linear (hidden = 0): d weights, no bias (a bias does not change the ordering).
    One hidden layer: hidden×d weights + hidden biases + hidden output weights."""
    return d if hidden == 0 else hidden * d + 2 * hidden


def feature_rank(params: np.ndarray, Z: np.ndarray, hidden: int) -> np.ndarray:
    """Rank for standardized feature rows Z [n, d]. Lower rank = higher priority."""
    p = np.asarray(params, dtype=np.float64)
    d = Z.shape[1]
    if hidden == 0:
        return Z @ p[:d]
    W = p[: hidden * d].reshape(hidden, d)
    b = p[hidden * d: hidden * d + hidden]
    v = p[hidden * d + hidden: hidden * d + 2 * hidden]
    H = Z @ W.T + b
    H = H / (1.0 + np.exp(-H))          # SiLU
    return H @ v


@dataclass
class FeatureSpec:
    """Which feature tier, which architecture, and the standardization used."""
    feature_set: str
    hidden: int
    mu: np.ndarray
    sd: np.ndarray

    @property
    def d(self) -> int:
        return n_features(self.feature_set)

    @property
    def n_params(self) -> int:
        return n_params(self.d, self.hidden)

    def standardize(self, X_full: np.ndarray) -> np.ndarray:
        return (X_full[:, feature_columns(self.feature_set)] - self.mu) / self.sd


def sample_features(sizes_u: np.ndarray, L: int):
    """(ages, X_full, valid) for a sample window, over ages below min(max sample, L)."""
    s = np.sort(np.asarray(sizes_u, dtype=np.int64))
    a_max = min(int(s[-1]), L)
    ages = np.arange(a_max)
    X, m, valid = features(s, ages)
    return ages, X, valid


def rank_from_features(spec: FeatureSpec, params: np.ndarray, ages: np.ndarray, Z_valid: np.ndarray,
                       valid: np.ndarray, L: int) -> np.ndarray:
    rank = np.full(L, np.inf)
    if valid.any():
        rank[ages[valid]] = feature_rank(params, Z_valid, spec.hidden)
    return rank


def feature_policy(spec: FeatureSpec, params: np.ndarray, sizes_u: np.ndarray, L: int,
                   name: str = "label-free") -> Policy:
    """Tabulate the learned rank over ages [0, L) for a sample window (units);
    ages at or beyond the largest sample get rank ∞ (PLCFS fallback), as empirical Gittins does."""
    s = np.asarray(sizes_u, dtype=np.int64)
    if len(s) == 0:
        return plcfs_policy(L)
    ages, X, valid = sample_features(s, L)
    Z = spec.standardize(X[valid]) if valid.any() else np.zeros((0, spec.d))
    return policy_from_rank(rank_from_features(spec, params, ages, Z, valid, L), name)


# ------------------------------------------------------------------- problems

@dataclass
class Stage0Problem:
    """One distribution; the objective is the mean response time itself."""
    F: GridDistribution
    rho: float
    n_busy: int
    n_knots: int

    @property
    def n_params(self) -> int:
        return self.n_knots

    def policy(self, x: np.ndarray) -> Policy:
        return policy_from_rank(knot_rank(x, self.F.max_u + 1), "knots")

    def fitness(self, x: np.ndarray, seed: int) -> float:
        r = simulate(self.policy(x), self.F, self.rho, self.n_busy, int(seed))
        return r.mean_response_time

    @property
    def busy_per_fitness(self) -> int:
        return self.n_busy


@dataclass
class Stage1Problem:
    """Many training distributions, one sample window each; the objective is the
    mean over distributions of MRT / MRT(FCFS) on the same seed."""
    dists: list
    samples: list
    spec: FeatureSpec
    rho: float
    n_busy: int
    _tabs: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self._tabs:
            for F, s in zip(self.dists, self.samples):
                L = F.max_u + 1
                ages, X, valid = sample_features(s, L)
                Z = self.spec.standardize(X[valid])
                self._tabs.append((ages, Z, valid, L))

    @property
    def n_params(self) -> int:
        return self.spec.n_params

    def policy(self, x: np.ndarray, k: int) -> Policy:
        ages, Z, valid, L = self._tabs[k]
        return policy_from_rank(rank_from_features(self.spec, x, ages, Z, valid, L), "label-free")

    def fitness(self, x: np.ndarray, seed: int) -> float:
        tot = 0.0
        for k, F in enumerate(self.dists):
            L = self._tabs[k][3]
            s = int(seed) + 7919 * k
            mrt = simulate(self.policy(x, k), F, self.rho, self.n_busy, s).mean_response_time
            ref = simulate(fcfs_policy(L), F, self.rho, self.n_busy, s).mean_response_time
            tot += mrt / ref
        return tot / len(self.dists)

    @property
    def busy_per_fitness(self) -> int:
        return self.n_busy * len(self.dists)


def make_spec(dists, samples, feature_set: str, hidden: int) -> FeatureSpec:
    """Standardization from the training windows' own feature rows (no labels involved)."""
    rows = []
    for F, s in zip(dists, samples):
        ages, X, valid = sample_features(s, F.max_u + 1)
        rows.append(X[valid][:, feature_columns(feature_set)])
    R = np.concatenate(rows)
    return FeatureSpec(feature_set, hidden, R.mean(0), R.std(0) + 1e-8)


# ---------------------------------------------------------------------- CMA-ES

_PROBLEM = None


def _init(problem):
    global _PROBLEM
    _PROBLEM = problem


def _task(args):
    x, seed = args
    return _PROBLEM.fitness(x, seed)


@dataclass
class SearchResult:
    x: np.ndarray               # the final distribution mean of CMA-ES
    history: list               # per-generation rows
    busy_periods: int           # simulated busy periods spent on fitness evaluations
    config: dict


def cmaes_search(problem, seed: int, popsize: int = 16, sigma0: float = 1.0, n_gen: int = 200,
                 seeds_per_gen: int = 4, workers: int = 12, x0=None, val_seeds=(900_001, 900_002, 900_003, 900_004),
                 val_every: int = 10, verbose: bool = True, pool: Pool | None = None) -> SearchResult:
    """CMA-ES on `problem.fitness` with common random numbers.

    Each generation draws `seeds_per_gen` fresh seeds; every candidate is scored on
    all of them and the mean is its fitness. `val_seeds` are fixed seeds on which
    the current search mean is scored every `val_every` generations (a learning
    curve on the training problem, never on held-out distributions).
    """
    import cma
    n = problem.n_params
    x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64)
    opts = {"popsize": popsize, "seed": seed + 1, "verbose": -9,
            "tolfun": 0, "tolfunhist": 0, "tolx": 0, "tolflatfitness": 10**9, "tolstagnation": 10**9}
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    rng = np.random.default_rng(1_000_003 * seed + 12345)
    history, used = [], 0
    own_pool = pool is None
    if own_pool:
        pool = Pool(workers, initializer=_init, initargs=(problem,))
    try:
        for gen in range(n_gen):
            X = es.ask()
            seeds = rng.integers(0, 2**30, size=seeds_per_gen)
            vals = pool.map(_task, [(x, int(s)) for x in X for s in seeds], chunksize=1)
            f = np.asarray(vals).reshape(len(X), seeds_per_gen).mean(1)
            es.tell(X, f.tolist())
            used += len(X) * seeds_per_gen * problem.busy_per_fitness
            row = dict(gen=gen + 1, best=float(f.min()), pop_mean=float(f.mean()), sigma=float(es.sigma),
                       seed=seed)
            if (gen + 1) % val_every == 0 or gen == n_gen - 1 or gen == 0:
                v = pool.map(_task, [(np.array(es.mean), int(s)) for s in val_seeds], chunksize=1)
                row["val"] = float(np.mean(v))
                if verbose:
                    print(f"    gen {gen + 1:4d}  pop best {f.min():.4f}  mean {f.mean():.4f}  "
                          f"val(mean) {row['val']:.4f}  sigma {es.sigma:.3f}", flush=True)
            history.append(row)
    finally:
        if own_pool:
            pool.close()
            pool.join()
    cfg = dict(popsize=popsize, sigma0=sigma0, n_gen=n_gen, seeds_per_gen=seeds_per_gen, n_params=n,
               busy_per_fitness=problem.busy_per_fitness, optimizer_seed=seed)
    return SearchResult(np.array(es.mean), history, used, cfg)


# ------------------------------------------------------------------ evaluation

def paired_mrt(policies: dict, F: GridDistribution, rho: float, n_busy: int, seed: int) -> dict:
    """Mean response time of every policy on the same seed (common random numbers)."""
    out = {}
    for name, pol in policies.items():
        r = simulate(pol, F, rho, n_busy, seed)
        assert not r.overflow, name
        out[name] = r.mean_response_time
    return out


def mean_ci(x, level: float = 0.95):
    """(mean, half-width of the t confidence interval)."""
    from scipy import stats
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))


def ordering_agreement(learned: np.ndarray, reference: np.ndarray) -> float:
    """Spearman rank correlation over ages where both ranks are finite."""
    from scipy.stats import spearmanr
    m = np.isfinite(learned) & np.isfinite(reference)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(learned[m], reference[m]).correlation)


def monotone_rescale(learned: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Quantile-match `learned` onto `reference` over the ages where both are finite:
    the result has the ordering of `learned` and the values of `reference`, so the two
    curves coincide exactly when the orderings agree. Non-finite ages stay ∞."""
    out = np.full_like(reference, np.inf, dtype=np.float64)
    m = np.isfinite(learned) & np.isfinite(reference)
    if m.sum() == 0:
        return out
    order = np.argsort(learned[m], kind="stable")
    ranks = np.empty(m.sum(), dtype=np.int64)
    ranks[order] = np.arange(m.sum())
    out[m] = np.sort(reference[m])[ranks]
    return out
