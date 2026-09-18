"""
imitation.py: a rank function learned by regressing onto empirical Gittins ranks.

An MLP maps features of the conditional excess sample (S - a | S > a) at age a to
log(rank / mean excess), where the target rank is the exact empirical Gittins rank of the
same sample. The trained network can be tabulated over integer ages for any sample window
and used as a SOAP policy.

Features at age a, from the sorted sample s (units), with c = #{s > a} and mean excess
m = mean(s - a | s > a), in three nested sets (FEATURE_SETS):
    tail_only:  log G_bar(a), log c, log m
    no_hazard:  + log(q_p / m) for p in {0.1, 0.25, 0.5, 0.75, 0.9}, log(max excess / m)
    full:       + log(gap to the next sample atom / m), log(mass at that atom / G_bar(a))
Scale enters only through log m (in units).

Training distributions are random grid distributions (Gaussian mixtures, bounded Pareto,
lognormal, Weibull, few-atom discrete) with the paper's 1-6-14 and Pareto(2, 1.2, 500)
parameters excluded, and random sample sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, lognorm, weibull_min

from .distributions import GridDistribution, bounded_pareto
from .gittins import Policy, gittins_policy, plcfs_policy, policy_from_rank

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
FEATURE_NAMES = (["log_tail", "log_count", "log_m"]
                 + [f"log_q{int(p * 100)}_over_m" for p in QUANTILES] + ["log_max_over_m"]
                 + ["log_gap_over_m", "log_next_mass"])
# Nested feature sets for the ablation
FEATURE_SETS = {
    "tail_only": FEATURE_NAMES[:3],        # survivor mass, sample count, mean excess
    "no_hazard": FEATURE_NAMES[:-2],       # + conditional-excess quantiles
    "full": FEATURE_NAMES,                 # + hazard: gap to next atom, mass at it
}


def feature_columns(feature_set: str) -> list[int]:
    return [FEATURE_NAMES.index(f) for f in FEATURE_SETS[feature_set]]


def n_features(feature_set: str) -> int:
    return len(FEATURE_SETS[feature_set])


# ------------------------------------------------------------------ features

def features(sorted_u: np.ndarray, ages_u: np.ndarray):
    """Full feature matrix (all of FEATURE_NAMES) for ages (units) from a sorted sample (units).

    Returns (X [n_ages, d], m [n_ages] mean excess in units, valid [n_ages]);
    rows with valid == False (no sample above a) are zero.
    """
    s = np.asarray(sorted_u, dtype=np.int64)
    a = np.asarray(ages_u, dtype=np.int64)
    n = len(s)
    idx = np.searchsorted(s, a, side="right")          # samples <= a
    c = n - idx
    valid = c > 0
    X = np.zeros((len(a), len(FEATURE_NAMES)))
    m = np.zeros(len(a))
    if not valid.any():
        return X, m, valid
    av, iv, cv = a[valid], idx[valid], c[valid]
    cs = np.concatenate([[0], np.cumsum(s, dtype=np.float64)])
    sum_above = cs[n] - cs[iv]
    mv = (sum_above - cv * av) / cv
    cols = [np.log(cv / n), np.log(cv), np.log(mv)]
    for p in QUANTILES:
        pos = iv + np.floor(p * (cv - 1)).astype(np.int64)
        cols.append(np.log((s[pos] - av) / mv))
    cols.append(np.log((s[n - 1] - av) / mv))
    nxt = s[iv]
    gap = nxt - av
    mass = np.searchsorted(s, nxt, side="right") - iv
    cols.append(np.log(gap / mv))
    cols.append(np.log(mass / cv))
    X[valid] = np.stack(cols, axis=1)
    m[valid] = mv
    return X, m, valid


# ------------------------------------------------- random training distributions

HOLD_OUT_MIX = (np.array([1.0, 6.0, 14.0]), 0.5)      # the paper's 1-6-14
HOLD_OUT_PARETO = (2.0, 1.2)                          # the paper's (x_m, α)


def _mixture(rng, h):
    k = int(rng.integers(1, 5))
    means = np.sort(rng.uniform(0.5, 15.0, size=k))
    sds = rng.uniform(0.1, 2.0, size=k)
    w = rng.dirichlet(np.ones(k))
    if k == 3 and np.all(np.abs(means - HOLD_OUT_MIX[0]) < 0.5) and np.all(np.abs(sds - HOLD_OUT_MIX[1]) < 0.15):
        return None

    def cdf(x):
        x = np.asarray(x, dtype=np.float64)[..., None]
        return (norm.cdf((x - means) / sds) * w).sum(axis=-1)

    return GridDistribution.from_continuous_cdf(cdf, h, 16.0, h, "mixture")


def _pareto(rng, h):
    x_m = rng.uniform(0.5, 5.0)
    alpha = rng.uniform(0.7, 3.0)
    upper = float(np.exp(rng.uniform(np.log(50.0), np.log(1000.0))))
    if abs(alpha - HOLD_OUT_PARETO[1]) < 0.1 and abs(x_m - HOLD_OUT_PARETO[0]) < 0.3:
        return None
    return bounded_pareto(h=h, x_m=x_m, alpha=alpha, upper=upper)


def _lognormal(rng, h):
    mu, sigma = rng.uniform(-0.5, 2.5), rng.uniform(0.2, 1.5)
    return GridDistribution.from_continuous_cdf(lambda x: lognorm.cdf(x, s=sigma, scale=np.exp(mu)),
                                               h, 500.0, h, "lognormal")


def _weibull(rng, h):
    scale, shape = rng.uniform(1.0, 10.0), rng.uniform(0.5, 2.5)
    return GridDistribution.from_continuous_cdf(lambda x: weibull_min.cdf(x, c=shape, scale=scale),
                                               h, 500.0, h, "weibull")


def _discrete(rng, h):
    k = int(rng.integers(2, 7))
    atoms = np.unique(np.round(np.exp(rng.uniform(np.log(0.1), np.log(50.0), size=k)) / h).astype(np.int64))
    atoms = atoms[atoms > 0]
    if len(atoms) < 2:
        return None
    return GridDistribution(atoms, rng.dirichlet(0.7 * np.ones(len(atoms))), h, "discrete")


FAMILIES = [_mixture, _pareto, _lognormal, _weibull, _discrete]
FAMILY_WEIGHTS = np.array([0.35, 0.2, 0.15, 0.15, 0.15])


def random_distribution(rng: np.random.Generator, h: float = 0.01) -> GridDistribution:
    while True:
        fam = FAMILIES[rng.choice(len(FAMILIES), p=FAMILY_WEIGHTS)]
        G = fam(rng, h)
        if G is not None and len(G.atoms_u) >= 1:
            return G


def make_dataset(n_dists: int, seed: int = 0, ages_per_dist: int = 64, h: float = 0.01,
                 n_range=(30, 3000)):
    """Random (distribution, sample, ages) triples with exact empirical-Gittins targets.

    Returns X_full (all features incl. hazard), y = log(r / m), group id per row.
    Ages: half uniform over [0, max sample), half just below random sample atoms.
    """
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    for g in range(n_dists):
        F = random_distribution(rng, h)
        n = int(np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1]))))
        s = np.sort(F.sample_u(rng, n))
        G = GridDistribution.empirical(s, h)
        L = int(s[-1]) + 1
        rank = gittins_policy(G, L).rank
        half = ages_per_dist // 2
        a1 = rng.integers(0, s[-1], size=half)
        atoms = G.atoms_u
        pick = atoms[rng.integers(0, len(atoms), size=ages_per_dist - half)]
        a2 = pick - rng.integers(1, 21, size=len(pick))
        ages = np.unique(np.clip(np.concatenate([a1, a2]), 0, s[-1] - 1))
        X, m, valid = features(s, ages)
        r = rank[ages]
        ok = valid & np.isfinite(r) & (r > 0)
        Xs.append(X[ok])
        ys.append(np.log(r[ok] / (m[ok] * h)))
        gs.append(np.full(ok.sum(), g))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


# ------------------------------------------------------------------- network

def _mlp(d_in: int, hidden: int, depth: int):
    from torch import nn
    layers, w = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(w, hidden), nn.SiLU()]
        w = hidden
    layers.append(nn.Linear(w, 1))
    return nn.Sequential(*layers)


@dataclass
class RankNet:
    """A trained MLP with input standardization on one of the FEATURE_SETS."""
    feature_set: str
    mu: np.ndarray
    sd: np.ndarray
    state: dict
    hidden: int = 128
    depth: int = 3

    def module(self):
        import torch
        net = _mlp(n_features(self.feature_set), self.hidden, self.depth)
        net.load_state_dict({k: torch.as_tensor(v) for k, v in self.state.items()})
        net.eval()
        return net

    def predict(self, X_full: np.ndarray) -> np.ndarray:
        """log(r / m) for full feature rows (the net selects its own columns)."""
        import torch
        net = self.module()
        X = select_features(X_full, self.feature_set)
        with torch.no_grad():
            z = torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32)
            return net(z).squeeze(-1).numpy().astype(np.float64)

    def save(self, path: str):
        import torch
        torch.save(dict(feature_set=self.feature_set, mu=self.mu, sd=self.sd, state=self.state,
                        hidden=self.hidden, depth=self.depth), path)

    @classmethod
    def load(cls, path: str) -> "RankNet":
        import torch
        d = torch.load(path, weights_only=False)
        return cls(**d)


def select_features(X_full: np.ndarray, feature_set: str) -> np.ndarray:
    return X_full[:, feature_columns(feature_set)]


def train(X_full: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_set: str = "full",
          seed: int = 0, epochs: int = 40, batch: int = 2048, lr: float = 2e-3, hidden: int = 128,
          depth: int = 3, val_frac: float = 0.1, verbose: bool = True):
    """Train on a group-wise split; returns (RankNet, history as a list of dicts)."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = select_features(X_full, feature_set)
    ug = np.unique(groups)
    val_groups = set(rng.choice(ug, size=int(val_frac * len(ug)), replace=False).tolist())
    is_val = np.array([g in val_groups for g in groups])
    mu, sd = X[~is_val].mean(0), X[~is_val].std(0) + 1e-8
    Xt = torch.as_tensor((X[~is_val] - mu) / sd, dtype=torch.float32)
    yt = torch.as_tensor(y[~is_val], dtype=torch.float32)
    Xv = torch.as_tensor((X[is_val] - mu) / sd, dtype=torch.float32)
    yv = torch.as_tensor(y[is_val], dtype=torch.float32)

    net = RankNet(feature_set, mu, sd, {}, hidden, depth)
    model = _mlp(X.shape[1], hidden, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    N = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, batch):
            b = perm[i: i + batch]
            loss = nn.functional.mse_loss(model(Xt[b]).squeeze(-1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).squeeze(-1)
            val = nn.functional.mse_loss(pv, yv).item()
            r2 = 1 - val / yv.var().item()
        history.append(dict(epoch=ep + 1, train_mse=tot / N, val_mse=val, val_r2=r2, feature_set=feature_set))
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            print(f"  [{feature_set}] epoch {ep + 1:3d}  train {tot / N:.4f}  val {val:.4f}  R² {r2:.3f}",
                  flush=True)
    net.state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return net, history


# -------------------------------------------------------------------- policy

def nn_policy(net: RankNet, sizes_u: np.ndarray, h: float, L: int, name: str = "NN") -> Policy:
    """Tabulate the learned rank over ages [0, L) for a sample window (units).

    Ages at or beyond the largest sample get rank ∞ (PLCFS fallback), exactly
    as empirical Gittins does.
    """
    s = np.sort(np.asarray(sizes_u, dtype=np.int64))
    if len(s) == 0:
        return plcfs_policy(L)
    a_max = min(int(s[-1]), L)
    ages = np.arange(a_max)
    X, m, valid = features(s, ages)
    rank = np.full(L, np.inf)
    if valid.any():
        rank[ages[valid]] = m[valid] * h * np.exp(net.predict(X[valid]))
    return policy_from_rank(rank, name)
