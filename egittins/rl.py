"""
rl.py: policy-gradient training of a rank function (Section 13, optional).

The policy class is the rank table r(a) over integer ages, either from `RankNet` features of
a fixed sample window or a table of interpolated knots. Training makes it stochastic: at
every decision point (busy-period start, arrival, completion) the job to serve is drawn with
probability proportional to exp(-r(age) / tau) and runs until the next event. An episode is
one busy period; its return is minus the sum of response times, and the return-to-go from a
decision is minus the area under the number-in-system curve from that decision on. REINFORCE
with batch-normalized advantages and Adam.

Two arms: from scratch (knot table, constant initialization) and fine-tuning the imitation
net. Evaluation is deterministic (tau -> 0) in the standard SOAP simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .distributions import GridDistribution
from .gittins import Policy, policy_from_rank
from .imitation import RankNet, features, select_features, _mlp
from .simulate import simulate

BIG_RANK = 1e6      # stands in for rank ∞ (ages beyond the sample) in the stochastic policy


@njit(cache=True)
def _episodes_kernel(rank, lam, h, atoms_u, cdf, n_busy, seed, tau, cap, max_dec, max_flat):
    """Simulate n_busy busy periods under the softmax policy, recording decisions.

    Returns (seg_ptr[D+1], flat_ages[M], chosen[D], rtg[D], area_total[n_busy],
             n_jobs_total, D, M, overflow).
    """
    np.random.seed(seed)
    L = rank.shape[0]
    K = atoms_u.shape[0]
    arr = np.empty(cap, dtype=np.int64)
    size = np.empty(cap, dtype=np.int64)
    att = np.empty(cap, dtype=np.int64)
    seg_ptr = np.empty(max_dec + 1, dtype=np.int64)
    flat = np.empty(max_flat, dtype=np.int64)
    chosen = np.empty(max_dec, dtype=np.int64)
    dec_area = np.empty(max_dec)              # cumulative area at the decision, within the episode
    dec_ep = np.empty(max_dec, dtype=np.int64)
    rtg = np.empty(max_dec)
    area_total = np.zeros(n_busy)
    logits = np.empty(cap)
    D = 0
    M = 0
    n_jobs = 0
    overflow = 0
    t = np.int64(0)
    seg_ptr[0] = 0

    for ep in range(n_busy):
        u = np.random.random()
        idx = np.searchsorted(cdf, u, side='right')
        if idx >= K:
            idx = K - 1
        arr[0] = t
        size[0] = atoms_u[idx]
        att[0] = 0
        m = 1
        x = np.random.exponential(1.0 / lam)
        next_arr = t + max(np.int64(1), np.int64(np.floor(x / h + 0.5)))
        area = 0.0
        ep_first_dec = D
        while m > 0:
            # ---- decision: sample a job with P ∝ exp(-rank/tau)
            if D >= max_dec or M + m > max_flat:
                overflow = 1
                break
            best = -1e300
            for k in range(m):
                a = att[k]
                r = rank[a] if a < L else BIG_RANK
                if r > BIG_RANK:
                    r = BIG_RANK
                logits[k] = -r / tau
                if logits[k] > best:
                    best = logits[k]
            tot = 0.0
            for k in range(m):
                logits[k] = np.exp(logits[k] - best)
                tot += logits[k]
            u = np.random.random() * tot
            s = m - 1
            acc = 0.0
            for k in range(m):
                acc += logits[k]
                if u < acc:
                    s = k
                    break
            for k in range(m):
                flat[M + k] = att[k]
            chosen[D] = s
            dec_area[D] = area
            dec_ep[D] = ep
            M += m
            D += 1
            seg_ptr[D] = M
            # ---- run s until completion or next arrival
            dt = size[s] - att[s]
            if next_arr - t < dt:
                dt = next_arr - t
            area += m * dt * h
            t += dt
            att[s] += dt
            if att[s] == size[s]:
                n_jobs += 1
                m -= 1
                if s != m:
                    arr[s] = arr[m]
                    size[s] = size[m]
                    att[s] = att[m]
            while next_arr == t:
                if m >= cap:
                    overflow = 1
                    break
                u = np.random.random()
                idx = np.searchsorted(cdf, u, side='right')
                if idx >= K:
                    idx = K - 1
                arr[m] = t
                size[m] = atoms_u[idx]
                att[m] = 0
                m += 1
                x = np.random.exponential(1.0 / lam)
                next_arr = t + max(np.int64(1), np.int64(np.floor(x / h + 0.5)))
            if overflow == 1:
                break
        if overflow == 1:
            break
        area_total[ep] = area
        for d in range(ep_first_dec, D):
            rtg[d] = -(area - dec_area[d])
        t = next_arr
    return seg_ptr[: D + 1], flat[:M], chosen[:D], rtg[:D], area_total, n_jobs, D, M, overflow


@dataclass
class RLConfig:
    tau: float = 0.5
    episodes_per_iter: int = 256
    iters: int = 600
    lr: float = 1e-3
    clip: float = 1.0
    cosine: bool = True
    knots: int = 0          # scratch arm: 0 = one parameter per age, else this many interpolated knots
    entropy: float = 0.0
    eval_every: int = 25
    eval_busy: int = 3000


def train_rl(F: GridDistribution, sample_u: np.ndarray, rho: float, init: RankNet | None,
             feature_set: str, seed: int, cfg: RLConfig, verbose: bool = True):
    """REINFORCE on busy-period episodes. Returns (params, curve rows, final Policy).

    init = a RankNet: fine-tune it (the rank is mean excess × exp(net(features))).
    init = None: "from scratch", a table of log ranks over integer ages (one
    parameter per age, or cfg.knots linearly interpolated knots), initialized
    constant (rank ≡ E[S]) so the deterministic policy starts as FCFS.
    """
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    L = F.max_u + 1
    h = F.h
    lam = rho / F.mean()
    s = np.sort(np.asarray(sample_u, dtype=np.int64))
    a_max = min(int(s[-1]), L)
    ages = np.arange(a_max)
    X_full, m_units, valid = features(s, ages)
    if init is None:
        n_knots = cfg.knots if cfg.knots > 0 else a_max
        theta = torch.zeros(n_knots, dtype=torch.float64, requires_grad=True)
        scale = float(s.mean() * h)
        params = [theta]
        net = None
        # linear interpolation of the knot values onto integer ages
        pos = torch.linspace(0, n_knots - 1, a_max, dtype=torch.float64)
        lo = torch.clamp(pos.floor().long(), max=n_knots - 2)
        frac = pos - lo.double()

        def table():
            logr = theta[lo] * (1 - frac) + theta[lo + 1] * frac if n_knots > 1 else theta.expand(a_max)
            return scale * torch.exp(logr)
    else:
        assert init.feature_set == feature_set
        model = init.module()
        model.train()
        net = RankNet(feature_set, init.mu, init.sd, {}, init.hidden, init.depth)
        X = select_features(X_full, feature_set)
        Xt = torch.as_tensor((X - init.mu) / init.sd, dtype=torch.float32)
        mt = torch.as_tensor(m_units * h, dtype=torch.float64)
        params = list(model.parameters())

        def table():
            return mt * torch.exp(model(Xt).squeeze(-1).double())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iters) if cfg.cosine else None
    true_ref = simulate(_gittins_ref(F), F, rho, cfg.eval_busy, seed=777).mean_response_time

    def full_rank(tab_np):
        r = np.full(L, np.inf)
        r[:a_max] = tab_np
        return r

    curve = []
    rng = np.random.default_rng(seed)
    for it in range(cfg.iters + 1):
        if it % cfg.eval_every == 0:
            with torch.no_grad():
                pol = policy_from_rank(full_rank(table().numpy()), "rl")
            mrt = simulate(pol, F, rho, cfg.eval_busy, seed=777).mean_response_time
            curve.append(dict(iter=it, ratio=mrt / true_ref, seed=seed))
            if verbose:
                print(f"    iter {it:4d}  MRT/true Gittins = {mrt / true_ref:.4f}", flush=True)
            if it == cfg.iters:
                break
        tab = table()
        rank_np = full_rank(tab.detach().numpy())
        ep_seed = int(rng.integers(0, 2**31 - 1))
        out = _episodes_kernel(rank_np, lam, h, F.atoms_u, F.cdf, cfg.episodes_per_iter, ep_seed,
                               cfg.tau, 20000, 400_000, 4_000_000)
        seg_ptr, flat, chosen, rtg, area_total, n_jobs, D, M, overflow = out
        assert not overflow
        seg_id = np.repeat(np.arange(D), np.diff(seg_ptr))
        flat_t = torch.as_tensor(flat)
        seg_t = torch.as_tensor(seg_id)
        big = torch.full((L,), BIG_RANK, dtype=torch.float64)
        big[:a_max] = tab
        logits = -torch.clamp(big[flat_t], max=BIG_RANK) / cfg.tau
        seg_max = torch.full((D,), -1e300, dtype=torch.float64).scatter_reduce(0, seg_t, logits, reduce="amax")
        ex = torch.exp(logits - seg_max[seg_t])
        seg_sum = torch.zeros(D, dtype=torch.float64).scatter_add(0, seg_t, ex)
        logp_all = logits - (seg_max + torch.log(seg_sum))[seg_t]
        chosen_flat = torch.as_tensor(seg_ptr[:-1] + chosen)
        logp = logp_all[chosen_flat]
        adv = torch.as_tensor(rtg)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = -(adv * logp).mean()
        if cfg.entropy > 0:
            ent = -(torch.exp(logp_all) * logp_all)
            loss = loss - cfg.entropy * torch.zeros(D, dtype=torch.float64).scatter_add(0, seg_t, ent).mean()
        opt.zero_grad()
        loss.backward()
        if cfg.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
        opt.step()
        if sched is not None:
            sched.step()
    if net is not None:
        net.state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    with torch.no_grad():
        pol = policy_from_rank(full_rank(table().numpy()), "rl")
    return net if net is not None else theta.detach().numpy(), curve, pol


def _gittins_ref(F: GridDistribution) -> Policy:
    from .gittins import gittins_policy
    return gittins_policy(F, F.max_u + 1)
