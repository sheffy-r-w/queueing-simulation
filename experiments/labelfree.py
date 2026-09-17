"""
Section 13 (label-free): a rank function found by CMA-ES on simulated mean response
time alone — no Gittins labels anywhere in training.

  stage0   positive control on 1-6-14, ρ = 0.8: log-rank at 32 knots over age (the RL
           scratch-arm class), CMA-ES with common random numbers, 3 optimizer seeds;
           fresh-seed paired evaluation against true Gittins and FCFS.
           -> results/s13_labelfree_stage0*.csv, figures/s13_labelfree_stage0.png
  train    Stage 1: for each feature tier in egittins.imitation.FEATURE_SETS, a linear
           (or one-hidden-layer) map from the tier's features to a rank, trained across
           N random distributions (1-6-14 and Pareto(2, 1.2, 500) held out) with loss
           = mean over distributions of MRT / MRT(FCFS) on the same seed.
           -> results/s13_labelfree_train*.csv, results/s13_labelfree_params.npz
  static   held-out evaluation, the §7.1 protocol at n = 500 with the SAME trial seeds as
           results/s13_imitation_static.csv: true Gittins, empirical Gittins, FCFS, the
           three existing imitation nets, and every label-free run, paired; Spearman
           ordering agreement with the Gittins rank.
           -> results/s13_labelfree_static*.csv, results/s13_labelfree_ordering.csv
  figures  figures/s13_labelfree_overlay_<arch>.png, figures/s13_labelfree_static_<arch>.png

Run:  python -m experiments.labelfree stage0|train|static|figures [--quick] [--hidden 8]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from egittins.distributions import GridDistribution, one_six_fourteen, bounded_pareto
from egittins.gittins import gittins_policy, fcfs_policy
from egittins.imitation import FEATURE_SETS, FEATURE_NAMES, random_distribution
from egittins.labelfree import (Stage0Problem, Stage1Problem, FeatureSpec, make_spec, cmaes_search,
                                knot_rank, policy_from_rank, feature_policy, paired_mrt, mean_ci,
                                ordering_agreement, monotone_rescale, sample_features, _init)
from egittins.simulate import simulate

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
RHO = 0.8
N_SAMPLES = 500
HELD_OUT = {"1-6-14": one_six_fourteen, "bounded-Pareto": bounded_pareto}
HPARAMS = os.path.join(RES, "s13_labelfree_hparams.csv")


def log_hparams(row: dict):
    """Every optimizer configuration that was run, appended in order (nothing is deleted)."""
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **row)
    jl = HPARAMS.replace(".csv", ".jsonl")
    with open(jl, "a") as f:
        f.write(json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}) + "\n")
    with open(jl) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pd.DataFrame(rows).to_csv(HPARAMS, index=False)          # union of columns, rewritten each time


# ------------------------------------------------------------------- stage 0

def run_stage0(seeds: int, n_gen: int, popsize: int, seeds_per_gen: int, n_busy: int, sigma0: float,
               knots: int, workers: int, trials: int, eval_busy: int):
    F = one_six_fourteen()
    L = F.max_u + 1
    prob = Stage0Problem(F, RHO, n_busy, knots)
    val_seeds = (900_001, 900_002, 900_003, 900_004)
    g_val = np.mean([simulate(gittins_policy(F, L), F, RHO, n_busy, s).mean_response_time for s in val_seeds])
    hist, thetas, t0 = [], {}, time.time()
    with Pool(workers, initializer=_init, initargs=(prob,)) as pool:
        for seed in range(seeds):
            print(f"stage 0, optimizer seed {seed}", flush=True)
            res = cmaes_search(prob, seed, popsize, sigma0, n_gen, seeds_per_gen, workers, pool=pool,
                               val_seeds=val_seeds, val_every=5)
            for r in res.history:
                if "val" in r:
                    r["val_ratio"] = r["val"] / g_val
            hist += res.history
            thetas[f"seed{seed}"] = res.x
            log_hparams(dict(stage="stage0", tier="knots32", arch="knots", **res.config,
                             train_busy_periods=res.busy_periods, wall_s=round(time.time() - t0, 1),
                             final_val_ratio=res.history[-1]["val"] / g_val))
    pd.DataFrame(hist).to_csv(os.path.join(RES, "s13_labelfree_stage0_curves.csv"), index=False)
    np.savez(os.path.join(RES, "s13_labelfree_stage0_params.npz"), **thetas)
    print(f"training done in {time.time() - t0:.0f} s", flush=True)

    # fresh-seed paired evaluation
    rng = np.random.default_rng(77)
    control = rng.normal(size=knots)                     # a random 32-knot table, same class
    pols = {"true_gittins": gittins_policy(F, L), "fcfs": fcfs_policy(L),
            "random_knots": policy_from_rank(knot_rank(control, L), "random")}
    for k, th in thetas.items():
        pols[f"cmaes_{k}"] = policy_from_rank(knot_rank(th, L), k)
    jobs = [(pols, F, RHO, eval_busy, 2**30 + 60_000 + t) for t in range(trials)]
    with Pool(workers) as pool:
        rows = pool.starmap(paired_mrt, jobs)
    df = pd.DataFrame(rows)
    df.insert(0, "trial", range(trials))
    df.to_csv(os.path.join(RES, "s13_labelfree_stage0.csv"), index=False)
    summ = []
    truth = gittins_policy(F, L).rank
    for c in df.columns:
        if c in ("trial", "true_gittins"):
            continue
        m, ci = mean_ci(df[c] / df["true_gittins"])
        rank = pols[c].rank
        summ.append(dict(policy=c, ratio=m, ci=ci, n=trials,
                         spearman_vs_true=ordering_agreement(rank, truth)))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "s13_labelfree_stage0_summary.csv"), index=False)
    print("\nStage 0 (MRT / true Gittins, paired, 95% CI):")
    print(summ.round(4).to_string(index=False))
    plot_stage0()


def plot_stage0():
    import matplotlib.pyplot as plt
    from egittins.plotting import use_style, savefig, headline, INK, INK2, AQUA, VIOLET, BLUE
    use_style()
    F = one_six_fourteen()
    L = F.max_u + 1
    truth = gittins_policy(F, L).rank
    thetas = np.load(os.path.join(RES, "s13_labelfree_stage0_params.npz"))
    curves = pd.read_csv(os.path.join(RES, "s13_labelfree_stage0_curves.csv"))
    summ = pd.read_csv(os.path.join(RES, "s13_labelfree_stage0_summary.csv")).set_index("policy")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1, 1.2]))
    fig.subplots_adjust(wspace=0.24)
    cols = [AQUA, VIOLET, BLUE]
    for (seed, sub), c in zip(curves.dropna(subset=["val_ratio"]).groupby("seed"), cols):
        ax.plot(sub.gen, sub.val_ratio, color=c, lw=1.4, label=f"optimizer seed {seed}")
    ax.axhline(1.0, color=INK, ls="--", lw=1, label="true Gittins")
    ax.axhline(summ.loc["fcfs", "ratio"], color=INK2, ls=":", lw=1, label="first-come first-served (FCFS)")
    ax.set_xlabel("CMA-ES generation")
    ax.set_ylabel("mean response time / true Gittins  (fixed validation seeds)")
    ax.set_title("(a) search mean during training, 3 optimizer seeds")
    ax.legend(loc="upper right")
    ages = np.arange(L) * F.h
    ax2.plot(ages, truth, color=INK, lw=2.4, label="true Gittins rank")
    for (k, th), c in zip(thetas.items(), cols):
        r = monotone_rescale(knot_rank(th, L), truth)
        ax2.plot(ages, r, color=c, lw=1.2, alpha=0.9,
                 label=f"CMA-ES, {k.replace('seed', 'seed ')}  (ratio {summ.loc['cmaes_' + k, 'ratio']:.3f})")
    ax2.set_xlabel("age  a")
    ax2.set_ylabel("rank, after monotone rescale onto the Gittins values")
    ax2.set_title("(b) learned 32-knot rank tables, ordering only")
    ax2.legend(loc="upper left")
    headline(fig, "Stage 0: label-free search on 1-6-14 (positive control)",
             f"ρ = 0.8; fitness = simulated mean response time, common random numbers within a generation; "
             "evaluation paired on fresh seeds.", top=0.82)
    savefig(fig, os.path.join(FIG, "s13_labelfree_stage0.png"))


# --------------------------------------------------------------------- stage 1

def training_set(n_train: int, seed: int, h: float = 0.01):
    """Random distributions from the imitation generators (held-out parameters excluded
    there), each with one window of N_SAMPLES; skip anything degenerate for a scheduler."""
    rng = np.random.default_rng(seed)
    dists, samples, rows = [], [], []
    while len(dists) < n_train:
        F = random_distribution(rng, h)
        s = F.sample_u(rng, N_SAMPLES)
        cv2 = F.variance() / F.mean() ** 2
        if len(np.unique(s)) < 5 or cv2 < 0.05:
            continue
        L = F.max_u + 1
        fc = simulate(fcfs_policy(L), F, RHO, 2000, 1).mean_response_time
        gi = simulate(gittins_policy(F, L), F, RHO, 2000, 1).mean_response_time   # logged only, never trained on
        dists.append(F)
        samples.append(s)
        rows.append(dict(k=len(dists) - 1, family=F.name, mean=F.mean(), cv2=cv2, max=F.max_u * h,
                         atoms=len(F.atoms_u), fcfs_over_gittins_2000bp=fc / gi))
    return dists, samples, pd.DataFrame(rows)


def run_train(tiers, hidden: int, n_train: int, dist_seed: int, seeds: int, n_gen: int, popsize: int,
              seeds_per_gen: int, n_busy: int, sigma0: float, workers: int):
    dists, samples, info = training_set(n_train, dist_seed)
    info.to_csv(os.path.join(RES, "s13_labelfree_train_dists.csv"), index=False)
    print(info.round(3).to_string(index=False), flush=True)
    arch = "linear" if hidden == 0 else f"mlp{hidden}"
    ppath = os.path.join(RES, "s13_labelfree_params.npz")
    params = dict(np.load(ppath, allow_pickle=True)) if os.path.exists(ppath) else {}
    hpath = os.path.join(RES, "s13_labelfree_train_curves.csv")
    hist = pd.read_csv(hpath).to_dict("records") if os.path.exists(hpath) else []
    hist = [r for r in hist if r.get("arch") != arch]           # rerun of the same arch replaces it
    for tier in tiers:
        spec = make_spec(dists, samples, tier, hidden)
        prob = Stage1Problem(dists, samples, spec, RHO, n_busy)
        print(f"\n[{tier}, {arch}] {spec.n_params} parameters, {len(dists)} training distributions", flush=True)
        with Pool(workers, initializer=_init, initargs=(prob,)) as pool:
            for seed in range(seeds):
                t0 = time.time()
                res = cmaes_search(prob, seed, popsize, sigma0, n_gen, seeds_per_gen, workers, pool=pool,
                                   val_every=10)
                for r in res.history:
                    r.update(tier=tier, arch=arch)
                hist += res.history
                key = f"{tier}|{arch}|seed{seed}"
                params[key] = res.x
                params[key + "|mu"] = spec.mu
                params[key + "|sd"] = spec.sd
                log_hparams(dict(stage="stage1", tier=tier, arch=arch, hidden=hidden, n_train=len(dists),
                                 dist_seed=dist_seed, n_busy=n_busy, **res.config,
                                 train_busy_periods=res.busy_periods, wall_s=round(time.time() - t0, 1),
                                 final_val_loss=res.history[-1]["val"]))
                print(f"  seed {seed}: final val loss (MRT/FCFS, training dists) {res.history[-1]['val']:.4f}, "
                      f"{time.time() - t0:.0f} s, {res.busy_periods:,} busy periods", flush=True)
                np.savez(ppath, **params)
                pd.DataFrame(hist).to_csv(hpath, index=False)


def load_runs():
    """{key: (FeatureSpec, params)} for every trained label-free run."""
    d = dict(np.load(os.path.join(RES, "s13_labelfree_params.npz"), allow_pickle=True))
    runs = {}
    for k, v in d.items():
        if "|mu" in k or "|sd" in k:
            continue
        tier, arch, seed = k.split("|")
        hidden = 0 if arch == "linear" else int(arch[3:])
        runs[k] = (FeatureSpec(tier, hidden, d[k + "|mu"], d[k + "|sd"]), v)
    return runs


_NETS = {}


def get_net(fs):
    from egittins.imitation import RankNet
    if fs not in _NETS:
        _NETS[fs] = RankNet.load(os.path.join(RES, f"imitation_{fs}.pt"))
    return _NETS[fs]


def _worker_init():
    import torch
    torch.set_num_threads(1)


def mean_excess_policy(sizes_u, L):
    """Rank = conditional mean excess E[S − a | S > a] of the sample: the ordering the label-free
    tail-only search converges to (and the b = ∞ member of the Gittins family, an upper bound on the rank)."""
    ages, X, valid = sample_features(sizes_u, L)
    rank = np.full(L, np.inf)
    rank[ages[valid]] = np.exp(X[valid][:, FEATURE_NAMES.index("log_m")])
    return policy_from_rank(rank, "mean excess")


def _static_trial(args):
    """Same seeds as experiments/imitation.py::_static_trial, so rows pair with s13_imitation_static.csv."""
    from egittins.imitation import nn_policy
    dist_name, trial, n_busy, runs = args
    F = HELD_OUT[dist_name]()
    L = F.max_u + 1
    rng = np.random.default_rng(2_000_003 * trial + 17)
    s = F.sample_u(rng, N_SAMPLES)
    truth = gittins_policy(F, L)
    emp = gittins_policy(GridDistribution.empirical(s, F.h), L)
    pols = {"true_gittins": truth, "empirical_gittins": emp, "fcfs": fcfs_policy(L),
            "mean_excess": mean_excess_policy(s, L)}
    for fs in FEATURE_SETS:
        pols[f"imitation_{fs}"] = nn_policy(get_net(fs), s, F.h, L)
    for key, (spec, x) in runs.items():
        pols["labelfree_" + key] = feature_policy(spec, x, s, L)
    out = paired_mrt(pols, F, RHO, n_busy, 50_000 + trial)
    out.update(dist=dist_name, trial=trial)
    lim = int(s.max())
    order = [dict(dist=dist_name, trial=trial, policy=name,
                  spearman_vs_true=ordering_agreement(p.rank[:lim], truth.rank[:lim]),
                  spearman_vs_empirical=ordering_agreement(p.rank[:lim], emp.rank[:lim]))
             for name, p in pols.items() if name != "true_gittins"]
    return out, order


def run_static(trials: int, n_busy: int, workers: int):
    runs = load_runs()
    jobs = [(d, t, n_busy, runs) for d in HELD_OUT for t in range(trials)]
    print(f"static: {trials} trials × {n_busy} busy periods, n = {N_SAMPLES}, ρ = {RHO}, "
          f"{len(runs)} label-free runs", flush=True)
    rows, orders = [], []
    with Pool(workers, initializer=_worker_init) as pool:
        for i, (r, o) in enumerate(pool.imap_unordered(_static_trial, jobs)):
            rows.append(r)
            orders += o
            if (i + 1) % max(1, len(jobs) // 8) == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows).sort_values(["dist", "trial"])
    df.to_csv(os.path.join(RES, "s13_labelfree_static.csv"), index=False)
    od = pd.DataFrame(orders)
    od.to_csv(os.path.join(RES, "s13_labelfree_ordering.csv"), index=False)
    summ = []
    for d, sub in df.groupby("dist"):
        for c in df.columns:
            if c in ("dist", "trial", "true_gittins"):
                continue
            m, ci = mean_ci(sub[c] / sub["true_gittins"])
            oc = od[(od.dist == d) & (od.policy == c)]
            summ.append(dict(dist=d, policy=c, ratio=m, ci=ci, n=len(sub),
                             spearman_vs_true=oc.spearman_vs_true.mean(),
                             spearman_vs_empirical=oc.spearman_vs_empirical.mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "s13_labelfree_static_summary.csv"), index=False)
    print("\nHeld-out static evaluation (MRT / true Gittins, paired, 95% CI; Spearman vs true Gittins rank):")
    piv = summ.pivot(index="policy", columns="dist", values=["ratio", "ci", "spearman_vs_true"]).round(3)
    print(piv.to_string())


# --------------------------------------------------------------------- figures

TIER_LABEL = {"tail_only": "tail + mean excess only", "no_hazard": "+ conditional quantiles", "full": "+ hazard (full)"}


def best_seed_by_training(arch: str) -> dict:
    """Per tier, the optimizer seed with the lowest final training-validation loss (never held-out)."""
    h = pd.read_csv(os.path.join(RES, "s13_labelfree_train_curves.csv"))
    h = h[(h.arch == arch)].dropna(subset=["val"])
    last = h.sort_values("gen").groupby(["tier", "seed"]).last().reset_index()
    return {t: int(sub.sort_values("val").iloc[0].seed) for t, sub in last.groupby("tier")}


def plot_figures(arch: str):
    import matplotlib.pyplot as plt
    from egittins.plotting import use_style, savefig, headline, INK, INK2, BLUE, VIOLET, AQUA, ORANGE
    use_style()
    runs = load_runs()
    best = best_seed_by_training(arch)
    summ = pd.read_csv(os.path.join(RES, "s13_labelfree_static_summary.csv"))
    tiers = [t for t in FEATURE_SETS if t in best]
    colors = {"tail_only": INK2, "no_hazard": AQUA, "full": VIOLET}
    styles = {"tail_only": ":", "no_hazard": "--", "full": "-"}

    # overlay after monotone rescale, one sample per held-out distribution
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(wspace=0.24)
    for ax, (name, ctor) in zip(axes, HELD_OUT.items()):
        F = ctor()
        L = F.max_u + 1
        s = F.sample_u(np.random.default_rng(11), N_SAMPLES)
        truth = gittins_policy(F, L).rank
        lim = int(s.max())
        ages = np.arange(L) * F.h
        m = np.arange(L) < lim
        ax.plot(ages[m], truth[m], color=INK, lw=2.4, label="true Gittins")
        emp = gittins_policy(GridDistribution.empirical(s, F.h), L).rank
        ax.plot(ages[m], emp[m], color=BLUE, lw=1.0, alpha=0.9, label=f"empirical Gittins (n = {N_SAMPLES})")
        for t in tiers:
            key = f"{t}|{arch}|seed{best[t]}"
            spec, x = runs[key]
            r = monotone_rescale(feature_policy(spec, x, s, L).rank, truth)
            rho_s = summ[(summ.dist == name) & (summ.policy == "labelfree_" + key)].spearman_vs_true.iloc[0]
            ax.plot(ages[m], r[m], color=colors[t], ls=styles[t], lw=1.3,
                    label=f"label-free, {TIER_LABEL[t]}  (Spearman {rho_s:.2f})")
        ax.set_xlabel("age  a")
        ax.set_ylabel("rank, learned curves after monotone rescale onto Gittins values")
        ax.set_title(f"{name} job sizes (held out)")
        if name == "bounded-Pareto":
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(2, lim * F.h)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, columnspacing=1.8)
    headline(fig, "Label-free rank functions on the held-out distributions, ordering only",
             f"Each tier's best optimizer seed by training loss ({arch}); rank values are quantile-matched to "
             "the true Gittins rank so only the ordering over ages is compared.", top=0.82)
    savefig(fig, os.path.join(FIG, f"s13_labelfree_overlay_{arch}.png"))

    # paired ratios side by side
    order = ["empirical_gittins", "mean_excess", "imitation_full", "imitation_no_hazard", "imitation_tail_only"]
    labels = ["empirical Gittins (exact)", "rank by sample mean excess", "imitation, full", "imitation, no hazard",
              "imitation, tail only"]
    for t in tiers:
        for seed in range(3):
            key = f"labelfree_{t}|{arch}|seed{seed}"
            if key in set(summ.policy):
                order.append(key)
                labels.append(f"label-free, {TIER_LABEL[t]}, seed {seed}")
    order.append("fcfs")
    labels.append("FCFS")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    y = np.arange(len(order))[::-1]
    for ax, name in zip(axes, HELD_OUT):
        sub = summ[summ.dist == name].set_index("policy").loc[order]
        cols = [BLUE if p == "empirical_gittins" else AQUA if p == "mean_excess" else VIOLET if p.startswith("imitation") else
                ORANGE if p.startswith("labelfree") else INK2 for p in order]
        ax.errorbar(sub.ratio, y, xerr=sub.ci, fmt="o", ecolor=INK2, elinewidth=0.9, capsize=2.5,
                    markersize=5, markerfacecolor="none", color=INK2, zorder=2)
        ax.scatter(sub.ratio, y, c=cols, s=36, zorder=3)
        ax.axvline(1.0, color=INK, ls="--", lw=1)
        ax.set_title(f"{name} (held out)")
        ax.set_xlabel("mean response time / true Gittins  (paired, 95% CI)")
        if name == "bounded-Pareto":
            ax.set_xscale("log")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    headline(fig, "Held-out performance: label-free tiers next to the imitation nets",
             f"20 trials × 10,000 busy periods, n = {N_SAMPLES}, ρ = 0.8; the same trial seeds as the imitation table.",
             top=0.84)
    savefig(fig, os.path.join(FIG, f"s13_labelfree_static_{arch}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["stage0", "train", "static", "figures"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--seeds-per-gen", type=int, default=4)
    ap.add_argument("--busy", type=int, default=2000, help="busy periods per fitness seed (per distribution)")
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--knots", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=0, help="stage 1: 0 = linear, else hidden width")
    ap.add_argument("--tiers", default="tail_only,no_hazard,full")
    ap.add_argument("--n-train", type=int, default=14)
    ap.add_argument("--dist-seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--eval-busy", type=int, default=10_000)
    ap.add_argument("--arch", default="linear", help="figures: which architecture to draw")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.gens, a.popsize, a.seeds_per_gen, a.busy, a.trials, a.eval_busy, a.n_train = 2, 6, 6, 2, 300, 3, 1000, 3
    if a.which == "stage0":
        run_stage0(a.seeds, a.gens, a.popsize, a.seeds_per_gen, a.busy, a.sigma0, a.knots, a.workers,
                   a.trials, a.eval_busy)
    elif a.which == "train":
        run_train(a.tiers.split(","), a.hidden, a.n_train, a.dist_seed, a.seeds, a.gens, a.popsize,
                  a.seeds_per_gen, a.busy, a.sigma0, a.workers)
    elif a.which == "static":
        run_static(a.trials, a.eval_busy, a.workers)
    else:
        plot_figures(a.arch)
