"""
Old vs new at ρ = 0.98: results/baseline_comparison.csv (4,000 busy periods per trial)
against results/baseline_comparison_long.csv (10× longer), per (distribution, n, policy):

  median [IQR] of the paired ratio MRT / true-Gittins MRT on the same seed;
  pooled ratio  Σ_trials MRT / Σ_trials paired true-Gittins MRT, bootstrap 95% CI over trials;
  fraction of trials with ratio < 1.

Then two distinguishability checks on the long run (bounded Pareto):
  n = 100 vs n = 1000 (different seeds per n: bootstrap CI of the pooled-ratio difference);
  truncated vs untruncated at each n (same seed: paired, bootstrap CI of the pooled-ratio
  difference and the fraction of trials where truncated is lower).

Run:  python -m experiments.baseline_long_report  -> results/baseline_comparison_long_report.csv
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
B = 10_000


def pooled(num, den):
    return num.sum() / den.sum()


def boot_pooled(num, den, rng, B=B):
    n = len(num)
    idx = rng.integers(0, n, size=(B, n))
    return num[idx].sum(1) / den[idx].sum(1)


def cell_stats(sub, col, rng):
    r = (sub[col] / sub.true_paired).values
    num, den = sub[col].values, sub.true_paired.values
    bs = boot_pooled(num, den, rng)
    q1, med, q3 = np.percentile(r, [25, 50, 75])
    return dict(median=med, q1=q1, q3=q3, pooled=pooled(num, den), pooled_lo=np.percentile(bs, 2.5),
                pooled_hi=np.percentile(bs, 97.5), frac_below_1=float((r < 1).mean()), trials=len(r),
                busy=None)


def main():
    old = pd.read_csv(os.path.join(RES, "baseline_comparison.csv"))
    new = pd.read_csv(os.path.join(RES, "baseline_comparison_long.csv"))
    old, new = old[old.rho == 0.98], new[new.rho == 0.98]
    rng = np.random.default_rng(0)
    rows = []
    for label, df in (("old (3,000 bp empirical / 4,000 bp paired)", old), ("new (40,000 bp)", new)):
        for (d, n), sub in df.groupby(["dist", "n"]):
            for col in ("empirical", "truncated"):
                st = cell_stats(sub, col, rng)
                st.pop("busy")
                rows.append(dict(run=label, dist=d, n=n, policy=col, **st))
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(RES, "baseline_comparison_long_report.csv"), index=False)

    print("| distribution | n | policy | run | median [IQR] | pooled ratio (95% bootstrap CI) | P(ratio < 1) |")
    print("|---|---:|---|---|---|---|---:|")
    for (d, n, p), sub in rep.groupby(["dist", "n", "policy"], sort=False):
        for _, r in sub.iterrows():
            print(f"| {d} | {n} | {p} | {r.run} | {r['median']:.3f} [{r.q1:.2f}, {r.q3:.2f}] | "
                  f"{r.pooled:.3f} [{r.pooled_lo:.3f}, {r.pooled_hi:.3f}] | {r.frac_below_1:.2f} |")

    # distinguishability on the long run, bounded Pareto
    P = new[new.dist == "bounded-Pareto"]
    print("\nLong run, bounded Pareto, ρ = 0.98:")
    for col in ("empirical", "truncated"):
        a, b = P[P.n == 100], P[P.n == 1000]
        ba = boot_pooled(a[col].values, a.true_paired.values, rng)
        bb = boot_pooled(b[col].values, b.true_paired.values, rng)
        diff = ba - bb
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"  {col}: pooled(n=100) − pooled(n=1000) = {pooled(a[col].values, a.true_paired.values) - pooled(b[col].values, b.true_paired.values):.3f}, "
              f"95% CI [{lo:.3f}, {hi:.3f}] -> {'distinguishable' if lo > 0 or hi < 0 else 'NOT distinguishable'}")
    for n in (100, 1000):
        s = P[P.n == n]
        num, den = s.truncated.values, s.true_paired.values
        num2 = s.empirical.values
        idx = rng.integers(0, len(s), size=(B, len(s)))
        diff = (num[idx].sum(1) - num2[idx].sum(1)) / den[idx].sum(1)     # paired: same seed per trial
        lo, hi = np.percentile(diff, [2.5, 97.5])
        frac = float((s.truncated < s.empirical).mean())
        print(f"  n={n}: pooled(truncated) − pooled(empirical) = {(num.sum() - num2.sum()) / den.sum():.3f}, "
              f"95% CI [{lo:.3f}, {hi:.3f}], truncated lower in {frac:.0%} of trials -> "
              f"{'distinguishable' if lo > 0 or hi < 0 else 'NOT distinguishable'}")
    return rep


if __name__ == "__main__":
    main()
