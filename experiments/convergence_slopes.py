"""
Observed convergence rates: for each (distribution, load, policy ∈ {empirical, truncated}) the
log-log slope of (pooled ratio − 1) against n over n = 10, 100, 1000, with a bootstrap 95% CI
over trials (trials resampled independently within each n cell). ρ = 0.8 from
results/baseline_comparison.csv (10,000 busy periods per trial), ρ = 0.98 from
results/baseline_comparison_long.csv (40,000). Pooled ratio = Σ trial MRT / Σ paired true-Gittins MRT.

Cells where the truncated policy equals the empirical one at n = 10 (the truncation level exceeds
every sample) are dropped from the truncated fit, which is then a two-point slope.

Theoretical reference (Theorems 6.1 and 6.3 with α → ∞, bounded / light-tailed limit): the gap is
O(n^{-1/3}) truncated and O(n^{-1/4}) untruncated, i.e. slopes −1/3 and −1/4 as upper bounds on
the decay of the guaranteed gap.

Run:  python -m experiments.convergence_slopes  -> results/convergence_slopes.csv
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
NS = [10, 100, 1000]
B = 10_000
THEORY = {"empirical": -0.25, "truncated": -1 / 3}


def slope(ns, gaps):
    x, y = np.log(ns), np.log(gaps)
    return np.polyfit(x, y, 1)[0] if len(ns) >= 2 else np.nan


def main():
    rng = np.random.default_rng(0)
    old = pd.read_csv(os.path.join(RES, "baseline_comparison.csv"))
    new = pd.read_csv(os.path.join(RES, "baseline_comparison_long.csv"))
    df = pd.concat([old[old.rho == 0.8], new[new.rho == 0.98]])
    rows = []
    for (d, rho), sub in df.groupby(["dist", "rho"]):
        for pol in ("empirical", "truncated"):
            ns_used, note = list(NS), ""
            if pol == "truncated":
                s10 = sub[sub.n == 10]
                if np.allclose(s10.truncated.values, s10.empirical.values):
                    ns_used, note = [100, 1000], "n = 10 dropped (truncated = empirical there); two-point slope"
            cells = {n: sub[sub.n == n] for n in ns_used}
            gaps = np.array([cells[n][pol].sum() / cells[n].true_paired.sum() - 1 for n in ns_used])
            pt = slope(ns_used, gaps)
            bs = np.empty(B)
            for b in range(B):
                g = []
                for n in ns_used:
                    c = cells[n]
                    idx = rng.integers(0, len(c), size=len(c))
                    g.append(c[pol].values[idx].sum() / c.true_paired.values[idx].sum() - 1)
                g = np.array(g)
                bs[b] = slope(ns_used, g) if np.all(g > 0) else np.nan
            ok = bs[np.isfinite(bs)]
            lo, hi = np.percentile(ok, [2.5, 97.5])
            th = THEORY[pol]
            rows.append(dict(dist=d, rho=rho, policy=pol, n_points=len(ns_used),
                             gap_n10=gaps[0] if 10 in ns_used else np.nan,
                             gap_n100=gaps[ns_used.index(100)], gap_n1000=gaps[ns_used.index(1000)],
                             slope=pt, ci_lo=lo, ci_hi=hi, theory=th,
                             ci_excludes_theory=bool(lo > th or hi < th), boot_valid=len(ok) / B, note=note))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "convergence_slopes.csv"), index=False)
    print("| distribution, ρ | policy | gap at n = 10 / 100 / 1000 | observed slope [95% CI] | guaranteed rate | CI excludes it? |")
    print("|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        g10 = "–" if np.isnan(r.gap_n10) else f"{r.gap_n10:.3f}"
        note = f" ({r.note})" if r.note else ""
        print(f"| {r.dist}, {r.rho} | {r.policy} | {g10} / {r.gap_n100:.3f} / {r.gap_n1000:.3f} | "
              f"{r.slope:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]{note} | {r.theory:.2f} | {'yes' if r.ci_excludes_theory else 'no'} |")
    return out


if __name__ == "__main__":
    main()
