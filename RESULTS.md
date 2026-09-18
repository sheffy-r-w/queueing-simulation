# RESULTS: every number, with its confidence interval and the command that produced it

> **A note from me (Shefali).** This file was prepared with AI assistance (Claude Code) and I
> checked its numbers against the CSVs. If you are new to the repo: `egittins/` is the simulator
> and rank-function library, the scripts in `experiments/` run the experiments and write
> `results/*.csv` and `figures/*.png`, and each section below names the script it ran and the
> CSV it reads from. Every number here comes from one of those CSVs.

All simulations: M/G/1, grid step h = 0.01, inter-arrivals rounded to the nearest
quantum, load ρ = 0.8 unless stated. Every comparison is **paired** (common random
numbers: the same seed, hence the same arrival/size stream, for every policy in a
trial). "MRT" = mean response time. CIs are 95% t intervals over independent trials
unless stated. Machine: Apple Silicon Mac, 14 cores, Python 3.12, numba 0.67.

Run everything from the repo root inside `.venv`:

```bash
source .venv/bin/activate
python -m pytest tests -q                 # 58 tests
```

---

## §5a  Rank computation and validation

**Command:** `python -m experiments.rank_hull` → `figures/s05a_rank_hull.png`, `results/rank_timing.csv`

Both kernels tabulate the Gittins rank at every integer age below the last atom.
"Direct" is Definition 2.5 evaluated at every age against every later atom, O(K·L);
"hull" is the convex-hull sweep, O(K + L). Output is **bit-identical** on every
distribution below (max relative difference 0.0), and on 10 random pmfs with masses
spanning 6 orders of magnitude agrees with an exact brute-force evaluation of
Definition 2.5 to 1e-10 (`tests/test_gittins.py`).

| distribution | atoms K | ages L | direct (ms) | hull (ms) | speed-up |
|---|---:|---:|---:|---:|---:|
| 1-6-14 (true) | 1,593 | 1,601 | 1.16 | 0.015 | 80× |
| bounded Pareto (true) | 49,801 | 50,001 | 1,175 | 0.36 | 3,300× |
| empirical Pareto, n = 10 | 10 | 50,001 | 0.019 | 0.015 | – |
| empirical Pareto, n = 100 | 97 | 50,001 | 0.14 | 0.056 | 2.6× |
| empirical Pareto, n = 1,000 | 542 | 50,001 | 0.70 | 0.086 | 8× |
| empirical Pareto, n = 10,000 | 1,870 | 50,001 | 5.0 | 0.15 | 33× |
| empirical Pareto, n = 100,000 | 5,632 | 50,001 | 30.2 | 0.19 | 160× |

Simulator validation (`tests/test_simulator.py`): FCFS matches Pollaczek–Khinchine and
PLCFS matches E[S]/(1−ρ) on both distributions at ρ = 0.5 and 0.8 (within 3% for 1-6-14,
5–6% for the heavy-tailed Pareto at 20–40k busy periods); Gittins on a deterministic size
equals FCFS exactly; an independent quantum-by-quantum simulator gave bit-identical
per-job response times.

---

## Throughput of the simulator (branch `baseline-long`)

**Command:** `python -m experiments.sim_benchmark --repeats 3 --workers 14` → `results/sim_benchmark.csv`
(idle machine, numba warm; mean of 3 seeds; one `simulate` call each).

| distribution, ρ | busy periods | jobs per call | FCFS | empirical Gittins, n = 1,000 | true Gittins |
|---|---:|---:|---|---|---|
| 1-6-14, 0.8 | 10,000 | 49,816 | 0.004 s (13.1 M jobs/s) | 0.062 s (0.80 M jobs/s) | 0.168 s (0.30 M jobs/s) |
| 1-6-14, 0.98 | 4,000 | 187,234 | 0.031 s (6.0 M jobs/s) | 1.05 s (0.18 M jobs/s) | 3.07 s (0.061 M jobs/s) |
| bounded Pareto, 0.8 | 10,000 | 50,955 | 0.006 s (8.9 M jobs/s) | 0.054 s (0.95 M jobs/s) | 0.144 s (0.35 M jobs/s) |
| bounded Pareto, 0.98 | 4,000 | 203,592 | 0.148 s (1.4 M jobs/s) | 0.50 s (0.41 M jobs/s) | 1.51 s (0.13 M jobs/s) |

Cost is set by the number of scheduling decisions, not jobs: FCFS decides only at arrivals and
completions; empirical Gittins at every atom of its n-point sample; true Gittins at every grid
point the served job crosses, with an O(m) scan over the m jobs in the system each time, which
is why ρ = 0.98 (m ≈ 50) on 1-6-14 (dense atoms, jobs of ~700 quanta) is the slowest cell.
(Timings are for the reference kernel; the skip-ahead kernel below is the default from this
branch on.)

**Full existing baseline grid** (§5b/§11: 2 distributions × 2 loads × 3 n × 100 trials, three
policies per trial on the same stream, plus the 4 × 100,000-busy-period true-Gittins reference per
(distribution, load)). Extrapolated from the table above (reference kernel): 2,940 core-seconds,
~3.5 min wall on 14 workers with perfect scaling; ~25 min at the script's default `--workers 2`.

**Measured** (2026-09-17, skip-ahead kernel, `python -m experiments.baseline_comparison --workers 14
--out <scratch>`, output deleted afterwards): **wall-clock 860 s = 14.3 min, 2,079 CPU-seconds**, so
the 14 workers were busy 17% of the time: the four true-Gittins references (4 × 100,000 busy periods
each) run serially in the main process before each cell's pool starts, and dominate the wall-clock.
**Total jobs simulated: 491,681,028** (counted exactly: 149,261,514 per policy over the 1,200 trials
× 3 policies, plus 43,896,486 in the references). The ρ = 0.98 cells are 94% of the job count.

**Reproduction check of `results/baseline_comparison.csv`** (`results/baseline_rerun_check.csv`, per
cell): the rerun's `true_paired` and `ell` columns are bit-identical to the CSV in all 1,200 trials.
The `empirical` and `truncated` columns are reproduced at ρ = 0.8 (identical in most trials; a few
n = 10 trials differ by < 1e-4 relative, consistent with last-bit differences in the rank table
flipping a tie) but **not at ρ = 0.98**, where every trial differs, by up to 157% in a single trial,
and the cell medians move (empirical: 1-6-14 n = 10 / 100 / 1,000: 3.677 → 3.775, 1.336 → 1.375,
1.057 → 1.035; Pareto: 1.573 → 1.593, 1.136 → 1.126, 1.112 → 1.084). The long unpaired reference
column also differs (e.g. 1-6-14 ρ = 0.8: 27.029 → 27.014). **Cause (bisected 2026-09-17):** the ρ = 0.98 empirical and truncated values were simulated with
**3,000 busy periods per trial** (the run's original setting; the CSV's first write, 83989fd, and its
completion, bfba071), and every probed value reproduces bit-exactly with the current code at 3,000
busy periods on the trial's seed. Commit f4241ad (2026-09-16) then added the `true_paired` column with
the script's new default of **4,000** busy periods and kept the empirical and truncated values
("unchanged"). So each ρ = 0.98 ratio in the CSV divides a 3,000-busy-period mean response time by a
4,000-busy-period one on the same seed: the first 3,000 busy periods are shared, the last 1,000 are
not, and the pairing is broken there (which is also why the CSV's "residual IQR below 1 at ρ = 0.98"
in the note above was larger than pairing alone would leave). The ρ = 0.8 cells used 10,000 busy
periods for all columns and are correctly paired. No code version is at fault; the current code is
consistent with every version probed (a2148d4 … f4241ad give the same values).
Consequence: quote the ρ = 0.8 cells and the 40,000-busy-period ρ = 0.98 cells (below,
all columns from one run); the 4,000-busy-period ρ = 0.98 rows of `baseline_comparison.csv` and the
"old" column of the comparison table are mis-paired. Regenerating them consistently takes 14 min on
14 workers (`--loads 0.98 --busy098 4000 --out ...`).

---

## §5b, §11  Baseline comparison (paper §7.1 protocol)

**Command:** `python -m experiments.baseline_comparison` (~30 min) → `figures/baseline_comparison.png`, `results/baseline_comparison.csv`

100 trials per (distribution, ρ, n). Each trial: draw n samples, build empirical Gittins
and truncated empirical Gittins (ℓ with Ḡ(ℓ) = n^{-1/3}(1−ρ)^{2/3}), simulate 10,000 busy
periods (at ρ = 0.98: 3,000 for the empirical and truncated columns, 4,000 for the paired true-Gittins
column; see the reproduction check in the Throughput section). Ratio = trial MRT / MRT of true Gittins simulated on the *same*
arrival stream (same seed and busy-period count: paired, common random numbers). The
"true Gittins MRT" column and the FCFS/PLCFS ratios use a separate long reference
(4 × 100,000 busy periods). Median [IQR] and mean ± CI over the 100 trials.

| distribution, ρ | true Gittins MRT | FCFS / opt | PLCFS / opt | n | empirical Gittins | truncated empirical Gittins |
|---|---:|---:|---:|---:|---|---|
| 1-6-14, 0.8 | 27.03 | 1.085 | 1.305 | 10 | 1.387 [1.22, 1.60]; 1.446 ± 0.061 | 1.779 [1.42, 1.99]; 1.717 ± 0.066 |
| | | | | 100 | 1.044 [1.02, 1.08]; 1.060 ± 0.011 | 1.373 [1.29, 1.46]; 1.386 ± 0.025 |
| | | | | 1000 | 1.005 [1.00, 1.01]; 1.006 ± 0.001 | 1.170 [1.15, 1.19]; 1.172 ± 0.005 |
| 1-6-14, 0.98 | 250.6 | 1.116 | 1.408 | 10 | 3.68 [2.31, 5.51]; 4.00 ± 0.41 | 3.68 [2.31, 5.51]; 4.00 ± 0.41 |
| | | | | 100 | 1.336 [1.12, 1.64]; 1.490 ± 0.115 | 1.720 [1.38, 2.30]; 1.928 ± 0.155 |
| | | | | 1000 | 1.057 [0.98, 1.10]; 1.031 ± 0.023 | 1.338 [1.24, 1.47]; 1.346 ± 0.044 |
| bounded Pareto, 0.8 | 25.38 | 5.142 | 1.583 | 10 | 1.161 [1.12, 1.23]; 1.207 ± 0.033 | 1.237 [1.18, 1.30]; 1.246 ± 0.019 |
| | | | | 100 | 1.046 [1.04, 1.07]; 1.052 ± 0.005 | 1.152 [1.12, 1.18]; 1.151 ± 0.009 |
| | | | | 1000 | 1.017 [1.01, 1.03]; 1.019 ± 0.002 | 1.077 [1.06, 1.09]; 1.077 ± 0.004 |
| bounded Pareto, 0.98 | 143.2 | 10.53 | 2.805 | 10 | 1.573 [1.36, 1.86]; 1.639 ± 0.080 | 1.573 [1.36, 1.86]; 1.639 ± 0.080 |
| | | | | 100 | 1.136 [1.01, 1.24]; 1.129 ± 0.038 | 1.170 [1.08, 1.27]; 1.171 ± 0.040 |
| | | | | 1000 | 1.112 [1.00, 1.18]; 1.077 ± 0.031 | 1.089 [0.97, 1.15]; 1.052 ± 0.028 |

Notes. Pairing matters: a single 4,000-busy-period run of true Gittins itself, divided by
the long reference, has median 0.87–0.94 and SD 0.25–0.30 across seeds at ρ = 0.98 (heavy-tailed
response times), so unpaired ratios put several medians below 1 in the earlier version of this
table; paired, every empirical-Gittins median is ≥ 1. The residual IQR below 1 at ρ = 0.98 is
the within-trial noise that pairing does not remove.
The paper's Fig. 7.2 gives 1.074 for FCFS/opt on 1-6-14 at ρ = 0.8 vs 1.085 here (the
discretization lifts E[S] to 7.06). At n = 10 and ρ = 0.98, ℓ exceeds every sample, so
truncated = untruncated.

### ρ = 0.98 with 10× the busy periods (branch `baseline-long`)

*Old = the original CSV, whose ρ = 0.98 empirical and truncated values are 3,000-busy-period means divided by 4,000-busy-period paired true-Gittins means (mis-paired; see the Throughput section). New = one consistent 40,000-busy-period run.*

**Command:** `python -m experiments.baseline_comparison --loads 0.98 --busy098 40000 --trials 100 --workers 12 --out baseline_comparison_long`
(40 min wall on 12 workers with the reference kernel, of which ~16 min is the two serial 4 × 100,000-busy-period
references) → `results/baseline_comparison_long.csv`, `figures/s11_baseline_comparison_long.png`;
`python -m experiments.baseline_long_report` → `results/baseline_comparison_long_report.csv`.
Same protocol and pairing as above (one seed per trial shared by empirical, truncated and true Gittins), 40,000
busy periods per trial for every column; the trial seeds are the same as the original run, so the sample windows
are the same and only the arrival streams are longer. Pooled ratio = Σ trial MRT / Σ paired true-Gittins MRT,
bootstrap 95% CI over trials (10,000 resamples). Long references: 1-6-14 246.6 ± 1.7 (was 250.6),
Pareto 146.9 ± 5.2 (was 143.2).

| distribution | n | policy | run | median [IQR] | pooled ratio [95% CI] | P(ratio < 1) |
|---|---:|---|---|---|---|---:|
| 1-6-14 | 10 | empirical | old (3,000 / 4,000 bp) | 3.68 [2.31, 5.51] | 4.03 [3.61, 4.47] | 0.00 |
| | | | **new (40,000 bp)** | 3.82 [2.64, 6.37] | 4.26 [3.87, 4.66] | 0.00 |
| | 100 | empirical | old (3,000 / 4,000 bp) | 1.336 [1.12, 1.64] | 1.503 [1.386, 1.629] | 0.10 |
| | | | **new (40,000 bp)** | 1.361 [1.15, 1.73] | 1.522 [1.422, 1.630] | 0.00 |
| | 100 | truncated | old (3,000 / 4,000 bp) | 1.720 [1.38, 2.30] | 1.937 [1.783, 2.097] | 0.03 |
| | | | **new (40,000 bp)** | 1.798 [1.48, 2.30] | 1.978 [1.848, 2.116] | 0.00 |
| | 1000 | empirical | old (3,000 / 4,000 bp) | 1.057 [0.98, 1.10] | 1.032 [1.007, 1.058] | 0.32 |
| | | | **new (40,000 bp)** | 1.039 [1.02, 1.07] | 1.052 [1.042, 1.062] | 0.00 |
| | 1000 | truncated | old (3,000 / 4,000 bp) | 1.338 [1.24, 1.47] | 1.356 [1.307, 1.408] | 0.05 |
| | | | **new (40,000 bp)** | 1.376 [1.31, 1.47] | 1.400 [1.374, 1.426] | 0.00 |
| bounded Pareto | 10 | empirical (= truncated) | old (3,000 / 4,000 bp) | 1.573 [1.36, 1.86] | 1.669 [1.578, 1.761] | 0.03 |
| | | | **new (40,000 bp)** | 1.587 [1.39, 1.89] | 1.666 [1.596, 1.743] | 0.00 |
| | 100 | empirical | old (3,000 / 4,000 bp) | 1.136 [1.01, 1.24] | 1.134 [1.090, 1.178] | 0.25 |
| | | | **new (40,000 bp)** | 1.132 [1.08, 1.23] | 1.175 [1.146, 1.210] | 0.00 |
| | 100 | truncated | old (3,000 / 4,000 bp) | 1.170 [1.08, 1.27] | 1.180 [1.135, 1.229] | 0.18 |
| | | | **new (40,000 bp)** | 1.195 [1.11, 1.27] | 1.221 [1.194, 1.249] | 0.00 |
| | 1000 | empirical | old (3,000 / 4,000 bp) | 1.112 [1.00, 1.18] | 1.086 [1.051, 1.119] | 0.28 |
| | | | **new (40,000 bp)** | 1.099 [1.07, 1.13] | 1.106 [1.096, 1.116] | 0.00 |
| | 1000 | truncated | old (3,000 / 4,000 bp) | 1.089 [0.97, 1.15] | 1.055 [1.025, 1.084] | 0.28 |
| | | | **new (40,000 bp)** | 1.067 [1.05, 1.09] | 1.074 [1.067, 1.080] | 0.00 |

(At n = 10 the truncation level exceeds every sample, so truncated = empirical in both runs; the 1-6-14
n = 10 truncated rows equal the empirical rows and are omitted.)

Reading. With 10× the busy periods, **no trial in any cell has a ratio below 1** (it was 10–32% of trials
at n ≥ 100 before), the IQRs at n = 1,000 shrink from ~0.15 to ~0.05, and the pooled-ratio CIs narrow about
3×. Where the two runs differ in the pooled estimate (1-6-14 n = 1,000: 1.032 → 1.052; Pareto n = 100:
1.134 → 1.175; Pareto n = 1,000 truncated 1.055 → 1.074) the old CI contains or touches the new estimate: the
old run at these cells was one low draw of a heavy-tailed statistic, not a different answer. The new values are
the ones to quote for ρ = 0.98.

**Are the bounded-Pareto cells now distinguishable?** Yes, all four comparisons (long run, bootstrap over trials):

| comparison (bounded Pareto, ρ = 0.98) | difference of pooled ratios | 95% CI | note |
|---|---:|---|---|
| empirical: n = 100 − n = 1,000 | +0.070 | [+0.037, +0.105] | different seeds per n |
| truncated: n = 100 − n = 1,000 | +0.147 | [+0.119, +0.176] | different seeds per n |
| n = 100: truncated − empirical | +0.045 | [+0.017, +0.072] | paired; truncated lower in 24% of trials |
| n = 1,000: truncated − empirical | **−0.032** | [−0.045, −0.019] | paired; truncated lower in 71% of trials |

So at n = 1,000 on the heavy-tailed Pareto at ρ = 0.98, truncation *helps* (1.074 vs 1.106) and the
effect is now outside the noise; at n = 100 it hurts (1.221 vs 1.175). n = 100 vs n = 1,000 is
distinguishable for both policies. None of this changes the ρ = 0.8 cells, which were not rerun.

### Skip-ahead simulator kernel (branch `baseline-long`)

**Command:** `python -m experiments.kernel_speedup --repeats 3` → `results/kernel_speedup.csv`; test:
`tests/test_simulator.py::test_skip_kernel_is_bit_identical_to_reference`.

`egittins.simulate.simulate(..., kernel="skip")` (now the default; `kernel="reference"` is the original) keeps the
same selection rule but, after selecting the served job, walks its chain of atoms with an O(1) test — does it
still beat the best *waiting* job, with the same finite-FCFS / infinite-PLCFS tie-break on sequence number — and
only stops where the reference kernel would have switched jobs. The waiting-job scan is done only when an atom
falls inside the run, so FCFS/PLCFS pay nothing. Per-job response times and sizes are **bit-identical** to the
reference kernel on both distributions at ρ = 0.8 and 0.98 for FCFS, PLCFS, true Gittins, empirical (n = 30,
1,000) and truncated Gittins, and arbitrary wiggly tables with ∞ regions (the test), and on every timing call below.

| distribution, ρ | FCFS | empirical Gittins, n = 1,000 | true Gittins |
|---|---|---|---|
| 1-6-14, 0.8 (10,000 bp) | 0.004 → 0.004 s (1.0×) | 0.073 → 0.025 s (2.9×) | 0.204 → 0.063 s (3.2×) |
| 1-6-14, 0.98 (4,000 bp) | 0.037 → 0.038 s (1.0×) | 1.29 → 0.18 s (7.3×) | 3.79 → 0.43 s (**8.7×**) |
| bounded Pareto, 0.8 (10,000 bp) | 0.007 → 0.007 s (1.0×) | 0.065 → 0.026 s (2.5×) | 0.176 → 0.082 s (2.1×) |
| bounded Pareto, 0.98 (4,000 bp) | 0.188 → 0.188 s (1.0×) | 0.63 → 0.14 s (4.4×) | 1.87 → 1.67 s (1.1×) |

The gain is the O(m) scan removed at each atom crossing, so it grows with the number of jobs in the system
(ρ = 0.98) and with atom density (true 1-6-14: every grid point). The dense Pareto at ρ = 0.98 gains little
because its jobs are up to 50,000 quanta long and the walk itself, one step per grid point, dominates; a
range-maximum structure over the rank table would make that O(log L) per decision and is the next step.
The full baseline grid would take ~1,000 core-seconds instead of ~2,940 with this kernel.

### Observed convergence rates vs the guaranteed ones

**Command:** `python -m experiments.convergence_slopes` → `results/convergence_slopes.csv`

For each (distribution, load, policy), the log-log slope of (pooled ratio − 1) against n over
n = 10, 100, 1,000 (**three points per fit**, two where noted), pooled ratio = Σ trial MRT / Σ paired
true-Gittins MRT; ρ = 0.8 from `baseline_comparison.csv` (10,000 busy periods per trial), ρ = 0.98
from `baseline_comparison_long.csv` (40,000). 95% CI = percentile bootstrap over trials, resampled
independently within each n cell (10,000 resamples). Theoretical reference: Theorems 6.1 and 6.3
with α → ∞ (bounded / light-tailed limit) bound the gap by O(n^{−1/3}) truncated and O(n^{−1/4})
untruncated, i.e. slopes of −0.33 and −0.25 on the *guaranteed* gap; an observed slope steeper
than the guarantee is consistent with it (the bound is an upper bound), a shallower one is not.

| distribution, ρ | policy | gap at n = 10 / 100 / 1,000 | observed slope [95% CI] | guaranteed | CI excludes the guaranteed slope? |
|---|---|---|---|---|---|
| 1-6-14, 0.8 | empirical | 0.445 / 0.060 / 0.006 | −0.92 [−0.97, −0.88] | −0.25 | yes (steeper) |
| 1-6-14, 0.8 | truncated | 0.718 / 0.386 / 0.172 | −0.31 [−0.33, −0.29] | −0.33 | yes, barely (shallower) |
| 1-6-14, 0.98 | empirical | 3.264 / 0.522 / 0.052 | −0.90 [−0.95, −0.85] | −0.25 | yes (steeper) |
| 1-6-14, 0.98 | truncated | – / 0.978 / 0.400 | −0.39 [−0.45, −0.32] (two points: n = 10 dropped, truncated = empirical there) | −0.33 | no |
| bounded Pareto, 0.8 | empirical | 0.209 / 0.052 / 0.019 | −0.52 [−0.56, −0.47] | −0.25 | yes (steeper) |
| bounded Pareto, 0.8 | truncated | 0.247 / 0.151 / 0.077 | −0.25 [−0.27, −0.23] | −0.33 | yes (shallower) |
| bounded Pareto, 0.98 | empirical | 0.666 / 0.175 / 0.106 | −0.40 [−0.43, −0.37] | −0.25 | yes (steeper) |
| bounded Pareto, 0.98 | truncated | – / 0.221 / 0.074 | −0.48 [−0.54, −0.41] (two points: n = 10 dropped, truncated = empirical there) | −0.33 | yes (steeper) |

Reading. The **untruncated** empirical policy converges far faster than its n^{−1/4} guarantee in every
cell (slopes −0.4 to −0.9; on 1-6-14 close to n^{−1}), so the bound is loose there. The **truncated**
policy tracks its n^{−1/3} guarantee closely at ρ = 0.8 (−0.31 on 1-6-14, −0.25 on the Pareto; both CIs
exclude −0.33 but by 0.02–0.08, and the Pareto's is *shallower* than guaranteed, which the theorem does
not cover since its constant is for the light-tailed limit); at ρ = 0.98 the truncated fits rest on two
points and say little. With three n values per fit the slopes are descriptive, not estimates of an
asymptotic rate: the 1-6-14 truncated gap at n = 1,000 (0.17) is dominated by the truncation cost
itself, not by sampling error, which is why its slope is shallow.

---

## §8.5 (earlier run)  Window size under mode-shift drift

**Command:** `python -m experiments.drift_window` (~15 min) → `figures/drift_window.png`, `results/drift_window.csv`

F_k = (1−θ_k)·F(1-6-14) + θ_k·F(3-8-14), θ_k a triangle wave with period T busy periods;
5 seeds × 6,000 measured busy periods (+600 warm-up); ratio to an oracle using true Gittins
for the current F_k. Mean ± CI over seeds.

| T | FCFS | all history | P = 10 | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|---|---|---|
| ∞ (stationary) | 1.091 ± 0.003 | 1.001 ± 0.001 | 1.469 ± 0.048 | 1.208 ± 0.037 | 1.056 ± 0.003 | 1.022 ± 0.002 | 1.007 ± 0.002 | 1.003 ± 0.001 |
| 1000 | 1.014 ± 0.002 | 1.015 ± 0.003 | 1.483 ± 0.034 | 1.193 ± 0.036 | 1.051 ± 0.008 | 1.019 ± 0.004 | 1.011 ± 0.002 | 1.026 ± 0.006 |
| 300 | 1.013 ± 0.003 | 1.013 ± 0.004 | 1.461 ± 0.024 | 1.196 ± 0.037 | 1.053 ± 0.007 | 1.027 ± 0.002 | 1.024 ± 0.006 | 1.015 ± 0.004 |
| 100 | 1.012 ± 0.003 | 1.013 ± 0.002 | 1.474 ± 0.041 | 1.193 ± 0.036 | 1.058 ± 0.008 | 1.040 ± 0.005 | 1.018 ± 0.004 | 1.014 ± 0.002 |

(The mode shift moves mass toward larger, less variable sizes, where FCFS is itself near-optimal
for much of the cycle — hence FCFS ≈ 1.01 under drift. The mean-preserving drift below is the
cleaner test.)

---

## §13  k-updating vs. static vs. genie under drift

**Command:** `python -m experiments.kupdating_drift` (~5 min on 12 cores) →
`figures/s13_drift_timecourse.png`, `figures/s13_drift_sweep.png`, `results/s13_*.csv`

Engine: `egittins/kupdating.py`. Genie = true Gittins for the current F_k; static = empirical
Gittins fitted once at the start of measurement from the last 500 completed jobs; k-updating
refits from the last w completed jobs at every busy-period start; FCFS. 600 warm-up busy
periods. Ratio = MRT / genie MRT, mean ± CI over trials.

### Headline: one-way weight drift on 1-6-14 (40 trials × 4,000 busy periods)

Component weights move linearly (0.1, 0.3, 0.6) → (0.6, 0.3, 0.1) over the run, so the
workload goes from mostly long jobs (CV² ≈ 0.2, where Gittins ≈ FCFS) to mostly short jobs
with a heavy 14-mode (CV² ≈ 1.1, FCFS/opt 1.42). The stationary control holds the final
weights (0.6, 0.3, 0.1).

| policy | one-way drift | stationary |
|---|---|---|
| FCFS | 1.105 ± 0.003 | 1.417 ± 0.003 |
| static (w = 500, fit once) | **1.113 ± 0.004** | 1.020 ± 0.004 |
| k-updating, w = 50 | 1.115 ± 0.006 | 1.141 ± 0.006 |
| k-updating, w = 200 | 1.029 ± 0.002 | 1.036 ± 0.002 |
| k-updating, w = 500 | **1.012 ± 0.001** | 1.014 ± 0.001 |
| k-updating, w = 2000 | 1.004 ± 0.000 | 1.004 ± 0.001 |

Time course (`results/s13_headline_blocks.csv`, blocks of 100 busy periods pooled over
trials): the static fit, made when the workload was low-variability, behaves like FCFS for
the whole run and both climb from 1.00 to 1.41 by the end; w = 500 stays within 1.00–1.03
throughout. The genie is the 1.0 line by construction.

Earlier version (drift in the opposite direction, (0.6, 0.3, 0.1) → (0.1, 0.3, 0.6), in git
history before 2026-09-16): FCFS 1.109, static 1.188, w = 500 1.013 over the run; the static
fit ended at 1.36 while FCFS *fell* to 1.00 because the drift target has CV² ≈ 0.2 and
Gittins gains nothing there (stationary FCFS/Gittins = 1.425 at (0.6, 0.3, 0.1), 1.111 midway,
1.000 at (0.1, 0.3, 0.6)). Together the two directions say: a stale fit can be worse than
FCFS or merely no better than it; k-updating with w ≥ 200 tracks the genie either way.

### Three kinds of drift side by side (`figures/s13_drift_types.png`, `results/s13_drift_types.csv`)

Whole-run ratio to the genie, ρ = 0.8; rows are the headline run (40 trials × 4,000 busy
periods), the T = 2000 cell of the sweep (20 × 2,500), and the Pareto tail run (40 × 4,000).

| drift | FCFS | static (fit once, w = 500) | k-updating, w = 500 |
|---|---|---|---|
| weights shift (0.1,0.3,0.6)→(0.6,0.3,0.1) | 1.105 ± 0.003 | 1.113 ± 0.004 | 1.012 ± 0.001 |
| shape drift, load pinned (mean-preserving, T = 2000) | 1.099 ± 0.006 | 1.042 ± 0.004 | 1.012 ± 0.002 |
| tail gets heavier (Pareto α 2.0 → 1.2) | 3.576 ± 0.174 | 1.039 ± 0.009 | 1.024 ± 0.004 |

Reading: *what* drifts decides whether a static fit needs replacing. When the weights of the
modes shift, the stale fit is no better than FCFS; when the shape drifts with the load pinned
it loses ~4%; when only the tail index moves, the rank function barely changes (the ratio of
complementary CDFs is nearly the same for α = 2.0 and 1.2 over the ages that matter) and the
static fit stays within 4% even as FCFS goes from 1.6× to 5× the genie. k-updating with
w = 500 is within 1–2.5% in all three.

### Window × drift speed: mean-preserving weight drift (20 trials × 2,500 busy periods per cell)

Weights on the line w₁ + 6w₆ + 14w₁₄ = 7 (E[S] and hence λ pinned), w₆ a triangle wave
between 0.10 and 0.70 with period T; stationary control at w₆ = 0.40.

| policy | T = 100 | T = 500 | T = 2000 | T = 8000 | stationary |
|---|---|---|---|---|---|
| FCFS | 1.076 ± 0.005 | 1.085 ± 0.006 | 1.099 ± 0.006 | 1.127 ± 0.005 | 1.047 ± 0.003 |
| static (w = 500) | 1.050 ± 0.004 | 1.049 ± 0.005 | 1.042 ± 0.004 | 1.017 ± 0.003 | 1.013 ± 0.006 |
| w = 50 | 1.122 ± 0.007 | 1.120 ± 0.011 | 1.114 ± 0.010 | 1.114 ± 0.012 | 1.117 ± 0.007 |
| w = 200 | 1.066 ± 0.005 | 1.034 ± 0.004 | 1.030 ± 0.003 | 1.027 ± 0.003 | 1.029 ± 0.003 |
| w = 500 | 1.043 ± 0.003 | **1.023 ± 0.003** | 1.012 ± 0.002 | 1.013 ± 0.002 | 1.011 ± 0.002 |
| w = 2000 | **1.031 ± 0.003** | 1.038 ± 0.005 | **1.010 ± 0.001** | **1.004 ± 0.001** | **1.003 ± 0.001** |

Reading: w = 50 is variance-limited everywhere (~1.12); at T = 500 the best window is 500
(w = 2000 lags: 1.038); at T = 100 the drift is faster than any window can track and the
longest window, which averages over cycles, is best. "Too long is stale" holds only in the
T = 500 column: w = 2000 is best or tied in the other four columns and in the headline run.

### Bounded-Pareto tail drift (40 trials × 4,000 busy periods)

α moves linearly 2.0 → 1.2 over the run (x_m = 2, bound 500); stationary control at α = 2.0.

| policy | tail drift α 2.0 → 1.2 | stationary α = 2.0 |
|---|---|---|
| FCFS | 3.58 ± 0.17 | 1.96 ± 0.10 |
| static (w = 500) | 1.039 ± 0.009 | 1.021 ± 0.005 |
| w = 50 | 1.068 ± 0.009 | 1.047 ± 0.004 |
| w = 200 | 1.035 ± 0.004 | 1.025 ± 0.003 |
| w = 500 | 1.024 ± 0.004 | 1.015 ± 0.002 |
| w = 2000 | 1.011 ± 0.002 | 1.006 ± 0.002 |

---

## §13  Imitation-learned rank function

**Command:** `python -m experiments.imitation` (~10 min; needs torch) →
`figures/s13_rank_overlay.png`, `results/s13_imitation_*.csv`, `results/imitation_*.pt`

MLP (3 × 128, SiLU) from features of the conditional-excess sample to log(rank / mean excess);
4,000 random distributions (Gaussian mixtures, bounded Pareto, lognormal, Weibull, few-atom
discrete; 1-6-14 and Pareto(2, 1.2, 500) parameters held out), sample sizes 30–3,000,
254k (age, window) rows, 40 epochs, 10% of distributions held out for validation.

| feature set | features | held-out MSE (log rank) | held-out R² |
|---|---|---:|---:|
| tail + mean excess only | log Ḡ(a), log count, log m | 0.687 | 0.26 |
| + conditional quantiles | + q₁₀, q₂₅, q₅₀, q₇₅, q₉₀, max (÷ m) | 0.153 | 0.84 |
| + hazard (full) | + gap to next atom ÷ m, mass at it | **0.0085** | **0.991** |

### Static evaluation (20 trials × 10,000 busy periods, n = 500, ratio to true Gittins)

| policy | 1-6-14 | bounded Pareto |
|---|---|---|
| empirical Gittins (exact) | 1.007 ± 0.003 | 1.023 ± 0.005 |
| NN, full features | **1.012 ± 0.004** | **1.022 ± 0.004** |
| NN, no hazard features | 1.010 ± 0.003 | 1.116 ± 0.022 |
| NN, tail + mean excess only | 1.095 ± 0.003 | 1.030 ± 0.013 |
| FCFS | 1.089 ± 0.001 | 5.04 ± 0.14 |

### As a k-updating policy (w = 500) under one-way weight drift (20 trials × 4,000 busy periods, ratio to genie)

Drift (0.1, 0.3, 0.6) → (0.6, 0.3, 0.1), as in the §13 headline.

| policy | ratio |
|---|---|
| FCFS | 1.106 ± 0.005 |
| static empirical Gittins | 1.113 ± 0.005 |
| k-updating empirical Gittins, w = 500 | 1.011 ± 0.001 |
| k-updating NN full, w = 500 | **1.017 ± 0.001** |
| k-updating NN no hazard, w = 500 | 1.023 ± 0.002 |
| k-updating NN tail-only, w = 500 | 1.116 ± 0.005 |

Reading (revised after the label-free experiment, §13 below). On 1-6-14 the tail-only net sits
at FCFS level (1.095 vs FCFS 1.089), but that is a fact about this regression, not about the
features: ranking by the sample mean excess E[S − a | S > a], which is one of the three tail-only
features, gives 1.015 ± 0.003 on the same windows and seeds (`results/s13_labelfree_static_summary.csv`).
The tail-only net was fitted to log(rank / mean excess) and its prediction scrambles the ordering
that mean excess alone already carries. With labels, the quantile and hazard features buy the last
0.5–1%: 1.010–1.011 on 1-6-14 and 1.022 on the Pareto, against 1.015 / 1.033 for mean excess and
1.007 / 1.023 for exact empirical Gittins. On the heavy-tailed Pareto the Gittins rank is nearly
monotone in age, so even the tail-only net is close (1.030); the no-hazard net's wiggles at small
ages cost it 1.116. Each imitation tier is a single training seed (seed 0), so the tier-to-tier
differences carry no seed-variance estimate.

---

## §13  RL: REINFORCE on the rank function

**Command:** `python -m experiments.rl` (~3 min) → `figures/s13_rl.png`, `results/s13_rl_curves.csv`, `results/s13_rl_final.csv`

1-6-14, ρ = 0.8, one fixed window of 500 samples. Stochastic policy: at every arrival and
completion the served job is drawn ∝ exp(−rank(age)/τ). Episodes = busy periods, return =
−(sum of response times), return-to-go, batch-normalized advantages, Adam with cosine decay,
2,000 iterations × 1,024 episodes, gradient-norm clip 1. Deterministic evaluation (SOAP
simulator) paired against true Gittins on 10,000 busy periods.

| arm | parametrization | lr, τ | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|
| from scratch | log-rank at 32 knots over age, linear interpolation, constant init (= FCFS, 1.093) | 0.03, 2.0 | 1.090 | 1.090 | 1.159 |
| fine-tune | the full-feature imitation net (starts at 1.001) | 3e-5, 0.3 | 1.002 | 1.005 | 1.008 |

Reading: fine-tuning reproduces — it holds the imitation optimum (the policy-gradient signal
is too noisy to improve on it, and small enough not to damage it; lr ≥ 1e-4 does damage it,
to ~1.02–1.10). From scratch, two seeds settle on a table whose maximum is at age 0 (r(0) = 22
and 31 against r ≤ 8 elsewhere): a new arrival never preempts, so the deterministic simulator
executes it exactly as FCFS (the 1.090 is the FCFS value on that evaluation seed) whatever the
wiggles below age 0's rank do. The third seed puts its maximum mid-run (age 7.6) and is worse
than FCFS (1.159). No seed found the two-mode structure. An earlier variant with one free parameter per age (1,600
parameters) diverged to 1.31–2.01 on all seeds: each age parameter sees only the few
decisions taken at that exact age, and the noise wins.

Decision: fine-tune reproduces across seeds; scratch reproduces its *qualitative*
conclusion (does not beat FCFS) but not its number. Include only as a one-liner ("policy
gradient on busy-period episodes is too noisy to discover the rank structure; imitation
supplies it") or leave out.

---

## §13  Label-free rank search (CMA-ES on simulated response time; no Gittins labels)

**Commands** (branch `labelfree`; `pip install cma`; ~9 min of compute in total on 12 workers):

```bash
python -m experiments.labelfree stage0                                   # ~1 min train + eval
python -m experiments.labelfree train --hidden 0 --busy 1000 --seeds-per-gen 2 --gens 120 --popsize 16              # linear, 9 runs, ~3.5 min
python -m experiments.labelfree train --hidden 8 --busy 1000 --seeds-per-gen 2 --gens 120 --popsize 16 --sigma0 0.5 # one hidden layer of 8, 9 runs, ~4 min
python -m experiments.labelfree static                                   # held-out evaluation, ~10 s
python -m experiments.labelfree figures --arch linear; python -m experiments.labelfree figures --arch mlp8
```

→ `figures/s13_labelfree_stage0.png`, `figures/s13_labelfree_overlay_{linear,mlp8}.png`,
`figures/s13_labelfree_static_{linear,mlp8}.png`, `results/s13_labelfree_*.csv`,
`results/s13_labelfree_params.npz`. Every optimizer configuration run is one row of
`results/s13_labelfree_hparams.csv` (no setting was tried and dropped; the two `--quick`
smoke tests are not logged). Module: `egittins/labelfree.py`; tests: `tests/test_labelfree.py` (10).

Why: the imitation nets regress onto exact Gittins ranks, so they cannot say what a scheduler
needs to *know*; REINFORCE never beat FCFS because its gradient is too noisy. Here the only
training signal is the simulated mean response time, optimized by CMA-ES (derivative-free) with
**common random numbers**: every candidate in a generation is scored on the same seeds, fresh
seeds every generation. Only the ordering of the rank over ages matters to a SOAP policy, so
learned ranks are compared with Gittins by Spearman correlation and, in the overlays, after a
quantile-matching monotone rescale. Gittins is used only in evaluation.

### Stage 0 — positive control on 1-6-14, ρ = 0.8 (`results/s13_labelfree_stage0*.csv`)

Policy class = log-rank at 32 knots over age, linearly interpolated (the RL scratch arm's class),
initialized constant (= FCFS). CMA-ES popsize 16, σ₀ = 1, 150 generations, 4 fresh seeds × 2,000
busy periods per candidate per generation: **19.2 M busy periods per optimizer seed**, 18–20 s each.
Evaluation on 20 fresh trials × 10,000 busy periods, paired against true Gittins on the same seed.

| policy | MRT / true Gittins (mean ± 95% CI) | Spearman vs true Gittins rank |
|---|---|---:|
| FCFS | 1.089 ± 0.001 | – |
| random 32-knot table (control, N(0,1) knots) | 1.067 ± 0.001 | −0.21 |
| CMA-ES, optimizer seed 0 | **1.0048 ± 0.0005** | 0.76 |
| CMA-ES, optimizer seed 1 | **1.0049 ± 0.0005** | 0.62 |
| CMA-ES, optimizer seed 2 | **1.0049 ± 0.0005** | 0.59 |

All three seeds recover both dips of the Gittins rank (ages ≈ 1 and ≈ 6) and the rise after each
mode; the tables are noisy beyond age 8, where almost no scheduling contest happens, which is what
keeps the Spearman at 0.6–0.8 despite the 1.005 ratio. On fixed validation seeds the search mean
goes from 1.017–1.071 (generation 1) to 1.0067 by generation 150 for every seed
(`s13_labelfree_stage0_curves.csv`). The random-table control matters: on 1-6-14 *any*
non-degenerate preemptive table beats FCFS (1.067), so "below FCFS" is a low bar; 1.005 with the
two dips is not. Verdict: the RL scratch arm's failure (1.090–1.159) was the optimizer, not the
policy class.

### Stage 1 — features from a sample window, trained across distributions

Policy = linear map (or one SiLU hidden layer of 8) from the standardized feature tier
(`egittins.imitation.FEATURE_SETS`: tail_only ⊂ no_hazard ⊂ full; 3 / 9 / 11 features, 3 / 9 / 11
or 40 / 88 / 104 parameters) to a rank, tabulated over integer ages from one window of n = 500,
∞ beyond the largest sample. **Training set**: 14 random distributions from the imitation generators
with the held-out parameters excluded (`s13_labelfree_train_dists.csv`: 6 mixtures, 3 bounded
Pareto, 2 Weibull, 2 few-atom discrete, CV² 0.07–3.6, FCFS/Gittins 1.00–2.52), one window each.
Loss = mean over the 14 of MRT / MRT(FCFS) on the same seed. CMA-ES popsize 16, σ₀ = 1 (linear) or
0.5 (MLP), 120 generations, 2 fresh seeds × 1,000 busy periods × 14 distributions per candidate:
**53.8 M busy periods per run**, 21–35 s each; 18 runs = 968 M busy periods (the fixed-seed
validation checks every 10 generations add ~1%). The mean over training distributions of
Gittins/FCFS (the oracle floor, never used in training) is 0.805; final training-validation
losses were tail_only 0.8134 / 0.810–0.814 (linear / MLP), no_hazard 0.8086–0.8088 / 0.8082–0.8088,
full 0.8085–0.8091 / 0.8089–0.8107, all seeds (`s13_labelfree_hparams.csv`).

### Held-out static evaluation (`results/s13_labelfree_static*.csv`, `results/s13_labelfree_ordering.csv`)

Exactly the imitation protocol and trial seeds (20 trials × 10,000 busy periods, n = 500, ρ = 0.8;
all rows paired on the same arrival stream). The three imitation nets are re-simulated here on the
same seeds (their numbers match `s13_imitation_static_summary.csv`). Spearman = mean over trials of
the rank correlation with the true Gittins rank over ages below the sample maximum.

| policy | 1-6-14 | Spearman | bounded Pareto | Spearman |
|---|---|---:|---|---:|
| empirical Gittins (exact) | 1.007 ± 0.003 | 0.94 | 1.023 ± 0.005 | 0.47 |
| **rank by sample mean excess** E[S − a \| S > a] | 1.015 ± 0.003 | 0.80 | 1.033 ± 0.004 | 0.25 |
| imitation, full | 1.011 ± 0.004 | 0.94 | 1.022 ± 0.004 | 0.47 |
| imitation, no hazard | 1.010 ± 0.003 | 0.98 | 1.116 ± 0.022 | 0.53 |
| imitation, tail + mean excess only | 1.095 ± 0.003 | 0.69 | 1.030 ± 0.013 | 0.61 |
| label-free, tail + mean excess, linear, seeds 0 / 1 / 2 | 1.015 ± 0.003 (all three) | 0.79 | 1.034 ± 0.004 (all three) | 0.22 |
| label-free, tail + mean excess, MLP-8, seeds 0 / 1 / 2 | 1.015 ± 0.003 (all three) | 0.78 | 1.023 / 1.022 / 1.021 ± 0.004 | 0.59–0.64 |
| label-free, + quantiles, linear, seeds 0 / 1 / 2 | 1.103 ± 0.003 / 1.073 ± 0.021 / 1.105 ± 0.003 | 0.91–0.93 | 1.013 / 1.013 / 1.014 ± 0.004 | 0.54–0.59 |
| label-free, + quantiles, MLP-8, seeds 0 / 1 / 2 | 1.059 ± 0.026 / 1.088 ± 0.022 / 1.102 ± 0.004 | 0.89–0.92 | 1.016 / 1.016 / 1.019 ± 0.005 | 0.51–0.55 |
| label-free, + hazard (full), linear, seeds 0 / 1 / 2 | 1.081 ± 0.022 / 1.059 ± 0.026 / 1.106 ± 0.005 | 0.90–0.93 | 1.015 / 1.015 / 1.014 ± 0.004 | 0.50–0.56 |
| label-free, + hazard (full), MLP-8, seeds 0 / 1 / 2 | 1.059 ± 0.026 / 1.113 ± 0.005 / 1.090 ± 0.019 | 0.88–0.90 | 1.035 / 1.036 / 1.019 ± 0.008 | 0.30–0.48 |
| FCFS | 1.089 ± 0.001 | – | 5.04 ± 0.14 | – |

Check (`results/s13_labelfree_mean_excess_check.csv`): every tail-only run's rank has Spearman
0.999 (linear) / 0.997 (MLP, 1-6-14) with the sample mean excess, and the same performance to three
decimals; the *true* distribution's mean excess gives 1.014 ± 0.003 (1-6-14) and 1.018 ± 0.003
(Pareto), so on the Pareto about 1.5% of the 3.3% is sampling error in the window.

Reading.
1. **What the label-free search finds is "serve the job with the least expected remaining work"**:
   with tail mass, count and mean excess to work with, all six runs (3 seeds × 2 architectures)
   converge to the ordering by conditional mean excess, which is within 1.5% of Gittins on 1-6-14
   and 3.3% on the Pareto, i.e. 0.8% and 1% behind exact empirical Gittins on the same windows. The
   mean excess is the b = ∞ member of the Gittins family (an upper bound on the rank) and it already
   dips just before each mode of 1-6-14, so it carries most of the "finish the job" structure.
2. **The richer tiers do not help without labels at this budget, and hurt on 1-6-14.** With
   quantiles or hazard features the search reaches a lower *training* loss (0.8085 vs 0.8134) but
   on held-out 1-6-14 lands anywhere from 1.06 to 1.11 depending on the seed, with CIs up to ±0.026
   (the same policy is good on some windows and FCFS-like on others), while on the Pareto it is
   slightly better (1.013–1.016). Fourteen training distributions, dominated by heavy tails where
   "younger first" is nearly optimal, are not enough for the search to find the atom structure that
   the imitation net was handed in its labels. Higher Spearman (0.9 vs 0.8 on 1-6-14) with worse
   response time also says Spearman over all ages is a weak proxy: what matters is the ordering at
   the few ages where jobs actually compete.
3. **This revises the imitation ablation's reading.** The imitation tail-only net (1.095) is far
   worse than ranking by mean excess (1.015) computed from the same three features, so "survivor
   mass and mean excess alone cannot see that an atom is coming" (§13 imitation, above) is not a
   statement about the features; it is a statement about that regression. The features do see it.
   What the hazard and quantile features buy, given labels, is the last 0.5–1%.
4. Not shown: anything at ρ = 0.98, other window sizes, drift, or whether more training
   distributions (or a loss that weights the low-variability cases more than the mean ratio to
   FCFS does) would let the label-free search use the richer features. Stage 2 (a size-aware
   control with SRPT as the known optimum) was skipped: the simulator's rank is a function of age
   only, so size-dependent ranks need a kernel rewrite.
