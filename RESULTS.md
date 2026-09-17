# RESULTS — every number in the deck, with its confidence interval and the command that produced it

All simulations: M/G/1, grid step h = 0.01, inter-arrivals rounded to the nearest
quantum, load ρ = 0.8 unless stated. Every comparison is **paired** (common random
numbers: the same seed, hence the same arrival/size stream, for every policy in a
trial). "MRT" = mean response time. CIs are 95% t intervals over independent trials
unless stated. Machine: Apple Silicon Mac, 14 cores, Python 3.12, numba 0.67.

Run everything from the repo root inside `.venv`:

```bash
source .venv/bin/activate
python -m pytest tests -q                 # 45 tests
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
(distribution, load)), extrapolated from the table: **2,940 core-seconds, ~3.5 min wall on 14
workers** (perfect scaling; ~25 min at the script's default `--workers 2`, matching the "~30 min"
noted above), **486 M simulated jobs** (1-6-14: 47 M at ρ = 0.8, 187 M at 0.98; Pareto: 48 M and
204 M). The ρ = 0.98 cells are 94% of the compute.

---

## §5b, §11  Baseline comparison (paper §7.1 protocol)

**Command:** `python -m experiments.baseline_comparison` (~30 min) → `figures/baseline_comparison.png`, `results/baseline_comparison.csv`

100 trials per (distribution, ρ, n). Each trial: draw n samples, build empirical Gittins
and truncated empirical Gittins (ℓ with Ḡ(ℓ) = n^{-1/3}(1−ρ)^{2/3}), simulate 10,000 busy
periods (4,000 at ρ = 0.98). Ratio = trial MRT / MRT of true Gittins simulated on the *same*
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

**Command:** `python -m experiments.baseline_comparison --loads 0.98 --busy098 40000 --trials 100 --workers 12 --out baseline_comparison_long`
(40 min wall on 12 workers with the reference kernel, of which ~16 min is the two serial 4 × 100,000-busy-period
references) → `results/baseline_comparison_long.csv`, `figures/s11_baseline_comparison_long.png`;
`python -m experiments.baseline_long_report` → `results/baseline_comparison_long_report.csv`.
Same protocol and pairing as above (one seed per trial shared by empirical, truncated and true Gittins), 40,000
busy periods per trial instead of 4,000; the trial seeds are the same as the original run, so the sample windows
are the same and only the arrival streams are longer. Pooled ratio = Σ trial MRT / Σ paired true-Gittins MRT,
bootstrap 95% CI over trials (10,000 resamples). Long references: 1-6-14 246.6 ± 1.7 (was 250.6),
Pareto 146.9 ± 5.2 (was 143.2).

| distribution | n | policy | run | median [IQR] | pooled ratio [95% CI] | P(ratio < 1) |
|---|---:|---|---|---|---|---:|
| 1-6-14 | 10 | empirical | 4,000 bp | 3.68 [2.31, 5.51] | 4.03 [3.61, 4.47] | 0.00 |
| | | | **40,000 bp** | 3.82 [2.64, 6.37] | 4.26 [3.87, 4.66] | 0.00 |
| | 100 | empirical | 4,000 bp | 1.336 [1.12, 1.64] | 1.503 [1.386, 1.629] | 0.10 |
| | | | **40,000 bp** | 1.361 [1.15, 1.73] | 1.522 [1.422, 1.630] | 0.00 |
| | 100 | truncated | 4,000 bp | 1.720 [1.38, 2.30] | 1.937 [1.783, 2.097] | 0.03 |
| | | | **40,000 bp** | 1.798 [1.48, 2.30] | 1.978 [1.848, 2.116] | 0.00 |
| | 1000 | empirical | 4,000 bp | 1.057 [0.98, 1.10] | 1.032 [1.007, 1.058] | 0.32 |
| | | | **40,000 bp** | 1.039 [1.02, 1.07] | 1.052 [1.042, 1.062] | 0.00 |
| | 1000 | truncated | 4,000 bp | 1.338 [1.24, 1.47] | 1.356 [1.307, 1.408] | 0.05 |
| | | | **40,000 bp** | 1.376 [1.31, 1.47] | 1.400 [1.374, 1.426] | 0.00 |
| bounded Pareto | 10 | empirical (= truncated) | 4,000 bp | 1.573 [1.36, 1.86] | 1.669 [1.578, 1.761] | 0.03 |
| | | | **40,000 bp** | 1.587 [1.39, 1.89] | 1.666 [1.596, 1.743] | 0.00 |
| | 100 | empirical | 4,000 bp | 1.136 [1.01, 1.24] | 1.134 [1.090, 1.178] | 0.25 |
| | | | **40,000 bp** | 1.132 [1.08, 1.23] | 1.175 [1.146, 1.210] | 0.00 |
| | 100 | truncated | 4,000 bp | 1.170 [1.08, 1.27] | 1.180 [1.135, 1.229] | 0.18 |
| | | | **40,000 bp** | 1.195 [1.11, 1.27] | 1.221 [1.194, 1.249] | 0.00 |
| | 1000 | empirical | 4,000 bp | 1.112 [1.00, 1.18] | 1.086 [1.051, 1.119] | 0.28 |
| | | | **40,000 bp** | 1.099 [1.07, 1.13] | 1.106 [1.096, 1.116] | 0.00 |
| | 1000 | truncated | 4,000 bp | 1.089 [0.97, 1.15] | 1.055 [1.025, 1.084] | 0.28 |
| | | | **40,000 bp** | 1.067 [1.05, 1.09] | 1.074 [1.067, 1.080] | 0.00 |

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
(w = 2000 is stale: 1.038); at T = 100 the drift is faster than any window can track and the
longest window, which averages over cycles, is best.

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

Reading: on 1-6-14 the tail-only net collapses to FCFS level (1.095 vs FCFS 1.089) — survivor
mass and mean excess alone cannot see "an atom is coming, finish the job"; the quantiles
recover most of it, the hazard features the rest. On the heavy-tailed Pareto the Gittins rank
is nearly monotone in age, so even the tail-only net is close (1.030); the no-hazard net's
wiggles at small ages cost it 1.116.

---

## §13  RL: REINFORCE on the rank function (optional slide)

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

Decision per HANDOFF: fine-tune reproduces across seeds; scratch reproduces its *qualitative*
conclusion (does not beat FCFS) but not its number. Include only as a one-liner ("policy
gradient on busy-period episodes is too noisy to discover the rank structure; imitation
supplies it") or leave out.
