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
