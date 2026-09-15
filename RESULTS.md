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
periods (4,000 at ρ = 0.98). Ratio = trial MRT / true-Gittins MRT (true Gittins from
4 × 100,000 busy periods). Median [IQR] and mean ± CI over the 100 trials.

| distribution, ρ | true Gittins MRT | FCFS / opt | PLCFS / opt | n | empirical Gittins | truncated empirical Gittins |
|---|---:|---:|---:|---:|---|---|
| 1-6-14, 0.8 | 27.03 | 1.085 | 1.305 | 10 | 1.360 [1.20, 1.56]; 1.434 ± 0.060 | 1.778 [1.40, 2.01]; 1.705 ± 0.068 |
| | | | | 100 | 1.046 [1.01, 1.08]; 1.050 ± 0.013 | 1.354 [1.28, 1.45]; 1.373 ± 0.027 |
| | | | | 1000 | 0.999 [0.97, 1.02]; 0.999 ± 0.006 | 1.156 [1.14, 1.20]; 1.163 ± 0.011 |
| 1-6-14, 0.98 | 250.6 | 1.116 | 1.408 | 10 | 3.37 [2.16, 4.97]; 3.88 ± 0.46 | 3.37 [2.16, 4.97]; 3.88 ± 0.46 |
| | | | | 100 | 1.329 [1.05, 1.75]; 1.514 ± 0.140 | 1.694 [1.39, 2.26]; 1.951 ± 0.179 |
| | | | | 1000 | 0.955 [0.83, 1.13]; 1.000 ± 0.048 | 1.213 [1.06, 1.52]; 1.314 ± 0.080 |
| bounded Pareto, 0.8 | 25.38 | 5.142 | 1.583 | 10 | 1.166 [1.12, 1.26]; 1.224 ± 0.039 | 1.247 [1.16, 1.34]; 1.262 ± 0.025 |
| | | | | 100 | 1.052 [1.01, 1.10]; 1.058 ± 0.012 | 1.155 [1.11, 1.20]; 1.158 ± 0.014 |
| | | | | 1000 | 1.040 [0.99, 1.08]; 1.037 ± 0.012 | 1.098 [1.04, 1.14]; 1.096 ± 0.013 |
| bounded Pareto, 0.98 | 143.2 | 10.53 | 2.805 | 10 | 1.406 [1.10, 1.81]; 1.568 ± 0.140 | 1.406 [1.10, 1.81]; 1.568 ± 0.140 |
| | | | | 100 | 0.942 [0.77, 1.17]; 1.036 ± 0.084 | 0.996 [0.78, 1.28]; 1.078 ± 0.090 |
| | | | | 1000 | 0.965 [0.76, 1.18]; 1.005 ± 0.069 | 0.924 [0.75, 1.16]; 0.976 ± 0.061 |

Notes. Ratios below 1 at ρ = 0.98 are simulation noise (heavy-tailed response times, few
busy periods per trial); the paired comparison *within* a trial is what is reliable there.
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

Component weights move linearly (0.6, 0.3, 0.1) → (0.1, 0.3, 0.6) over the run; the
stationary control holds (0.6, 0.3, 0.1).

| policy | one-way drift | stationary |
|---|---|---|
| FCFS | 1.109 ± 0.003 | 1.417 ± 0.003 |
| static (w = 500, fit once) | **1.188 ± 0.012** | 1.020 ± 0.004 |
| k-updating, w = 50 | 1.117 ± 0.005 | 1.141 ± 0.006 |
| k-updating, w = 200 | 1.030 ± 0.002 | 1.036 ± 0.002 |
| k-updating, w = 500 | **1.013 ± 0.001** | 1.014 ± 0.001 |
| k-updating, w = 2000 | 1.005 ± 0.001 | 1.004 ± 0.001 |

Time course (`results/s13_headline_blocks.csv`, blocks of 100 busy periods pooled over
trials): static rises from 1.02 to 1.36 by the end of the run; w = 500 stays within 1.00–1.03
throughout.

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

| policy | ratio |
|---|---|
| FCFS | 1.105 ± 0.004 |
| static empirical Gittins | 1.187 ± 0.015 |
| k-updating empirical Gittins, w = 500 | 1.013 ± 0.002 |
| k-updating NN full, w = 500 | **1.021 ± 0.002** |
| k-updating NN no hazard, w = 500 | 1.026 ± 0.002 |
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
