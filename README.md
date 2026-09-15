# queueing-simulation

Simulation of **empirical Gittins** scheduling in the M/G/1 queue, built around

> Shefali Ramakrishna, Amit Harlev, Ziv Scully. *Empirical Gittins for Data-Driven
> M/G/1 Scheduling with Arbitrary Job Size Distributions.* Proc. ACM Meas. Anal.
> Comput. Syst. 10(1), Article 9, March 2026. https://doi.org/10.1145/3788091

Two packages live here:

- **`egittins/`** — a self-contained, tested simulator (this branch). Reproduces the
  paper's Section 7 experiments and runs the Section 8.5 "learning while scheduling"
  experiment. Numba-compiled; the full Section 7 protocol runs in minutes on a laptop.
- **`sim/`** — an earlier, in-progress design (`Job`, `EmpiricalDistribution`). Untouched.

## Quick start

```bash
pip install numpy scipy matplotlib numba pandas pytest
python -m pytest tests -q                 # 18 tests: rank function vs. Definition 2.5,
                                          # simulator vs. Pollaczek-Khinchine and E[S]/(1-ρ)
python -m experiments.eda_tails           # figures/eda_tails.png, figures/rank_functions.png
python -m experiments.baseline_comparison # figures/baseline_comparison.png  (~30 min)
python -m experiments.drift_window        # figures/drift_window.png         (~15 min)
```

Every experiment writes its raw results to `results/*.csv` and can re-plot with `--plot`.

## What is implemented

| Concept (paper) | Code |
|---|---|
| Job size distributions on a grid, empirical distribution (Def. 2.1), truncation (Def. 2.4), the 1-6-14 and bounded Pareto distributions of §7.1 | `egittins/distributions.py` |
| Gittins rank function of a discrete distribution (Def. 2.5, Observation A.1), exactly, for every age on the grid | `egittins/gittins.py` |
| Preemptive M/G/1 under any SOAP policy (§2.1): min-rank service, FCFS tie-break, PLCFS fallback at rank ∞; FCFS and PLCFS baselines | `egittins/simulate.py` |
| §7.1 protocol: 100 trials × (distribution, ρ, n), empirical vs. truncated empirical Gittins vs. true Gittins, FCFS, PLCFS | `experiments/baseline_comparison.py` |
| §8.5 gray-box scheduler: refit empirical Gittins each busy period from the last *P* completed jobs, under distribution drift | `experiments/drift_window.py` |
| Data-first view of Lemma 2.3 (empirical vs. true tails and their ratio) and a reproduction of Fig. 1.1 | `experiments/eda_tails.py` |

### Modelling choices worth knowing

- **Grid step h = 0.01.** All job sizes are multiples of h, so every distribution is
  discrete and the Gittins rank is computed exactly rather than approximated. This is
  the paper's own choice (§7.1, footnote 7).
- **Time in quanta of h.** Inter-arrival times are rounded to the nearest quantum, so the
  mean inter-arrival time and hence the load are unbiased.
- **Skip-ahead simulation.** Between two atoms of the policy's distribution the Gittins
  rank is non-increasing in age (Observation A.1), and waiting jobs' ranks are frozen, so
  the job in service can only lose priority at an arrival or at its next atom. The
  simulator makes decisions only at those events, which is what makes 10,000 busy
  periods per trial cheap. An independent quantum-by-quantum simulator produced
  bit-identical per-job response times on the same random stream.
- **Validation.** FCFS matches Pollaczek–Khinchine and PLCFS matches E[S]/(1−ρ) to
  within simulation noise; the rank function matches a brute-force evaluation of
  Definition 2.5 in exact rational arithmetic; the Fig. 1.1 rank functions reproduce;
  the §7.2 ratios reproduce (e.g. 1-6-14, ρ = 0.8: FCFS/optimal 1.08 here vs. 1.07 in the paper).
- **Discretization lifts the 1-6-14 mean slightly** (mass below 0 is dropped): E[S] ≈ 7.06,
  not 7. Closed forms use `F.mean()`.

## The drift experiment (`experiments/drift_window.py`)

The paper's analysis is one-shot: the job size distribution is fixed. §8.5 proposes,
for the non-stationary case, to rebuild the empirical Gittins policy at the start of
each busy period from the most recent **P** completed jobs, and asks how fast the
distribution can drift while this stays near-optimal.

*P* is a window size, i.e. a hyperparameter trading **variance** (small *P*: noisy
empirical distribution) against **bias** (large *P*: stale, pre-drift data). The
experiment sweeps *P* under a drifting true distribution
F_k = (1−θ_k)·F_A + θ_k·F_B, with θ_k a triangle wave over *T* busy periods
(F_A = 1-6-14, F_B = 3-8-14: "a growing share of traffic comes from a new workload
type"), and reports mean response time relative to an **oracle** that uses the true
Gittins policy for the current F_k. All policies see the same arrival stream.

Why refit only at busy-period boundaries: within a busy period the set of *completed*
jobs is policy-dependent (Gittins finishes short jobs first), so it is a biased sample;
at a boundary every arrival has completed.

## Layout

```
egittins/       distributions.py  gittins.py  simulate.py  plotting.py
experiments/    eda_tails.py  baseline_comparison.py  drift_window.py
tests/          test_gittins.py  test_simulator.py
figures/        generated PNGs        results/   generated CSVs
sim/            earlier in-progress design (unchanged)
```
