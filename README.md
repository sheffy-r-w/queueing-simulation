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
pip install -r requirements.txt            # torch is only needed for the imitation and RL experiments
python -m pytest tests -q                 # 33 tests: rank function vs. Definition 2.5, hull vs.
                                          # direct kernel, simulator vs. P-K and E[S]/(1-ρ)
python -m experiments.slide_figures       # figures/s02_spectrum.png, figures/s04_gittins_intuition.png
python -m experiments.eda_tails           # figures/s08_tail_ratio.png, figures/s07_rank_functions.png
python -m experiments.rank_hull           # figures/s05a_rank_hull.png, results/rank_timing.csv
python -m experiments.baseline_comparison # figures/s11_baseline.png, s05b_baseline_teaser.png (~30 min)
python -m experiments.drift_window        # figures/drift_window.png         (~15 min)
python -m experiments.kupdating_drift     # figures/s13_drift_*.png, results/s13_*.csv (~5 min, 12 cores)
python -m experiments.imitation           # figures/s13_rank_overlay.png, results/s13_imitation_*.csv (~10 min; needs torch)
python -m experiments.rl                  # figures/s13_rl.png, results/s13_rl_*.csv (~3 min; needs torch)
python -m experiments.slide_export        # figures/slides/*.png: the same figures without the baked-in headline/subtitle
```

`RESULTS.md` lists every number (with CIs) and the command that produced it.

Every experiment writes its raw results to `results/*.csv` and can re-plot with `--plot`.

## What is implemented

| Concept (paper) | Code |
|---|---|
| Job size distributions on a grid, empirical distribution (Def. 2.1), truncation (Def. 2.4), the 1-6-14 and bounded Pareto distributions of §7.1 | `egittins/distributions.py` |
| Gittins rank function of a discrete distribution (Def. 2.5, Observation A.1), exactly, for every age on the grid; O(K + L) convex-hull kernel with the O(K · L) definition kept as a reference | `egittins/gittins.py` |
| Preemptive M/G/1 under any SOAP policy (§2.1): min-rank service, FCFS tie-break, PLCFS fallback at rank ∞; FCFS and PLCFS baselines | `egittins/simulate.py` |
| §7.1 protocol: 100 trials × (distribution, ρ, n), empirical vs. truncated empirical Gittins vs. true Gittins, FCFS, PLCFS | `experiments/baseline_comparison.py` |
| §8.5 gray-box scheduler: refit empirical Gittins each busy period from the last *P* completed jobs, under distribution drift | `experiments/drift_window.py` |
| Drift models (mean-preserving and one-way weight drift on 1-6-14, mode shift 1-6-14 → 3-8-14, bounded-Pareto tail drift α 2.0 → 1.2) as schedule ∘ family | `egittins/drift.py` |
| k-updating stream engine: genie / FCFS / static fit / k-updating with several windows on one common-random-number stream, paired ratios with t CIs | `egittins/kupdating.py`, `experiments/kupdating_drift.py` |
| Imitation-learned rank function: MLP from features of the conditional-excess sample to log rank, trained on random distributions with exact empirical-Gittins targets; feature ablation | `egittins/imitation.py`, `experiments/imitation.py` |
| REINFORCE on the rank function (softmax over −rank/τ at arrivals and completions, busy-period episodes, return-to-go): from scratch and fine-tuning the imitation net | `egittins/rl.py`, `experiments/rl.py` |
| Data-first view of Lemma 2.3 (empirical vs. true tails and their ratio) and a reproduction of Fig. 1.1 | `experiments/eda_tails.py` |

### Modelling choices worth knowing

- **Grid step h = 0.01.** All job sizes are multiples of h, so every distribution is
  discrete and the Gittins rank is computed exactly rather than approximated. This is
  the paper's own choice (§7.1, footnote 7).
- **Time in quanta of h.** Inter-arrival times are rounded to the nearest quantum, so the
  mean inter-arrival time and hence the load are unbiased.
- **Rank in O(K + L).** For age a with G_{i−1} ≤ a < G_i the rank is
  min_{j ≥ i} (E[S ∧ G_j] − E[S ∧ a]) / (F(G_j) − F(a)): the smallest slope from the
  point (F(a), E[S ∧ a]) to a later point (F(G_j), E[S ∧ G_j]), which lies on the lower
  convex hull of the later points. A right-to-left monotone stack builds each suffix hull
  in amortized O(1); within an interval the query point moves vertically, so the tangent
  only walks toward nearer atoms, over exactly the vertices the next push pops. Both
  numerator and denominator are formed from reverse-accumulated tail quantities
  (P(S > x), E[(S − x)⁺]) so ranks stay accurate where the remaining mass is tiny. The
  hull kernel is bit-identical to the O(K · L) kernel on every test distribution and
  ~3000× faster on the true bounded Pareto (49,801 atoms): 1.1 s → 0.4 ms
  (`results/rank_timing.csv`, `figures/s05a_rank_hull.png`).
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

## k-updating under drift (`experiments/kupdating_drift.py`)

The engine in `egittins/kupdating.py` generalizes the experiment above. A drift model
(`egittins/drift.py`) is a schedule u(k) ∈ [0, 1] over busy periods (triangle wave,
one-way ramp, or constant) composed with a one-parameter family of distributions;
the family is evaluated on a grid of u and the genie policy γ(F_k) is cached per grid
point. In every busy period the genie, FCFS, a **static** empirical Gittins policy
(fit once from the last 500 completed jobs at the start of measurement) and
**k-updating** empirical Gittins with windows w ∈ {50, 200, 500, 2000} all run on the
same seed, so ratios to the genie are paired. Three experiments:

- `headline` — one-way weight drift (0.1, 0.3, 0.6) → (0.6, 0.3, 0.1) on 1-6-14 (variability
  grows over the run), with a stationary control at the final weights
  → `figures/s13_drift_timecourse.png`, `results/s13_headline*.csv`.
- `sweep` — mean-preserving weight drift (load pinned) as a triangle wave with period
  T ∈ {100, 500, 2000, 8000} × window w → `figures/s13_drift_sweep.png`.
- `pareto` — bounded-Pareto tail drift α 2.0 → 1.2 → `results/s13_pareto*.csv`.
- after all three: `figures/s13_drift_types.png`, `results/s13_drift_types.csv` — FCFS, static
  and k-updating (w = 500) side by side under the three kinds of drift.

The load ρ is held fixed (λ_k = ρ / E[F_k]); for the mean-preserving family λ is then
constant too, so only the shape of the distribution changes.

## Imitation-learned rank (`experiments/imitation.py`)

Empirical Gittins is an exact but opaque map (age, sample window) → rank. An MLP is
trained to imitate it from *features* of the conditional-excess sample (S − a | S > a):
survivor mass and count, mean excess, conditional quantiles, and two **hazard**
features (gap to the next sample atom above a, and the mass at it). Targets are
log(r(a) / mean excess) from the exact hull kernel on ~4000 random grid distributions
(Gaussian mixtures, bounded Pareto, lognormal, Weibull, few-atom discrete; the paper's
1-6-14 and Pareto parameters are excluded) with random sample sizes 30–3000. The learned
rank is tabulated over integer ages and run in the same simulator (`policy_from_rank`
computes the re-decision points from the table's up-jumps, so skip-ahead stays exact for
non-monotone tables). The ablation trains the same net on nested feature sets
(tail + mean excess only ⊂ + quantiles ⊂ + hazard) and evaluates each statically (n = 500)
and as a k-updating policy (w = 500) under one-way drift.

## Layout

```
egittins/       distributions.py  gittins.py  simulate.py  drift.py  kupdating.py  imitation.py  rl.py  plotting.py
experiments/    slide_figures.py  eda_tails.py  rank_hull.py  baseline_comparison.py  drift_window.py
                kupdating_drift.py  imitation.py  rl.py
tests/          test_gittins.py  test_simulator.py  test_kupdating.py  test_imitation.py  test_rl.py
figures/        generated PNGs, named sNN_* by slide section (see HANDOFF.md)
results/        generated CSVs and the trained imitation nets (*.pt)
sim/            earlier in-progress design (unchanged)
```
