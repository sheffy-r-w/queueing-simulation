# HANDOFF — presentation build for the P&G technical presentation (Fri Sept 18, 12–1pm ET)

This file is the bridge between the Cowork session that built the `presentation`
branch and whoever continues the work (Shefali, or Claude Code on her Mac).
Read it fully before running anything.

## Goal

A single coherent codebase (`egittins/`) that produces every figure and number for
Shefali's talk, following her 15-section outline (below). Nothing is to be copied
wholesale from the older `gittins-lab` project; its ideas (O(n) rank, k-updating,
drift sweeps, imitation policy, RL) are to be **reimplemented inside `egittins`**
so all figures share one simulator, one random-number discipline, and one style.

Audience: ~12 P&G data scientists (supply chain, retail, media). Felix's brief:
show the problem, initial thinking/EDA, methods tried, tools/libraries,
tuning/corrections/validation, implementation, results. Assess technical mastery
and clear communication. No Kaggle.

## What is DONE and validated (do not redo)

| Piece | Where | Status |
|---|---|---|
| Grid distributions: 1-6-14, bounded Pareto, empirical, truncation | `egittins/distributions.py` | done, tested |
| Exact Gittins rank on integer ages (O(K·ages) numba kernel) | `egittins/gittins.py::_rank_kernel` | done; matches Def. 2.5 brute force in exact arithmetic (independent review) |
| Event-driven SOAP M/G/1 simulator, FCFS/PLCFS, closed-form checks | `egittins/simulate.py` | done; bit-identical to an independent quantum-stepping simulator; P-K and E[S]/(1-ρ) tests pass |
| EDA tail-ratio figure + Fig 1.1 reproduction | `experiments/eda_tails.py` → `figures/eda_tails.png`, `figures/rank_functions.png` | done |
| Section 7.1 protocol, 100 trials × 10k busy periods (3k at ρ=0.98) | `experiments/baseline_comparison.py` → `results/baseline_comparison.csv`, `figures/baseline_comparison.png` | done; reproduces paper Fig 7.2 (e.g. 1-6-14 ρ=0.8: FCFS/opt 1.084 vs paper 1.074) |
| Section 8.5 window-size sweep (mode-shift drift, ρ=0.8, 5 seeds × 6000 BPs) | `experiments/drift_window.py` → `results/drift_window.csv`, `figures/drift_window.png` | done |
| Tests (18) | `tests/` | `python -m pytest tests -q` |

Cross-checks against gittins-lab (its code lives in Shefali's zip, not in this repo):
rank functions agree to 1e-9 on random pmfs and a 300-sample empirical Pareto;
true-Gittins MRT 27.03 (here) vs 26.92 (gittins-lab big run) on 1-6-14 ρ=0.8;
FCFS/opt 1.084 vs 1.089.

Known discrepancy to be aware of: gittins-lab's *static* table reports empirical
Gittins n=10 at 1.13 on 1-6-14 ρ=0.8; the paper's Fig 7.2(a) and this repo both
give ~1.36 (median). Use paper-protocol numbers (this repo) on the static slide.

## What is NEXT (in order)

### A. O(n) convex-hull rank computation (Shefali's algorithm)
Rank at atom x_j = min over later atoms k of
(E[min(X,x_k)] − E[min(X,x_j)]) / (F(x_k) − F(x_j)): the minimum slope from point
j to any later point on the curve (F(x), E[min(X,x)]), which is attained on the
lower convex hull of the suffix → one backward monotone-stack pass, O(n) after
sorting. Between atoms the rank is linear in age *for a fixed argmin atom*, so
integer-age tabulation follows from the hull result; verify against
`_rank_kernel` (must agree to ~1e-12) and make it the default in `gittins_policy`.
Add `figures/s05a_rank_hull.png`: the (F, E[min]) points, the hull, one slope line.

### B. k-updating ("gray-box") engine + drift experiments
`egittins/kupdating.py`: stream of busy periods with a drifting true distribution
F_k; policies run on the SAME arrival/size stream (common random numbers, seed per
busy period): genie (true Gittins for current F_k), static empirical fit at t=0,
k-updating with window w (refit at each busy-period start from the last w
completed sizes), FCFS. Report paired MRT ratios vs genie with 95% t-CIs over
trials. Drift models to implement in `egittins/drift.py`:
  - mean-preserving weight drift on 1-6-14 (triangle wave; E[X] fixed so load is pinned)
  - one-way weight drift (0.6,0.3,0.1) → (0.1,0.3,0.6) over the run
  - mode shift 1-6-14 → 3-8-14 (already in `experiments/drift_window.py`, fold it in)
  - bounded-Pareto tail drift α 2.0 → 1.2
Experiments: (1) one-way drift headline table; (2) drift-period {100,500,2000,8000} ×
window {50,200,500,2000} sweep → `figures/s13_drift_sweep.png`; (3) Pareto drift table.
Scale: ≥20 trials × ≥4000 busy periods per cell for the headline; 6+ trials × 2500 for the sweep.

### C. Imitation-learned rank policy (PyTorch)
`egittins/imitation.py`: an MLP mapping features of (age a, sample window) → log rank.
Features: survivor mass Ḡ(a), conditional-excess quantiles of (S−a | S>a),
mean excess, and **hazard features** (gap to the next sample atom above a, and
the mass at it). Train on ~4k random discrete distributions (mixtures, Pareto-like,
lognormal-like; hold out the 1-6-14 and Pareto parameters), targets = exact
empirical-Gittins ranks. Evaluate as a SOAP policy (tabulate rank over integer
ages) in the simulator: static and k-updating (w=500). Ablation: same net without
hazard features (expected: collapses toward FCFS on 1-6-14). Figures:
`figures/s13_rank_overlay.png` (true Gittins vs NN), performance table.

### D. RL — time-boxed to Wednesday night; include only if it reproduces
Stochastic policy over the same rank-function class (softmax over −rank/τ at
decision points), REINFORCE with busy-period episodes, reward = −(area under the
queue-length curve), return-to-go, batch-normalized advantages. Arms: from scratch;
fine-tune from the imitation net. 3 seeds each. Expected: scratch → monotone rank
≈ FCFS (~1.09), fine-tune ≈ true Gittins (~1.00). If either does not reproduce
across seeds, leave RL out of the deck.

### E. Deliverables
- `figures/sNN_*.png` named by outline section (see below), consistent style
  (`egittins/plotting.py`), slide-sized.
- `RESULTS.md`: every number with CI and the exact command that produced it.
- README updated; tests green; commit; push `presentation`.

## Shefali's outline (figure names should map to these numbers)

1. Queueing theory and why it matters (running example: a support/pick queue)
2. Spectrum of information: exact sizes → SRPT; distribution → Gittins; samples → this work; nothing → FCFS/PS  (`s02_spectrum.png`)
3. The natural, important, unsolved regime: samples only
4. First idea: plug in the empirical distribution; what Gittins is, rank-function intuition (`s04_gittins_intuition.png`)
5. EDA/tools: does it perform well? yes. Simulator design, O(n) rank precomputation, validation (`s05a_rank_hull.png`, `s05a_validation` table, `s05b_baseline_teaser.png`)
6. How similar results were proven before: rank functions
7. The rank functions are different (`s07_rank_functions.png`)
8. What IS similar: the ratio of complementary CDFs (`s08_tail_ratio.png`)
9. The bounds (one slide, in words)
10. Practical takeaways: sample complexity, truncated vs untruncated, load
11. Performance results in full (`s11_baseline.png`)
12. The distribution may also drift: what then?
13. Empirical: static vs k-updating vs genie; window sweep; imitation (and RL if it reproduces) (`s13_*.png`)
14. Next steps: Google Borg traces (BigQuery), real workloads
15. Practical wrap-up and where this applies

## Conventions
- Grid step h = 0.01; time in quanta of h; inter-arrivals rounded to nearest quantum.
- Every comparison paired via common random numbers; report ratios with CIs.
- Slides are Shefali's job; this repo delivers figures, numbers, and code.
- Keep `sim/` untouched (her earlier design).
