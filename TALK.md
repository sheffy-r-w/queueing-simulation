# TALK — P&G technical presentation, Fri 2026-09-18, 12–1pm ET

Outline first, then Q&A prep. Figures are `figures/sNN_*.png`; every number is in
`RESULTS.md` with its CI and the command that produced it. The deck follows this
order slide for slide (appendix slides A1–A5 are for questions only).

Running example throughout: a **shared compute cluster / batch-job queue**. Job = a
submitted run, size = the work it needs (unknown until done), age = work done so far,
response time = submit-to-done, load ρ = fraction of time the worker is busy, busy
period = a burst of back-to-back work between idle moments, preemption = pausing one
job to run another. Pick queues and ticket routing appear only on the "where it
applies" slide, with the non-preemptive variant.

---

## Outline (~39 min + Q&A)

| # | Slide | On it | The sentence to land | min |
|---|---|---|---|---|
| 0 | Title | "How to order a queue well when you don't know how long each job takes, but you have a log of past jobs." | Practical question, clean answer. | 0.5 |
| 1 | The problem, in a picture | Compute queue; five jobs 1,1,1,6,14 present at once: longest-first mean 20.0, shortest-first 7.6 (2.6×). | Ordering is a free lever, and the right order depends on what you know about sizes. | 2 |
| 2 | Vocabulary, once | job, size, age, response time, load ρ, busy period, preemption — each with the compute-queue word. | Seven words; everything after uses them. | 1.5 |
| 3 | Spectrum of information (`s02_spectrum.png`) | Exact sizes → SRPT; distribution → Gittins; sample → this work; nothing → FCFS/PS. | Real systems have samples, and nobody had said what to do with them. | 2 |
| 4 | What Gittins does (`s04_gittins_intuition.png`) | Rank(age) ≈ expected remaining work ÷ chance of finishing soon; walk one job through 1-6-14; point at the non-monotone shape. | Gittins is expected cost per unit of progress; it needs the shape, not the mean. | 2.5 |
| 5 | Do the obvious thing (statement) | Replace the true distribution with the histogram of the last n jobs. Three questions: does it work, why, what if the world changes. | The whole project is those three questions. | 1 |
| 6 | First experiment (`s05b_baseline_teaser.png`) | 1-6-14, ρ = 0.8, 100 paired trials: n = 10 → 1.39×, n = 100 → 1.04×, n = 1000 → 1.005×; FCFS 1.08, PLCFS 1.31. | A hundred past jobs already gets within 5% of the optimum. | 2 |
| 7 | Toolbox: tools and why each | Python 3.12 + numpy/scipy (exact discrete distributions); numba (the hot path: 10k busy periods < 1 s); multiprocessing (trials over 12 cores); pandas (CSV per run, `--plot` re-renders); matplotlib (one style, one colour per policy); pytest (45 tests); torch (only the two learning experiments); git (branch → PR; older prototype reimplemented in one package). | Not the list, the reasons: numba because the loop is the hot path, CSV so figures never need a re-simulation, one package so every figure shares one simulator. | 0.5 |
| 8 | Implementation II: simulator and validation | Event-driven, skip-ahead, numba; common random numbers. Table: FCFS = Pollaczek–Khinchine; PLCFS = E[S]/(1−ρ); Gittins on constant sizes = FCFS exactly; rank vs brute-force definition 1e-10; independent simulator bit-identical; paper Fig 7.2 reproduced (1.085 vs 1.074). 45 tests. | Before trusting any result the simulator had to reproduce things I could compute by hand. | 2.5 |
| 9 | Implementation III: rank in O(n) (`s05a_rank_hull.png`) | Rank at an atom = min slope on the curve (F(x), E[S∧x]) → lower convex hull, one backward pass. True Pareto 1.2 s → 0.4 ms (3,300×); 1-6-14 80×; bit-identical. | The exact rank function is a convex-hull sweep, so refitting is essentially free. | 2 |
| 10 | Methods tried: the proof route that failed (`s07_rank_functions.png`) | Standard argument: rank functions close ⇒ performance close. Empirical rank function never converges (atoms), yet the policy is within 1%. Framing: rank function ~ density, tail ~ CDF; empirical CDFs converge, histograms don't. | The obvious argument is false, even though the experiment says the policy is fine. | 2 |
| 11 | What is stable: tail ratios (`s08_tail_ratio.png`) | Empirical/true survival ratio stays near 1 far into the tail. Bound in words: gap shrinks like a power of 1/n, constant grows as ρ → 1, provided the tail is cut at a level set by n and ρ. Theorem on A1. | You don't need the rank functions to agree; you need the tails to agree in ratio, and they do. | 2 |
| 12 | Theory vs practice: truncation | Paired medians, n = 1000: 1-6-14 ρ 0.8 1.005 vs 1.170; Pareto ρ 0.8 1.017 vs 1.077; 1-6-14 ρ 0.98 1.057 vs 1.338; Pareto ρ 0.98 1.112 vs 1.089 (CIs overlap). | The safety margin the theorem needs is a real cost; know when to ignore your own theorem. | 2 |
| 13 | Results in full (`s11_baseline.png`) | Rules of thumb: ~100 samples → ~5%; ~1000 → 1–2% at ρ 0.8; ρ 0.98 needs ~10× more data; heavy tails: FCFS 5–10× worse than any of these. | Even tiny samples beat the default by a lot; returns flatten fast, except near saturation. | 3 |
| 14 | Correction: pairing | Unpaired ratios put medians below 1 at ρ 0.98; a single 4,000-BP run of the optimum itself lands at 0.87–0.94 of its long-run mean (SD 0.25–0.30). Fix: optimum on the same arrival stream per trial; every median ≥ 1. | Heavy tails make short-run averages lie; pair your comparisons or you will publish noise. | 1.5 |
| 15 | The world changes (`s13_drift_timecourse.png`) | Weights (0.1,0.3,0.6) → (0.6,0.3,0.1) over 4,000 BPs, 40 trials. Static fit ≈ FCFS, both climb to 1.41; refit w = 500 stays within 1–3% (1.012 whole run). Refit at busy-period boundaries only. | A policy fitted once behaves like the default once the workload moves; refitting from a rolling window keeps you at the optimum. | 2.5 |
| 16 | Tuning the window (`s13_drift_sweep.png`) | w = 50 variance-limited (~1.12); T = 500 → best w = 500 (w = 2000 stale, 1.038); T = 100 nothing tracks, longest window wins by averaging. Rule: w ≈ 500 jobs ≈ 100 BPs. | It's retraining cadence: too short is variance, too long is lag, and you can measure both. | 2 |
| 17 | What drifts matters (`s13_drift_types.png`) | Mix shift: static 1.113 ≈ FCFS 1.105; shape drift, load pinned: static 1.042; heavier tail: FCFS 3.58 but static 1.039. Refit-500 within 1–2.5% in all three. | Not every drift needs a refit; the ones that change the shape do. | 2 |
| 18 | A learned rank function, as an ablation (`s13_rank_overlay.png`) | MLP on features of the recent sample → log rank; 4,000 synthetic distributions, 1-6-14 and Pareto held out. R² 0.26 → 0.84 → 0.99 as quantiles then hazard features are added; tail-only net collapses to FCFS (1.095 vs 1.089). As refit policy: NN 1.017 vs exact 1.011. RL: one sentence. | The exact rank costs 0.4 ms, so this is not for deployment: the ablation says which information Gittins needs — "is an atom coming?". | 2.5 |
| 19 | Practical takeaways | (1) Run Gittins on the histogram; 100 jobs to start. (2) Don't truncate at moderate load. (3) Refit from ~500 jobs at idle moments, never mid-burst. (4) Near-constant sizes: FCFS is already optimal. (5) Pair your simulations. | Here is what to do on Monday. | 1.5 |
| 20 | Next steps and where it applies | Borg traces (BigQuery): fit week 1, evaluate week 2. Multi-server. Non-preemptive variant. Fits: batch/data pipelines, pick queues, ticket routing. | Same recipe, real logs; the open question is multi-server. | 1.5 |
| 21 | Thank you | Citation, repo, one-line summary. | — | 0.5 |

Appendix: A1 theorem (replace the bracketed line with the statement from the paper),
A2 hull algorithm in three lines, A3 RL table, A4 the two drift experiments that
misled first, A5 why analytical rather than learned.

Timing: 0–5 problem 9.5 · 6–9 first result and implementation 7 · 10–12 methods and
truncation 6 · 13–14 results and pairing 4.5 · 15–17 drift 6.5 · 18 learned 2.5 ·
19–21 wrap 3.5 ≈ 39.5 min. Thirty-minute version: drop 18, fold 17 into one sentence on
15, fold 14 into a callout on 13. Confirm the talk/Q&A split with Felix or Luis.

---

## Q&A prep

### Likely questions, one line each (full answers below)
- Why not just learn the policy? → four reasons, next section.
- Why one server? → theory is single-server; estimation and refit carry over; multi-server is next.
- Is preemption realistic? → compute jobs yes; pick queues need the non-preemptive variant.
- Why not predict each job's size with a model? → complementary; that needs per-job features, this needs only history.
- How do you know the simulator is right? → closed forms, brute-force definition, independent simulator, the paper's figure.
- Why refit only at idle moments? → completed jobs mid-burst are a biased sample.
- What is truncation and why skip it? → the proof's tail cut; costs 6–17% at ρ 0.8, breaks even only near saturation on heavy tails.
- What breaks it? → correlated sizes, ρ near 1 with few samples, drift faster than the window, near-constant sizes (nothing to gain).
- Why these two distributions? → the paper's test bed: multimodal and heavy-tailed, where FCFS fails differently.
- How long does it run? → baseline ~30 min, drift ~5 min on 12 cores, imitation ~10 min, RL ~3 min.
- What would you do differently? → pair from day one; design drift experiments by checking the baseline on the endpoints first.
- What are the Borg traces and what exactly would you do? → BigQuery; fit week 1, evaluate week 2, then the window on real drift.
- Did you use AI coding tools? → answer plainly, see below.
- Where would this apply here? → batch/data pipelines, pick queues, ticket routing; one sentence each.

### Did you use AI coding tools?
Say what is true, briefly. A good shape: "Yes — for scaffolding, plotting code and refactoring. The algorithm (the hull sweep), the validation design, the experiment design and every number are mine and checked against closed forms and an independent simulator." Interviewers ask this now; a clear, unembarrassed answer reads as maturity.

### Why analytical, when you could just learn the optimal policy?
1. Learning takes a long time. Policy gradient on busy-period episodes never beat FCFS
   in 2,000 iterations × 1,024 episodes; the empirical policy needs ~100 samples and no
   training.
2. The data-driven approximation only exists because the analytical work found the
   optimal policy first. "Plug in the histogram" is a one-line idea with a provable
   guarantee *because* we know the answer is Gittins.
3. Analytical solutions give insight. The rank function says what information matters
   (the ablation confirmed it: hazard, not just tail mass), when FCFS is already fine
   (low variability), and why truncation costs what it costs.
4. Given all the information the scheduler has, learning converges to Gittins anyway.
   The analytical route gets there first and explains it.

### Tools and libraries, and why each
- **numpy / scipy**: distributions on a grid (step 0.01), discretization of the 1-6-14
  Gaussian mixture and bounded Pareto via their CDFs, prefix sums for the hull.
- **numba**: the simulator's event loop and the rank kernels are `@njit`; 10,000 busy
  periods in under a second. Without it the Section 7 protocol (100 trials × 12 cells)
  would take hours instead of ~30 minutes.
- **pandas**: every experiment writes tidy CSVs to `results/`; plots re-run from CSVs
  with `--plot` so figures are reproducible without re-simulating.
- **matplotlib**: one shared style (`egittins/plotting.py`): one colour per policy on
  every figure (blue = empirical/refit, orange = truncated/static, grey = FCFS, dashed
  black = optimal).
- **pytest**: 45 tests; rank vs brute-force Definition 2.5 in exact arithmetic, FCFS vs
  Pollaczek–Khinchine, PLCFS vs E[S]/(1−ρ), truncation definition, monotonicity between
  atoms.
- **torch**: the imitation MLP (3 × 128, SiLU) and REINFORCE; ~10 min and ~3 min runs.
- **multiprocessing**: trials in parallel across 12 cores for the drift sweeps.
- **git**: `presentation` branch, PR-merged to main; the older `gittins-lab` prototype
  was reimplemented here rather than copied so every figure shares one simulator, one
  random-number discipline and one style.

### Simulator design
- Event-driven, fully preemptive, one server, Poisson arrivals at λ = ρ / E[S].
- Time in quanta of the grid step; inter-arrivals rounded to the nearest quantum so the
  load is unbiased (< 0.2% effect at these sizes).
- **Skip-ahead**: between atoms of the policy's distribution the Gittins rank is
  non-increasing in age and waiting jobs' ranks are frozen, so the job in service can
  only lose priority at an arrival or at its next atom. Decisions happen only at those
  events. An independent quantum-by-quantum simulator gave bit-identical per-job
  response times.
- Ties at a finite rank broken FCFS; rank ∞ (past the last atom of the empirical
  distribution) falls back to preemptive LCFS, as in the paper.
- Warm-up of 600 busy periods before measurement in the drift runs.

### Validation, in one breath
FCFS matches the closed form; PLCFS matches E[S]/(1−ρ); Gittins on a deterministic size
is exactly FCFS; the rank kernel matches a brute-force evaluation of the definition to
1e-10; an independent simulator is bit-identical; the paper's Fig 7.2 reproduces
(1.085 vs 1.074, the difference from discretization lifting E[S] to 7.06).

### The O(n) rank computation
Rank at atom x_j = min over later k of (E[S∧x_k] − E[S∧x_j]) / (F(x_k) − F(x_j)): the
minimum slope from point j to a later point on the curve (F(x), E[S∧x]). The minimum
slope to a set of later points is attained on their lower convex hull, so one backward
monotone-stack pass gives all ranks in O(K) after sorting; between atoms the rank is
linear in age for a fixed argmin atom, so integer ages follow in O(K + L) versus
O(K·L) for the definition directly. Bit-identical output; true Pareto (49,801 atoms,
50,001 ages) 1.2 s → 0.4 ms.

### Why refit only at busy-period boundaries?
Within a busy period the set of *completed* jobs is policy-dependent: Gittins finishes
short jobs first, so the completed set is biased short. At a boundary every arrival has
completed, so the window is an unbiased sample.

### Why pairing / common random numbers?
Response times are heavy-tailed at high load; one 4,000-busy-period run of the optimal
policy itself lands at 0.87–0.94 of its long-run mean (SD 0.25–0.30 across seeds). Same
seed → same arrival and size stream for every policy in a trial; ratios are then
trial-by-trial. Same idea as matched-pairs A/B testing.

### Why one server?
The optimality theory is single-server. The estimation idea (plug in the histogram) and
the refit engine carry over unchanged; multi-server Gittins is the next-steps slide.

### Is preemption realistic?
For compute and batch jobs, yes. For a picker mid-order, less so; Gittins has
non-preemptive variants and the *estimation* question is the same. That is why the
running example is a compute queue and pick queues are on the applications slide.

### How does this compare with predicting each job's size?
Predictions feed SRPT-with-predictions; that needs per-job features and a model. This
needs only the history. They are complementary: Gittins can take a predicted-size
distribution per job class as its input.

### Why 1-6-14 and bounded Pareto?
The paper's test bed: one multimodal (three job types, where FCFS is 8% off and the
rank function is non-monotone), one heavy-tailed (where FCFS is 5–10× off). They are
the two regimes where FCFS fails differently.

### What breaks it?
Sizes correlated with arrivals or with each other; load near 1 with few samples (the
constant in the bound grows as ρ → 1); drift faster than the window can track
(T = 100 in the sweep); a workload with near-constant sizes gains nothing (FCFS is
already optimal there — the mode-shift experiment in A4 is the example).

### Truncation: what is it and when does it matter?
Definition: move all empirical mass at or above ℓ to a single atom at ℓ, with ℓ chosen so
the empirical survival function there equals n^(−1/3)(1−ρ)^(2/3). The proof needs it
because the tail ratio blows up where the sample runs out. In practice it costs 6–17%
at ρ = 0.8 and 28% on 1-6-14 at ρ = 0.98; it breaks even only on the heavy tail near
saturation (Pareto, ρ = 0.98: 1.112 vs 1.089, CIs overlap). At n = 10 and ρ = 0.98 the
cut exceeds every sample, so truncated = untruncated.

### The learned rank function: what exactly?
Features of the conditional-excess sample at age a: log survivor mass Ḡ(a), log count,
log mean excess m; conditional quantiles q10, q25, q50, q75, q90, max (÷ m); hazard
features: gap to the next sample atom above a (÷ m) and the mass at it. Target
log(rank / m). 4,000 random distributions (Gaussian mixtures, bounded Pareto,
lognormal, Weibull, few-atom discrete), sample sizes 30–3,000, 254k rows, 40 epochs,
10% held out; the 1-6-14 and Pareto(2, 1.2, 500) parameters never in training.
Held-out R²: 0.26 (tail + mean excess) → 0.84 (+ quantiles) → 0.991 (+ hazard).

### RL, if asked
Stochastic policy: at each decision the served job is drawn ∝ exp(−rank(age)/τ).
REINFORCE, busy-period episodes, return = −(sum of response times), return-to-go,
batch-normalized advantages, Adam with cosine decay, 2,000 iterations × 1,024 episodes.
From scratch (32 knots over age): two seeds put the maximum rank at age 0 so a new
arrival never preempts and the policy executes as FCFS (1.090); the third is worse
(1.159). Fine-tuning the imitation net holds 1.002–1.008. A one-parameter-per-age
variant diverged. Conclusion: the policy-gradient signal from busy-period episodes is
too noisy to discover the rank structure; imitation supplies it.

### Borg traces: what exactly would you do?
Google's cluster traces are public on BigQuery. Pull task run times per job class for
one week, fit the empirical policy, evaluate on the next week in the simulator with
the real arrival process; compare to FCFS and to a fit on the same week (the genie
proxy). Then the refit window on the real drift.

### Where this applies at P&G (say it once, briefly)
Batch and data pipelines with run-time history (preemption is cheap there);
fulfilment and pick queues with historical pick times (non-preemptive variant);
ticket routing with historical handle times. Same recipe: histogram, Gittins, refit.

### Numbers to have cold
- 1-6-14, ρ 0.8: FCFS 1.085, PLCFS 1.305; empirical Gittins n = 10/100/1000 medians
  1.387 / 1.044 / 1.005.
- Pareto, ρ 0.8: FCFS 5.14, PLCFS 1.58; n = 10/100/1000: 1.161 / 1.046 / 1.017.
- Pareto, ρ 0.98: FCFS 10.5, PLCFS 2.81; n = 10/100/1000: 1.573 / 1.136 / 1.112.
- Drift headline (40 × 4,000): FCFS 1.105, static 1.113, w = 50/200/500/2000:
  1.115 / 1.029 / 1.012 / 1.004.
- Hull timing: 1-6-14 1.16 ms → 0.015 ms; true Pareto 1,175 ms → 0.36 ms.
- Runs: baseline ~30 min; drift ~5 min on 12 cores; imitation ~10 min; RL ~3 min.
