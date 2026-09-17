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

## Outline (~38 min + Q&A)

Each slide's title is the question it answers; its last line is the question the next slide answers. Slide 4 poses the three questions the rest of the talk answers in order.

| # | Title (the question) | On it | Closing line (next slide's question) | min |
|---|---|---|---|---|
| 0 | Data-driven scheduling from samples (or, how to schedule when all you have is history) | Name, Cornell, P&G technical presentation, date; theory from the paper; simulator/experiments/deck built this week. | — | 0.5 |
| 1 | Why does the order of a queue matter? | Animation: five jobs 1,1,1,6,14, two service orders; longest-first mean 20.0, shortest-first 7.6. Labels on the picture: size, response time, preemption. | Shortest-first needs every job's size. What do we actually know about sizes? | 2.5 |
| 2 | What do we know about job sizes? | Spectrum (`s02_spectrum.png`): exact sizes → SRPT; distribution → Gittins; sample → this work; nothing → FCFS/PS. | In practice: a log of past sizes, and no one had said what to do with it. The nearest solved case is knowing the distribution — what does that look like? | 2 |
| 3 | What's optimal when you know the distribution? | Bare `slides/s04_gittins_intuition.png` with the walk-through annotations added in PowerPoint; define age here (the x-axis). Rank ≈ expected remaining work ÷ chance of finishing soon; lowest rank runs. | A rank function computed from the distribution. We have a sample, not the distribution. Can we just plug in the histogram? | 2.5 |
| 4 | The obvious thing: run Gittins on the histogram of the last n jobs | Three questions: does it work; why; what if the distribution changes. | (the spine of the talk) | 1 |
| 5 | Question one: does it work? | `s05b`: n = 10 → 1.39×, 100 → 1.04×, 1000 → 1.005×; FCFS 1.08, PLCFS 1.31; 100 paired trials. | Yes, in simulation. So: what did I build, and why should you believe it? | 2 |
| 6 | What did I build? | Event-driven simulator, one package, one script per figure; toolbox table with the reason per tool (numba, multiprocessing, pandas + --plot, matplotlib, pytest 45, torch, git). | …and why should you believe it? | 1.5 |
| 7 | Why should you believe the simulator? | Validation table: P-K, E[S]/(1−ρ), Gittins on constant sizes = FCFS, brute-force definition 1e-10, independent simulator bit-identical, paper Fig 7.2 (1.085 vs 1.074). | It refits a rank function thousands of times. Is that expensive? | 2 |
| 8 | Is the rank function expensive to compute? | `s05a`: hull sweep; bit-identical; true Pareto 1.2 s → 0.4 ms (3,300×). | No. It works and it's cheap. Question two: why does it work? | 2 |
| 9 | Why does it work? Are the rank functions close? | `s07`: empirical rank function never converges (sawtooth), policy within 1%. Rank function ~ density, tail ~ CDF. | No — they never converge. Then what is? | 2 |
| 10 | What is close? | `s08`: survival ratio near 1 far into the tail; bound in words (power of 1/n, constant grows as ρ → 1, tail cut at a level set by n and ρ). | The tails, in ratio. That gives a bound — if you cut the tail off. Does the cut cost anything? | 2 |
| 11 | Does the cut cost anything? | Table, n = 1000 medians: 1-6-14 ρ 0.8 1.005 vs 1.170; Pareto ρ 0.8 1.017 vs 1.077; 1-6-14 ρ 0.98 1.057 vs 1.338; Pareto ρ 0.98 1.112 vs 1.089. | Yes, at moderate load. So across loads and workloads: how much data do you need? | 2 |
| 12 | How much data do you need? | `s11` four panels; define load here. ~100 → ~5%; ~1000 → 1–2%; ρ 0.98 needs ~10× more; heavy tails: FCFS 5–10× worse. | About a hundred jobs. Every comparison here is paired; the first version wasn't, and it was wrong. How? | 3 |
| 13 | What went wrong unpaired? | Optimum's own 4,000-BP run at 0.87–0.94 of its long-run mean (SD 0.25–0.30); common random numbers; every median ≥ 1. | Fixed. Question three: what if the distribution changes? | 1.5 |
| 14 | What happens when the workload drifts? | `s13_drift_timecourse`: static ≈ FCFS, both to 1.41; refit w = 500 within 1–3% (1.012). Define busy period here; refit only at idle moments. | Refit from a rolling window. How big? | 2.5 |
| 15 | How big a window? | `s13_drift_sweep`: w = 50 ~1.12 everywhere; T = 500 → w = 500 best; T = 100 nothing tracks. Rule: ~500 jobs ≈ 100 busy periods. | About 500 jobs. Does every kind of drift need this? | 2 |
| 16 | Which drifts actually hurt? | `s13_drift_types`: mix shift static 1.113 ≈ FCFS 1.105; shape drift static 1.042; heavier tail FCFS 3.58, static 1.039; refit within 1–2.5%. | The ones that change the shape. Which parts of the shape does the policy actually use? | 2 |
| 17 | What does Gittins actually need to know? | `s13_rank_overlay` + ablation: R² 0.26 → 0.84 → 0.99; tail-only net = FCFS; hazard features supply 'an atom is coming'. Not for deployment (0.4 ms exact). | Whether an atom is coming. What does all this mean in practice? | 2.5 |
| 18 | What this means in practice | Five rules: histogram + Gittins from ~100 jobs; don't truncate at moderate load; refit ~500 jobs at idle moments; near-constant sizes → FCFS is fine; pair your simulations. | What's next? | 1.5 |
| 19 | What's next, and where does this apply? | Borg traces (BigQuery), multi-server, non-preemptive variant; pipelines, pick queues, ticket routing. | — | 1.5 |
| 20 | Close | Run Gittins on the histogram; refit from a rolling window. Citation, repo. | — | 0.5 |

Appendix (only if asked): A1 theorem (replace the bracketed line with the statement from the paper), A2 hull algorithm in three lines, A3 RL table, A4 the two drift experiments that misled first, A5 why analytical rather than learned.

Timing: 0–4 problem and method 8.5 · 5–8 question one 7.5 · 9–13 question two 10.5 · 14–17 question three 9 · 18–20 wrap 3.5 ≈ 39 min. Thirty-minute version: drop 17, fold 16 into one sentence on 14, fold 13 into a callout on 12. Confirm the talk/Q&A split with Felix or Luis.

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
