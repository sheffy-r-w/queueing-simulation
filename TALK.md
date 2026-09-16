# TALK — P&G data-science interview, Fri 2026-09-18, 12–1pm ET

Two parts: (1) what this audience is looking for and what I would change in the
15-section outline; (2) a slide-by-slide outline with timings, figures, and the
one sentence each slide has to land. Figures are `figures/sNN_*.png`; every number
is in `RESULTS.md`.

---

## Part 1 — What an ideal version looks like, and what to change

### What the brief actually asks for

Felix's list is a *project* arc, not a *paper* arc:

> problem → initial thinking and EDA → methods/practices/models tried → tools and
> libraries → tuning, corrections, validations → implementation → results

and the two things being scored are **technical mastery** and **clear communication**
to data scientists who are not queueing people (supply chain, retail, media).

The 15-section outline is paper-shaped: problem, method, then four sections on how
the proof works (6–9), then results. That ordering serves a theory audience. For this
one it has three problems:

1. **The proof sections (6–9) are the part they cannot evaluate**, and they sit in the
   middle where attention is lowest. Keep the *story* of that work (the obvious proof
   technique failed, a different invariant saved it), because "methods tried" is
   literally on the list, but compress it to two slides and move the theorem itself to
   an appendix slide for Q&A.
2. **The things they explicitly ask for are under-represented.** You have unusually
   good material for "tuning, corrections, validations" and "tools/implementation",
   and the outline gives it half of one section (§5). Specifically:
   - *Validation:* closed-form checks (Pollaczek–Khinchine, E[S]/(1−ρ)), an independent
     simulator with bit-identical output, a brute-force check of the rank definition,
     reproduction of the paper's figure.
   - *Corrections:* the unpaired-ratio artifact (medians below optimal at ρ = 0.98 until
     common random numbers), and the drift experiment whose target distribution made
     FCFS optimal so the baseline "improved". Both are honest, both show judgment, and
     interviewers remember them.
   - *Tuning:* the refit window as a bias–variance trade-off, the truncation level rule,
     the feature ablation for the learned rank function (R² 0.26 → 0.84 → 0.99).
   - *Tools:* numpy/numba/scipy/pandas/pytest/torch, the O(n) convex-hull rank
     algorithm (1.1 s → 0.4 ms), the skip-ahead event simulator.
3. **"Results" should be more than one figure.** They want to see what the answer
   means: rules of thumb someone in retail fulfilment could use on Monday.

### The changes, in order of importance

- **Reorder to the brief's arc.** Problem (5 min) → initial thinking + first experiment
  (5) → implementation and validation (5) → methods tried, including the failed proof
  route (5) → results and the truncation lesson (6) → drift, tuning, corrections (7) →
  learned rank function (3) → takeaways and next steps (3). About 38 minutes, leaving
  15–20 for questions in a 60-minute slot. Confirm with Felix how much is talk vs Q&A;
  the scheduler's guide may say.
- **One running example, used on every slide that introduces a concept.** Pick one the
  room lives in: an order-picking queue (retail/supply chain) or a support-ticket queue.
  "Job" = order, "size" = work to fulfil it, "response time" = order-to-done, "load" =
  how busy the picker is. Never say M/G/1, SOAP, or busy period without the translation
  the first time.
- **Lead each section with the question you asked, not the technique.** "Does the naive
  thing work?" → "Why does it work, when the usual argument says it shouldn't?" →
  "What happens when the world changes under you?" That is how a data scientist
  narrates a project, and it makes the theory feel like a step, not a detour.
- **Make the truncation result a "theory vs practice" slide.** The bound needs the tail
  truncated; in practice truncation costs 15–35% at ρ = 0.8 and only breaks even at
  ρ = 0.98. Saying "my own theorem's safety margin is a cost in practice, and here is
  when to ignore it" is exactly the mastery signal they want.
- **Frame k-updating as retraining cadence.** Every person in that room has chosen a
  retraining window for a production model. Window = 500 jobs ≈ 100 busy periods, too
  short is variance, too long is lag. Show the sweep, give the rule.
- **Cut, don't trim:** RL becomes one sentence on the imitation slide (it holds the
  imitation optimum and cannot discover the structure from scratch). The §8.5 mode-shift
  run is superseded by the sweep; leave it out. The Fig. 1.1 reproduction is appendix.
- **End with "where this applies at P&G"**, concretely and briefly: batch pipelines with
  historical run times, fulfilment queues with historical pick times, ticket routing.
  One slide, no over-claiming.

### What to have ready for Q&A (not on slides)

- Why one server? (Theory is single-server; the empirical-distribution idea and the
  refit engine carry over; multi-server is the next-steps slide.)
- Preemption: is it realistic? (Batch jobs, yes; a picker mid-order, less so. Gittins has
  non-preemptive variants; the *estimation* question is the same.)
- How does this compare to predicting each job's size with a model? (Predictions feed
  SRPT-with-predictions; that needs per-job features. This needs only history. They are
  complementary; Gittins can take predicted-size distributions.)
- Why 1-6-14 and bounded Pareto? (The paper's test bed: one multimodal, one heavy-tailed;
  the two regimes where FCFS fails differently.)
- What breaks it? (Sizes correlated with arrivals, load near 1 with few samples, drift
  faster than the window.)
- Borg traces: what exactly would you do? (Fit the empirical policy on one week, evaluate
  on the next in the simulator; the traces are on BigQuery.)

---

## Part 2 — Slide-by-slide outline (~38 min + Q&A)

Column "brief" says which item of Felix's list the slide serves. Times are targets.

| # | Slide | What is on it | The sentence to land | Brief | min |
|---|---|---|---|---|---|
| 0 | **Title.** Scheduling when all you have is history | Title, name, one plain-English line: "How to order a queue well when you don't know how long each job takes, but you have a log of past jobs." | This is a practical question with a clean answer. | — | 0.5 |
| 1 | **The problem, in a picture** | Running example: an order-picking (or ticket) queue. Orders arrive at random; each takes an unknown amount of work; you choose which to work on next; you can switch. Objective: mean time from arrival to done. Two toy orderings on 5 jobs showing order matters 2×. | Ordering is a free lever, and the right order depends on what you know about job sizes. | problem | 2 |
| 2 | **Vocabulary, once** | Job, size, age (work done so far), response time, load ρ (fraction of time busy), busy period (a burst of continuous work; the natural "episode"), preemption. Each with the running-example word next to it. | Five words; everything after uses them. | problem | 1.5 |
| 3 | **The spectrum of information** (`s02_spectrum.png`) | Exact sizes → SRPT (optimal). The full distribution → Gittins (optimal given only ages). Nothing → FCFS / round-robin. The middle column, "a sample of past sizes", is empty in the literature and is where every real system lives. | Real systems have samples, and nobody had said what to do with them. | problem / initial thinking | 2 |
| 4 | **What Gittins does** (`s04_gittins_intuition.png`) | Rank function on 1-6-14: rank(age) ≈ expected remaining work ÷ chance of finishing soon; serve the job with the lowest rank. Walk one job: at age 0 it's promising, at age 2 it's "probably a 6", at age 5.9 finish it. Point out the non-monotone shape. | Gittins is "expected cost per unit of progress", computed from the distribution, and it needs the *shape*, not just the mean. | initial thinking | 2.5 |
| 5 | **Initial thinking: do the obvious thing** | One line: replace the true distribution with the histogram of the last n jobs and run Gittins on it. Three questions written on the slide: Does it work? Why? What if the world changes? | The whole project is those three questions. | initial thinking | 1 |
| 6 | **EDA / first experiment: it works** (`s05b_baseline_teaser.png`) | 1-6-14, ρ = 0.8, 100 trials per box, paired to true Gittins. n = 10: 1.39×; n = 100: 1.04×; n = 1000: 1.005×. FCFS 1.08, PLCFS 1.31 as lines. | A hundred past jobs already gets within 5% of the optimum you'd get from the full distribution. | EDA / results | 2 |
| 7 | **Implementation I: the simulator, and how I trust it** | Event-driven preemptive simulator, decisions only at arrivals and "rank breakpoints" (skip-ahead), numba-compiled: 10k busy periods in under a second. Common random numbers: every policy sees the same arrivals. Validation table: FCFS = Pollaczek–Khinchine; PLCFS = E[S]/(1−ρ); Gittins on constant sizes = FCFS exactly; independent quantum-stepping simulator bit-identical; paper Fig. 7.2 reproduced (1.085 vs 1.074). 47 tests. | Before trusting any result I made the simulator reproduce four things I could compute by hand. | tools / validation / implementation | 2.5 |
| 8 | **Implementation II: computing the rank in O(n)** (`s05a_rank_hull.png`) | Rank at an atom = minimum slope from its point on the curve (F(x), E[min(S,x)]) to any later point → lower convex hull, one backward pass. Timing table: true Pareto 1.1 s → 0.4 ms (3,300×); bit-identical to the definition. Why it matters: refit every busy period, later. | The exact rank function is a convex-hull sweep, so refitting is essentially free. | tools / implementation | 2 |
| 9 | **Methods tried: the proof route that failed** (`s07_rank_functions.png`) | The standard way to prove "approximate policy ≈ optimal" is to show the rank functions are close. Overlay: empirical rank functions at n = 100 and 1000 do *not* converge to the true one (the histogram has atoms; the true distribution doesn't). Weeks on this. | The obvious argument is false, even though the experiment says the policy is fine. | methods tried | 2 |
| 10 | **What IS stable: tail ratios** (`s08_tail_ratio.png`) | Empirical vs true survival functions and their ratio: the ratio stays near 1 far into the tail. The bound, in words: the response-time gap shrinks like a power of 1/n, with a constant that grows as load → 1, provided you cut the tail at a level set by n and ρ. Theorem on an appendix slide. | You don't need the rank functions to agree; you need the tails to agree in ratio, and they do. | methods tried | 2 |
| 11 | **Theory vs practice: the truncation correction** (`s11_baseline.png`, one panel, or a 2-row table) | The proof needs truncation. Paired results: at ρ = 0.8, n = 1000, untruncated 1.005 vs truncated 1.17; at ρ = 0.98 the gap closes (1.11 vs 1.09, CIs overlap). Truncation is insurance against the tail you haven't seen; at moderate load it is over-priced. | The safety margin the theorem needs is a real cost; know when to ignore your own theorem. | corrections / results | 2 |
| 12 | **Results in full** (`s11_baseline.png`) | Four panels. Rules of thumb on the slide: ~100 samples → within ~5%; ~1000 → within 1–2% at ρ = 0.8; ρ = 0.98 needs ~10× more data; on heavy tails FCFS is 5–10× worse than any of these. | Even tiny samples beat the default by a lot; the return on more data flattens fast, except near saturation. | results | 3 |
| 13 | **A correction I had to make: pairing** | Before/after: unpaired ratios had medians *below* the optimum at ρ = 0.98. A single 4,000-busy-period run of the optimal policy itself lands at 0.87–0.94 of its own long-run mean (right-skewed, SD 0.25–0.30). Fix: simulate the optimum on the same arrival stream as each trial. Every median ≥ 1, CIs 3–10× tighter. | Heavy tails make short-run averages lie; pair your comparisons or you will publish noise. | corrections / validation | 1.5 |
| 14 | **The world changes: refit as you go** (`s13_drift_timecourse.png`) | Job-size mix drifts over 4,000 busy periods (mostly long → mostly short). "Genie" = optimal for the current distribution. Static fit (made at the start) tracks FCFS and both climb to 1.41×; refitting from the last 500 jobs at every busy period stays within 1–3%. Right panel: the same policies with no drift, for calibration. | A policy fitted once behaves like the default once the workload moves; refitting from a rolling window keeps you at the optimum. | methods / results | 2.5 |
| 15 | **Tuning the window** (`s13_drift_sweep.png`) | Window w ∈ {50, 200, 500, 2000} × drift period T. w = 50 is variance-limited everywhere (~1.12); at T = 500 the best is w = 500 (w = 2000 is stale, 1.038); at T = 100 nothing tracks and the longest window wins by averaging. Rule: w ≈ 500 jobs ≈ 100 busy periods. | It's retraining cadence: too short is variance, too long is lag, and you can measure both. | tuning | 2 |
| 16 | **What drifts matters** (`s13_drift_types.png`) | Three drifts side by side: mix shift (static ≈ FCFS, 1.11), shape drift with load pinned (static 1.04), heavier tail (FCFS 3.6× but static only 1.04). Refit-500 within 1–2.5% in all three. Tie back to slide 10: the tail index barely moves the rank function. | Not every drift needs a refit; the ones that change the *shape* do. | results / corrections | 2 |
| 17 | **Can a model learn the rank function?** (`s13_rank_overlay.png` + ablation table) | MLP from features of the recent sample (survival mass, conditional quantiles, mean excess, "distance to the next atom") to log rank; trained on 4,000 synthetic distributions, evaluated on held-out families. Ablation: R² 0.26 → 0.84 → 0.99 as hazard features are added; the tail-only net collapses to FCFS. As a refit policy: 1.017 vs 1.011 for the exact computation. One line: policy-gradient RL from scratch never beat FCFS; fine-tuning the imitation net just holds it. | The learned version works, and the ablation tells you *which* information Gittins needs: "is an atom coming?" | methods / tuning | 2.5 |
| 18 | **Practical takeaways** | Five bullets: (1) if you have a job log, run Gittins on the histogram; 100 jobs is enough to start. (2) Don't truncate at moderate load. (3) Refit from a rolling ~500-job window at natural boundaries (idle moments), never mid-burst. (4) If sizes are nearly constant, FCFS is already optimal, don't bother. (5) Pair your simulations. | Here is what to do on Monday. | results | 1.5 |
| 19 | **Next steps and where this applies** | Google Borg traces (BigQuery): fit on week 1, evaluate on week 2. Multi-server. Non-preemptive variant for physical queues. Where it fits: batch/data pipelines with run-time history, fulfilment and pick queues, ticket routing with historical handle times. | Same recipe, real logs; the open question is multi-server. | — | 1.5 |
| 20 | **Thank you / reference** | Paper citation (POMACS 2026), repo, one-line summary repeated from slide 0. | — | — | 0.5 |

Appendix slides (for Q&A only): the theorem statement; the drift-direction lesson (the
earlier ramp made FCFS optimal at the end; keep as an example of a misleading experiment);
the RL table; simulator event logic; the hull algorithm in three lines; `drift_window.png`.

### Timing check

Slides 0–5 (problem, 9.5 min) · 6–8 (first result, implementation, 6.5) · 9–11 (methods
tried and the truncation lesson, 6) · 12–13 (results and pairing, 4.5) · 14–16 (drift and
tuning, 6.5) · 17 (learned rank, 2.5) · 18–20 (wrap, 3.5). Total ≈ 39 min. If you need to
cut five minutes: drop slide 16 to one sentence on slide 14, and fold slide 13 into slide 12
as a "note on method" callout.

### Slide-level conventions

- Every figure slide: title is the finding, not the topic ("100 past jobs gets within 5%",
  not "Baseline results").
- Ratios everywhere, 1.0 = optimal, same colour for the same policy on every slide:
  black solid = true Gittins rank, black dashed = optimal / genie, blue = empirical
  Gittins (exact, incl. k-updating; lighter/darker blue for n or w), orange = a degraded
  variant (truncated, or a static fit), grey = FCFS (dotted) and PLCFS (dash-dot),
  violet = learned (NN) rank, aqua = RL from scratch. Every figure with ≥ 2 series has
  a legend naming every line style. Set in `egittins/plotting.py`; all figures follow it.
- One look: Avenir Next, a bold left-aligned headline that states the finding, a muted
  subtitle with the setup, hairline grid, thin marks, no chart junk.
- Numbers on slides come from `RESULTS.md`; each has a CI there if asked.
