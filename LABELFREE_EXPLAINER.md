# Label-free scheduling: what does a scheduler need to know?

*One page for defending the `s13_labelfree_*` results to ML practitioners. Numbers, CIs and commands are in RESULTS.md ("§13 Label-free rank search").*

## The question, and why the earlier experiments could not answer it

The imitation nets (`egittins/imitation.py`) reproduce the Gittins rank function from sample features, and the ablation says which features they need to *copy* it. But they were trained on exact Gittins ranks as targets, so as evidence about what a scheduler must know they are circular: the answer was in the labels. The policy-gradient arm (`egittins/rl.py`) tried to learn from response time alone and never beat FCFS, because the gradient estimate from busy-period episodes is too noisy to see the rank structure.

This experiment keeps the label-free goal and replaces the noisy gradient with a derivative-free search on a low-noise objective. **Gittins appears nowhere in training.** It is used only to score the result afterwards.

## The method in plain language

**CMA-ES in two sentences.** CMA-ES (Covariance Matrix Adaptation Evolution Strategy) keeps a Gaussian "cloud" of candidate parameter vectors, simulates each candidate, and moves and reshapes the cloud toward the candidates that scored best. It needs only scores, not gradients, so it does not care that the simulator is discrete, discontinuous and stochastic.

**Why common random numbers matter.** Two candidates simulated on different random arrival streams differ mostly because of the streams, not the policies; at 2,000 busy periods that noise (a few percent) is larger than the improvements being searched for (a few tenths of a percent). Simulating every candidate in a generation on the *same* seeds makes the comparison paired: the stream noise cancels and the ranking within a generation is nearly exact. Fresh seeds every generation stop the search from overfitting one stream. This is the same discipline every comparison in the deck already uses; here it is what makes the search work at all where policy gradient failed.

**Why ordering, not value.** A SOAP scheduler always serves the job with the lowest rank. Any strictly increasing transform of the rank function gives the identical schedule, so the *values* of a learned rank are meaningless and only the *ordering over ages* can be compared with Gittins. All comparisons therefore use Spearman correlation of rank over ages, and the overlay figures quantile-match the learned curve onto the Gittins values (same ordering as the learned rank, values borrowed from Gittins), so the two curves coincide exactly when the orderings agree.

**Stage 0, the positive control.** One distribution (1-6-14, ρ = 0.8), the rank a free table (log-rank at 32 knots over age, linearly interpolated, the same class the RL scratch arm used). If the search cannot find Gittins here, it cannot be used for anything else.

**Stage 1, the real experiment.** The rank is a small function of *features computed from a sample of 500 past job sizes* at each age, the same three nested feature tiers the imitation nets used:

| tier | what the scheduler can see at age a | in words |
|---|---|---|
| tail only | log survivor mass Ḡ(a), log count above a, log mean excess | "how many jobs get past this age, and how much work is left on average" |
| + quantiles | + quantiles of the remaining work (S − a given S > a) ÷ mean excess, and the max | "the *shape* of the remaining-work distribution" |
| + hazard (full) | + gap to the next observed size above a ÷ mean excess, mass at that size | "is a cliff of completions coming right now?" |

The function is trained across 14 random distributions drawn from the imitation generators (mixtures, bounded Pareto, lognormal, Weibull, few-atom discrete), with 1-6-14 and Pareto(2, 1.2, 500) held out entirely. Training across distributions is what makes the ablation meaningful: on one distribution every feature is just a re-encoding of age. Loss = mean over training distributions of (mean response time ÷ FCFS mean response time on the same seed). Evaluation on the two held-out distributions uses exactly the protocol and trial seeds of the imitation table, so all rows are paired.

## What the result shows and what it does not

**Stage 0 passed.** On 1-6-14 the search reaches 1.005 ± 0.0005 of true Gittins on all three optimizer seeds (FCFS 1.089; a random rank table of the same class 1.067), and every seed recovers both dips of the Gittins rank. The RL failure was the optimizer, not the policy class. Cost: 19 M busy periods per seed, 20 s each.

**Stage 1, the actual finding.** Given only tail mass, count and mean excess, every run (3 seeds × linear and 8-unit MLP) converges to the *same* ordering: **serve the job with the least expected remaining work**, E[S − a | S > a], estimated from the sample. On the held-out distributions that is 1.015 ± 0.003 (1-6-14) and 1.033 ± 0.004 (Pareto), against exact empirical Gittins 1.007 and 1.023 on the same windows and seeds. That rule was never written down anywhere in the code; the search found it.

**What the richer features did.** With quantiles or hazard features added, the search fits the training set slightly better but on held-out 1-6-14 lands anywhere between 1.06 and 1.11 depending on the seed, with wide confidence intervals, while doing slightly better on the Pareto (1.013–1.016). Fourteen training distributions, most of them heavy-tailed, are not enough for a label-free search to learn to *use* the atom structure that the imitation net was handed in its labels.

**A correction to the imitation slide.** The imitation "tail + mean excess only" net sits at 1.095, FCFS level, and RESULTS.md reads that as "those features cannot see that an atom is coming". Ranking by mean excess with the same three features gives 1.015. The features see it; that regression did not. The honest ablation statement is now: mean excess alone gets within 1.5% (1-6-14) and 3% (Pareto) of optimal; the hazard and quantile features, *given labels*, buy the remaining 0.5–1%.

**What it does not show.** Nothing at ρ = 0.98, under drift, or for other window sizes; nothing about whether more training distributions or a differently weighted loss would let the search exploit the richer features; and the Spearman numbers are a weak proxy (a 0.9 ordering can lose to a 0.8 ordering in response time, because only the ages where jobs actually compete matter). Stage 2 (size-aware control against SRPT) was not run: the simulator ranks by age only.

## Eight likely questions

1. **Isn't CMA-ES just random search with extra steps?** It is a randomized search, but it adapts the step size and the covariance of its proposals to the fitness landscape, so in 32–104 dimensions it converges in ~100 generations where naive random search would not. Here it is a tool, not the point.
2. **Why not gradients / RL?** We tried REINFORCE on the same policy class: its gradient estimate from busy-period episodes is dominated by stream noise and never beat FCFS. Pairing candidates on common random numbers removes that noise for a derivative-free method; there is no equivalent trick for a likelihood-ratio gradient.
3. **Common random numbers: doesn't scoring everyone on the same seeds overfit those seeds?** Within a generation yes, deliberately, so the comparison is paired. The seeds change every generation, so the search cannot exploit one stream; the final policy is evaluated on fresh seeds it never saw.
4. **You say ordering, not value. Then how do you compare with Gittins?** Spearman correlation of the rank over ages, and overlay plots after quantile-matching the learned curve onto the Gittins values, so identical orderings give identical curves. Response time is the real metric; Spearman is diagnostic only.
5. **How do you know Gittins never leaked into training?** The training loss is MRT divided by FCFS's MRT on the same seed; the code path from features to fitness never imports the Gittins kernel. Gittins is computed only in the evaluation script, on distributions the training never saw.
6. **Isn't "least expected remaining work" a known heuristic?** Yes: it is the b = ∞ term of the Gittins index and an upper bound on it; in the M/G/1 literature it is the natural first approximation. The point is that a label-free search rediscovers it from response time alone, and that on these distributions it is within 1.5–3% of optimal.
7. **Why did the richer features get worse on 1-6-14?** Fourteen training distributions, most of them heavy-tailed, reward "younger first"; with more parameters the search finds a solution that fits those slightly better and transfers worse. It is a sample-size and loss-weighting limitation of the experiment, not evidence that the features are harmful; with labels, the same features give 1.011.
8. **So why compute Gittins at all if mean excess is within 1.5%?** Because 1.5% at ρ = 0.8 is 3% on the Pareto, is untested at high load and under drift, and because empirical Gittins costs nothing extra: it is a linear-time computation on the same sample. Mean excess is the floor a practitioner gets for free; Gittins is the optimum.

## GO / NO-GO for one slide

**GO, for one slide, with a narrow claim.** The positive control is unambiguous (three seeds, 1.005, the Gittins shape recovered where policy gradient failed), and the Stage 1 result is a clean, quotable sentence: *searched only on response time, the scheduler rediscovers "least expected remaining work first" and lands within 1.5% of optimal on 1-6-14 and 3% on the Pareto.* Both numbers are paired, replicated across seeds, and the mean-excess interpretation is verified directly (Spearman 0.999).

Reasons for GO: it is the only label-free evidence in the deck; it answers the audience's natural question ("what does the scheduler need to know?") with a rule they can act on; it costs 9 minutes of compute to reproduce; every number has a CI.

Reasons for caution, to state on the slide or in the notes: the richer feature tiers did not improve under label-free search at this budget (say so; do not show them as a ladder); and the result revises the imitation ablation's reading of the tail-only net, so that slide's caption should change from "the features cannot see it" to "the regression did not use it". Do not claim anything about ρ = 0.98 or drift.
