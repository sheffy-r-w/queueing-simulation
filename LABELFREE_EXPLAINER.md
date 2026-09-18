# Label-free scheduling: what does a scheduler need to know?

> **A note from me (Shefali).** This explainer was prepared in part with AI assistance (Claude Code). The
> experiment it describes is `experiments/labelfree.py` (code in `egittins/labelfree.py`); its
> numbers are read from `results/s13_labelfree_static_summary.csv` and
> `results/s13_labelfree_stage0_summary.csv`, and are also in RESULTS.md with commands and CIs.

## The question, and why the earlier experiments could not answer it

The imitation nets (`egittins/imitation.py`) reproduce the Gittins rank function from sample features, and the ablation says which features they need to *copy* the rank function. But they were trained on exact Gittins ranks as targets, so as evidence about what a scheduler must know they are circular: since the the answer was literally in the labels. The policy-gradient arm (`egittins/rl.py`) tried to learn from response time alone and never beat FCFS, because the gradient estimate from busy-period episodes is too noisy to see the rank structure and exploit it.

This experiment keeps the label-free goal and replaces the noisy gradient with a derivative-free search on a low-noise objective. **Gittins appears nowhere in training.** It is used only to score the result afterwards.

## The method

**CMA-ES:** CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Keeps a Gaussian "cloud" of candidate parameter vectors, simulates each candidate, and moves and reshapes the cloud toward the candidates that scored best. It needs only scores, not gradients, so it does not care that the simulator is discrete, discontinuous and stochastic. (can we get a yay for derivative-free search)

**Why common random numbers (paired streams):** Two candidates simulated on different random arrival streams differ mostly because of the streams rather than the policies; at 2,000 busy periods that noise (a few percent) is larger than the improvements being searched for (a few tenths of a percent). Simulating every candidate in a generation on the *same* seeds makes the comparison paired, meaning the stream noise cancels and the ranking within a generation is nearly exact. Fresh seeds every generation stop the search from overfitting one stream. This is the same discipline every comparison in RESULTS.md uses; here it is what makes the search work at all where policy gradient failed.

**We try to learn ordering of ranks of ages rather than exact values** An age-based scheduler always serves the job with the lowest rank. Any strictly increasing transform of the rank function gives the identical schedule, so the *values* of a learned rank are meaningless and only the *ordering over ages* can be compared with Gittins. All comparisons therefore use Spearman correlation of rank over ages, and the overlay figures quantile-match the learned curve onto the Gittins values (same ordering as the learned rank, values borrowed from Gittins), so the two curves coincide exactly when the orderings agree.

**Stage 0, the positive control.** One distribution (1-6-14, ρ = 0.8), the rank a free table (log-rank at 32 knots over age, linearly interpolated, the same class the RL scratch arm used). If the search cannot find Gittins here, it cannot be used for anything else.

**Stage 1, the real experiment.** The rank is a small function of *features computed from a sample of 500 past job sizes* at each age, the same three nested feature tiers the imitation nets used:

| tier | what the scheduler can see at age a | in words |
|---|---|---|
| tail only | log survivor mass Ḡ(a), log (atom) count above a, log mean excess | "how many jobs get past this age, and how much work is left on average" |
| + quantiles | + quantiles of the remaining work (S − a given S > a) ÷ mean excess, and the max | "the *shape* of the remaining-work distribution" |
| + hazard (full) | + gap to the next observed size above a ÷ mean excess, mass at that size | "is a high likelihood of completion coming right now?" |

The function is trained across 14 random distributions drawn from the imitation generators (mixtures, bounded Pareto, lognormal, Weibull, few-atom discrete), with 1-6-14 and Pareto(2, 1.2, 500) held out entirely. Training across distributions is what I hope makes the ablation meaningful: on one distribution every feature is just a re-encoding of age. Loss = mean over training distributions of (mean response time ÷ FCFS mean response time on the same seed). Evaluation on the two held-out distributions uses exactly the protocol and trial seeds of the imitation table, so all rows are paired.

## What the result shows and what it does not

**Stage 0 passed.** On 1-6-14 the search reaches 1.005 ± 0.0005 of true Gittins on all three optimizer seeds (FCFS 1.089; a random rank table of the same class 1.067), and every seed recovers both dips of the Gittins rank. The RL failure was the optimizer, not the policy class. Cost: 19 M busy periods per seed, 20 s each.

**Stage 1, the actual finding.** Given only tail mass, count and mean excess, every run (3 seeds × linear and 8-unit MLP) converges to the *same* ordering: **serve the job with the least expected remaining work**, E[S − a | S > a], estimated from the sample. On the held-out distributions that is 1.015 ± 0.003 (1-6-14) and 1.033 ± 0.004 (Pareto), against exact empirical Gittins 1.007 and 1.023 on the same windows and seeds. That rule was never written down anywhere in the code; the search found it.

**What the richer features did.** With quantiles or hazard features added, the search fits the training set slightly better but on held-out 1-6-14 lands anywhere between 1.06 and 1.11 depending on the seed, with wide confidence intervals, while doing slightly better on the Pareto (1.013–1.016). Fourteen training distributions, most of them heavy-tailed, are not enough for a label-free search to learn to *use* the atom structure that the imitation net was handed in its labels.

**We do not show.** Nothing at ρ = 0.98, under drift, or for other window sizes; nothing about whether more training distributions or a differently weighted loss would let the search exploit the richer features; and the Spearman numbers are a weak proxy (a 0.9 ordering can lose to a 0.8 ordering in response time, because only the ages where jobs actually compete matter). Stage 2 (size-aware control against SRPT) was not run: the simulator ranks by age only.
