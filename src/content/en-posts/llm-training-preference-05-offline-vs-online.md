---
sourceId: 2026-10-20-llm-training-preference-05-offline-vs-online
slug: llm-training-preference-05-offline-vs-online
title: "Offline Preference Optimization versus Online Sampling: The Fixed Answer Pool and Regeneration"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-20T00:00:00.000Z
description: "How a fixed preference pool and on-policy sampling differ in reachable updates, distribution drift, sampling budget, and verification cost."
tags:
  - Online Preference Optimization
  - Off-Policy Sampling
  - Distribution Shift
  - Sampling Budget
category: LLM Training Research
series: LLM Training Research Notes · Preference Volume
seriesOrder: 5
lang: en
---

> **Scope note.** This is an English overview of a Chinese article comparing offline preference optimization with online sampling. The two-token probabilities, the loss and pull-strength values, the sampling counts, and the cost table are hand-computed teaching calculations and call counts, not measurements; no preference optimization was run, no GPU-hours or wall-clock times were measured, and no speedup is imported from another paper's report. The article analyzes a mechanism and proposes a control design; it does not establish that either procedure is better.

## Two procedures, and what each one changes

The previous note's objective works on a fixed set of preference pairs. The natural follow-up is: since the policy keeps changing during training, why not let it generate new answers and let a judge compare them?

That is online preference optimization. Both procedures work and both produce results, but they are not two versions of one method. They reach different sets of updates, they cost different amounts, and they fail differently.

Offline preference optimization freezes the answer pool into a dataset $\mathcal{D}=\{(x,y_w,y_l)\}$ and reads only that dataset during training. This is the construction the direct preference objective is built around, and its data is collected once, before training begins ([DPO, Section 4](https://arxiv.org/html/2305.18290v3#S4)). An online method samples new answers from the current policy at every step and then has a judge model or an executable checker produce a comparison or a score.

TRL 0.23.1's Online DPO Trainer takes the second route: each training iteration samples two responses from the current model, hands them to a judge model that selects the more preferred one, and then applies a preference optimization update ([TRL 0.23.1 Online DPO Trainer](https://huggingface.co/docs/trl/v0.23.1/en/online_dpo_trainer)).

The work on direct alignment from online AI feedback states the difference plainly: the preference datasets commonly used by direct alignment methods are collected before training and never updated afterwards, so the feedback is entirely offline; and because those responses often come from a different model than the one being aligned, while models keep evolving during training, the alignment stage is inevitably off-policy ([Direct Language Model Alignment from Online AI Feedback, arXiv:2402.04792v2](https://arxiv.org/abs/2402.04792)).

That observation holds for the offline method as well, just with the direction reversed: the offline method has no "on-policy" property at all. Its data comes from some historical policy, and the gap between that policy and the current one only widens as training proceeds.

## The same preferences, two sets of reachable updates

Start with the part that can be computed. The vocabulary has two tokens, replies have two tokens, and there are four candidates: "AA", "AB", "BA", "BB". The current policy and the reference give:

| Reply | $\pi_\theta$ | $\pi_{\mathrm{ref}}$ |
| --- | ---: | ---: |
| AA | 0.05 | 0.25 |
| AB | 0.15 | 0.25 |
| BA | 0.30 | 0.25 |
| BB | 0.50 | 0.25 |

The policy puts only 0.05 on "AA", and 0.80 on "BA" and "BB" together.

The offline dataset holds two pairs, and "AA" wins both:

| Pair | chosen | rejected | $h$ | loss ($\beta=0.1$) | pull strength |
| --- | --- | --- | ---: | ---: | ---: |
| A | AA | BA | $-1.7918$ | 0.7867 | 0.0545 |
| B | AA | BB | $-2.3026$ | 0.8149 | 0.0557 |

Here $h=\Delta(y_w)-\Delta(y_l)$ and the pull strength is $-\partial\ell/\partial h=\beta\big(1-\sigma(\beta h)\big)$, with the log ratio taken against the reference:

$$
\Delta(y)=\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

which for the first pair gives $\Delta(\text{AA})=\log(0.05/0.25)\approx-1.60944$ and $\Delta(\text{BA})=\log(0.30/0.25)\approx0.18232$, so $h\approx-1.79176$, matching the $-1.7918$ in the table.

The first-order effect of one offline step, in log-probability coordinates, is:

$$
\Delta\log\pi_\theta:\quad \text{AA } +0.1102,\quad \text{AB } 0,\quad \text{BA } -0.0545,\quad \text{BB } -0.0557
$$

The three non-zero entries sum to zero, as renormalization requires.

The important entry is "AB": it appears in no pair, so its log-probability is unchanged to first order. What an offline update can express is entirely "the comparisons already recorded in the pool"; relations outside the pool are unconstrained. This is not an implementation defect, it is the definition of a fixed answer pool.

A direct consequence follows. The offline method gives a full-strength signal to "AA" even though its current probability is only 0.05. The pull strength 0.0545 has nothing to do with $\pi_\theta(\text{AA})=0.05$, because it comes from that one recorded comparison rather than from a sampling frequency.

Now switch to online. Sampling $N=200$ responses from the current policy gives these expected counts:

| Reply | Expected count | Share |
| --- | ---: | ---: |
| AA | 10 | 5% |
| AB | 30 | 15% |
| BA | 60 | 30% |
| BB | 100 | 50% |

Two differences appear. First, "AA" occurs only 10 times in 200 samples. The update strength of an online method is proportional to sampling frequency, so the signal about it is diluted by a factor of 20; reaching a comparable effective sample size in the same number of steps requires more sampling.

Second, "AB" occurs 30 times. The online loop can hand it to the judge, and if the judge finds it better, a new comparison exists that is not in the offline data at all. Getting it costs two steps that both have to be paid for: generate, then judge.

The same fact in policy-gradient terms. If the reward is given only to "AA" and zero elsewhere, the mean baseline is 0.05 and the advantages are $+0.95$ and $-0.05$, and weighting by the sampled counts gives:

$$
\text{positive pull}=10\times0.95=9.5,\qquad \text{negative push}=190\times0.05=9.5
$$

The two are equal. The direction is right, but the positive half rests on 10 samples, so the variance of that estimate falls only as the sample count grows.

## Distribution drift and the judge inside the loop

The update rule of an online method contains the policy ratio

$$
\rho_\theta(y)=\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{old}}(y\mid x)}
$$

where the numerator is the policy being optimized and the denominator is the sampling policy that produced the batch. When the two disagree, $\rho_\theta(y)\neq1$, and importance weighting and clipping become necessary. PPO clips precisely in order to limit the size of a single step under this mismatch ([Proximal Policy Optimization Algorithms, arXiv:1707.06347v2](https://arxiv.org/abs/1707.06347)).

The converse is the reason offline data cannot simply be used as online data. Running an "online" update on old data generated by a fixed policy pushes the ratio far from 1, and the gradient is amplified past the point of usefulness.

Along this chain, the judge model is itself part of the policy. The judge's preference distribution shapes the reward, and it has its own tendencies toward verbosity, formatting, and position. MT-Bench and Chatbot Arena systematically record position bias, verbosity bias, and self-preference bias ([MT-Bench and Chatbot Arena, arXiv:2306.05685](https://arxiv.org/abs/2306.05685)). Online puts the judge inside the training loop, so the judge's bias keeps participating in optimization instead of affecting a one-off label.

## Cost, budget, and why online is not automatically better

The core advantage of the online route is finding things outside the pool. "AB" is the example: it was never compared inside the pool, so an offline loop never learns it. The corresponding cost of a fixed pool is symmetrical: if the best answer happens not to be in the pool, the offline method will never move toward it however long it trains. Repeated training on a small pool can also concentrate probability on the few chosen answers and produce diversity collapse, where every comparison in the pool is satisfied while behavior outside the pool degrades quickly. Which procedure to use therefore depends on the shape of the problem: if the answer space is essentially enumerated and the criterion is stable, offline is cheaper; online suits an open answer space, or a case where the model has to find better solutions itself.

Cost has to be compared on one scale. Taking 100 prompts per step and 2 responses per prompt online:

| Item | Offline DPO (100 prompts, 200 pairs) | Online DPO (100 prompts, 2 per prompt) |
| --- | ---: | ---: |
| Generation | 0 | 200 per step |
| Judge or checker calls | 0 | 200 per step |
| Policy model forward | 400 | 200 |
| Policy model backward | 200 | 200 |
| Reference model forward | 400, or 0 after precomputation | 200, or 0 after precomputation |

The offline cost is paid once, during data collection: 200 generations plus 200 annotations. Online converts that into a per-step cost. Over 50 training steps, online adds $200\times50=10000$ generations and the same number of checker calls, while offline data collection stays at 200. Measured in tokens the same point holds: offline consumes only the answers collected up front, while online consumes newly generated tokens at every step. These are counts, not measurements. GPU-hours and wall-clock time depend on model size, sampling length, hardware, and concurrency, and they have to be measured locally before they can enter a report. The source did no such timing and does not substitute another study's reported speedup.

Three situations make the online route the worse one. Reward hacking: as optimization pressure rises, the policy actively searches for answers that score well without being good; research on reward over-optimization discusses a proxy reward that keeps rising while another evaluation signal degrades, to an extent that depends on each study's setup ([Scaling Laws for Reward Model Overoptimization, arXiv:2210.10760](https://arxiv.org/abs/2210.10760)), and in preference learning this exploitation has a specific name, length exploitation ([Disentangling Length from Quality in DPO, arXiv:2403.19159v2](https://arxiv.org/abs/2403.19159)). Insufficient sampling budget: in the example above the good answer occupies 5% of the distribution, and the advantage of the online route depends precisely on sampling it, so a small budget only repeats the same comparison many times. Verification that costs money and carries bias: every answer passes through a checker or a judge; where the task has an executable checker the cost is bounded and the judgment is stable, but a model judge adds its own preferences on top of the cost, and online lets those preferences enter every update directly.

The comparison rests on assumptions: that the comparisons produced by the judge or checker are stable, that the sampling distribution can be approximated, and that a first-order analysis of probabilities represents the direction of one update. None of these approximations is exact in real training. A first-order conclusion states a direction, not a magnitude, and does not guarantee escaping a local minimum.

The control design that follows holds variables fixed rather than comparing two casual training runs. Use the same prompts, the same evaluation set, and the same total number of preference events consumed per step: the offline group draws the same number of pairs from the fixed pool each step, and the online group samples the same number of responses each step. Fix the evaluation protocol, then compare two things: the final win-rate on the preference test set, and the number of distinct response types that appeared during sampling. If the offline group has the higher win-rate while the response-type count does not grow, the difference came from the sampling budget. If the online group produces new response types and the evaluation improves alongside them, exploration genuinely brought gains from outside the pool. If the online group has the higher win-rate while the response-type count does not grow, amplified judge bias is the first thing to suspect.

Online preference optimization has now used both ends of the loop — sampling from the policy and turning a comparison into a signal — but the middle is missing, which is the definition of the reward. GRPO-style methods sample a group of answers for one prompt and replace absolute scores with within-group relative ranking to reduce the difficulty of value estimation ([DeepSeekMath, arXiv:2402.03300v3](https://arxiv.org/abs/2402.03300)). The next volume is about what that feedback actually checks.

## Where the Chinese original lives

The Chinese original is [What Distinguishes Offline Preference Optimization from Online Sampling: A Fixed Answer Pool and Regeneration](/posts/2026-10-20-llm-training-preference-05-offline-vs-online/). It holds the complete four-candidate probability table, the pair table with its loss and pull-strength columns, the first-order log-probability effects, the expected sampling counts for 200 draws, the policy-gradient baseline calculation, the full per-step cost table, the three ways online becomes worse, the assumption list, the control design with its three readings, and the reference list to DPO, OAIF, TRL 0.23.1, PPO, DeepSeekMath, and the over-optimization study. Its main limit is unchanged: the cost table is a call-count accounting, not a measurement, and no preference optimization was run. The next note in this volume returns to the evaluation side and asks why a higher preference score does not necessarily mean a better task result.
