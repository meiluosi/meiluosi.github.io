---
sourceId: 2026-10-21-fly-connectome-05-meta-learning
slug: fly-connectome-05-meta-learning
title: "Evolving the Learning Rule: A Search That Missed Forgetting"
editionLabel: ENGLISH OVERVIEW · RESEARCH NOTE
published: 2026-10-21T00:00:00.000Z
description: "Evolution over five learning-rule parameters stalled on the forgetting rate; fixing one starting point moved the gain from +4 to +7 to +14 to +21."
tags:
  - Meta-Learning
  - Evolution Strategies
  - Forgetting
  - Null Result
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 5
lang: en
---

> **Scope note.** This is an English overview of the fifth Chinese article of the Fly Connectome Research series. It reports an automatic search over five parameters of the learning rule, on the foraging task, with three wirings and three evolution seeds per wiring, 9 evolutions in total. The article's main result is about the search, not about the fly: the first evolution could not find a forgetting rate that a manual sweep later showed to be helpful, and the cause was the parameterization of the search space. It does not show that recursive self-improvement is near, and it does not show that the real wiring is better than the ablations; the wiring comparison was carried along and its conclusion did not change. The gains reported here are on one task, with intervals bootstrapped over worlds and not over the evolution itself.

## What is evolved, and how

The series began from recursive self-improvement: a system that improves its own way of learning, and becomes better at improving itself as a result. On a learner with real structure, such as the fruit fly mushroom body, the circuit is legible, the learning rule can be written out, and the wiring is real. What this article does is narrow: automatic tuning of five parameters of the learning rule. The motion constants, such as turn amplitude and innate attraction, are left fixed, and the starting point is the hand-tuned default.

| Parameter | Meaning |
| --- | --- |
| Learning rate for positive prediction error | How much a synapse changes when outcome beats expectation |
| Learning rate for negative prediction error | How much a synapse changes when outcome is worse than expectation, kept separate from the previous one |
| Forgetting rate | How fast synaptic strength decays back toward its initial value each step |
| KC sparsity | The fraction of the strongest KCs that takes part in encoding |
| Turning weight of value | How strongly the learned value influences turning |

The outer loop is an OpenAI-style evolution strategy. Noise is added to the center of the five parameters to produce 16 candidates, 8 antithetic pairs; each candidate forages in 12 training worlds and is ranked by mean net reward; the center moves along the direction favored by the ranking, for 25 generations. The training worlds take their reward-inversion periods from 1000, 2000, and 3000, half consistent and half conflicting, and do not overlap the evaluation worlds. Three wirings are evolved — real, shuffled, and random sparse — with 3 independent evolution seeds each, giving 9 evolutions.

## The evaluation protocol and the registered criterion

Evaluation uses the center of the final generation. No generation is cherry-picked and no seed is dropped. The evaluation uses 100 worlds that were never used before, under three conditions:

- **in-dist**, where the reward-inversion period has the same distribution as in training;
- **fast**, where the inversion period is 500 and the world is more volatile, out of distribution;
- **stationary**, where rewards never invert, also out of distribution.

The registered criterion requires the paired difference between evolved and hand-tuned default to have a confidence-interval lower bound above 0 and a gain of at least 6, and to hold in at least 2 of the 3 evolution seeds.

## The first evolution: small and tidy gains

Mean net reward on in-distribution worlds:

| Wiring | Hand-tuned default | After evolution | Gain |
| --- | --- | --- | --- |
| Real | +29.6 | +33.9 | +4.3 |
| Shuffled | +27.1 | +32.1 | +5.0 |
| Random sparse | +33.8 | +40.7 | +6.9 |

All directions are positive and most intervals exclude 0, but the magnitude mostly falls short of +6. By the criterion, only random sparse qualifies on in-distribution worlds (2 of 3 seeds), real gets 1, and shuffled gets 0. The reading at the time was that the hand-tuned defaults were already near optimal. Section four of the source overturns that reading.

The wiring comparison gave the same picture as the earlier foraging article. Real minus shuffled was +3.5, +0.5, and +1.5, which is equivalent or uncertain. Real minus random sparse was -2.7, -8.7, and -8.8, with random sparse clearly better in 2 of 3 runs. The real wiring showed no advantage.

What evolution selected was also informative. KC sparsity fell from 5% to about 2% in all 9 runs. The negative-error learning rate came out above the positive one in 7 of 9, which later turned out to give no benefit when isolated and is probably a by-product of the other parameters. The turning weight of value dropped slightly.

Then came the anomaly. The pre-written guess was that the more volatile the world, the more forgetting should help, because rewards invert. In all 9 evolutions the forgetting rate stayed pinned at the search lower bound, 1e-5. The out-of-distribution results were anomalous too: the gain was smallest in the more volatile fast worlds, from -0.5 to +2, and largest in the non-inverting worlds, from +4 to +9. The first draft explained this by saying that when rewards invert, relearning fast matters more than forgetting, and that a negative error can suppress the old value anyway. That explanation was marked as untested at the time.

## The manual sweep, and the corrected search

The forgetting rate was then swept by hand, under a separate pre-registration: no evolution, just a direct sweep of the forgetting rate together with the learning speed and the positive-versus-negative error asymmetry, over 100 brand-new worlds with seeds 6000 to 6099, under the same three conditions. The predictions were that forgetting would not help, that learning fast would help most in the more volatile worlds, and that the asymmetry would help slightly.

Nearly all of them were wrong.

- **Forgetting helps clearly in volatile worlds.** In the fast worlds, forgetting rates of 3e-3 and 1e-2 raised net reward from +5.2 to +17.4 and +17.7, a gain of +12 with intervals [+9.9, +14.4] and [+9.8, +15.0]. In worlds with inversion periods from 1000 to 3000, rates of 1e-3 and 3e-3 gained about +10. The two registered main conclusions: 1e-2 against 0 is "better", while 1e-3 against 0 is +3.5 with interval [+1.6, +5.4], below the bound of 4, so "uncertain".
- **Forgetting hurts badly in non-inverting worlds.** 1e-3 costs 12.8, 1e-2 costs 40.7, and 3e-2 costs 54.8.
- **Forcing forgetting onto the evolved parameters gives the same answer.** Adding 1e-3 forgetting to the 3 evolved results gains another +8 to +10 on in-distribution worlds, more than the +4 to +7 the evolution found on its own.
- **"Learning fast matters more" holds only half.** Lowering the learning rate to 0.25 times the default was worse in all three worlds; raising it to 2 and 4 times gave no extra benefit, so the default already sits on a learning-rate plateau.
- **"Learn faster on negative error" is not supported.** Doubling or quadrupling the negative-error learning rate alone made no difference in the fast worlds and was worse or uncertain in the non-inverting worlds.

The conclusion was that forgetting helps in volatile worlds and hurts in stable ones, which is consistent with the original guess; what failed was the evolution, which never had the chance to find it.

### Why the search could not find it

The forgetting rate is encoded as

$$
\mathrm{decay} = 10^{(-5 + 3.5 \cdot \mathrm{sigmoid}(\theta))}
$$

The hand-tuned default is "no forgetting", which encodes to a starting point of $\theta = -6$, deep in the flat tail of the sigmoid: moving $\theta$ by one unit changes the forgetting rate by only about $2 \times 10^{-7}$, so candidates differ by no reward difference that could be resolved. Reaching an effective 1e-3 needs $\theta \approx +0.3$, which is 6.3 away from the start, while each generation of the search moves only a few tenths. The history of the first evolution confirms the account: the forgetting parameter stayed around -6 in all 9 runs, between -5.6 and -6.3, with no direction.

So "9 evolutions never chose forgetting" is a property of the parameterization of the search space, not evidence that forgetting is useless. The source notes that this defect was not found at registration and that the review did not catch it either.

### The re-run, and what it costs

A second pre-registration changed exactly one thing: the forgetting parameter's starting $\theta$ moved from -6 to 0, a forgetting rate of about 5.6e-4 in the middle of the sigmoid, with everything else identical. The 9 evolutions were evaluated on 100 brand-new worlds with seeds 6100 to 6199.

- All 9 evolutions selected forgetting, with final forgetting rates from 1.2e-3 to 2.2e-3, confirming the attribution.
- The in-distribution gain moved from +4 to +7 up to +14 to +21. Across three wirings and three seeds, all 9 runs satisfy "lower bound above 0 and gain at least 6", and each seed gains another +8 to +15 over the first evolution.
- The fast worlds gained +7 to +11.
- The stationary worlds got worse by 8 to 15. The training worlds all invert, so evolution learned to forget and pays for it in worlds that never invert.
- The wiring conclusion did not change: real minus shuffled was +2.8, -0.9, and -1.0, with all intervals inside plus or minus 6, which counts as equivalent; real minus random sparse was -5.2, -8.9, and -4.3, with random sparse consistently better in direction but reaching "clearly better" on only 1 seed by the criterion.

So "evolution only produces small gains" was a consequence of the first search being stuck. After unlocking it, automatic tuning gains about three times as much in this setting, and the cost is explicit: worse in stable worlds that training never showed.

## How far this is from recursive self-improvement

The source lists three gaps.

1. **Only five numbers change.** The form of the rule, the structure of the circuit, and the encoding of odors are all fixed, so the outer loop never has the chance to discover a new learning rule.
2. **The outer loop is itself fixed.** Population, step size, and number of generations are all set by hand. The outer loop is not improved, so there is one layer and no recursion.
3. **The search space of the outer loop determines what it can find.** The forgetting episode is the direct example: the same evolution strategy, differing only in where the forgetting parameter starts, moves the gain from +4 to +7 to +14 to +21. A self-improving system first has to make sure its own improver has no blind spot.

The limits the source records are equally specific: only 3 evolution seeds, with intervals bootstrapped over worlds and not over the variation of the evolution itself; an objective that is mean reward on training worlds, so a different set of worlds may prefer different parameters, as the stationary drop shows; hand-tuned defaults that were themselves chosen on tuning worlds rather than being an arbitrary starting point; a re-run starting point of $\theta = 0$ that deliberately carries some forgetting, with no test of whether other starts reach the same place; and only five parameters, with no structural search.

## Where the Chinese original lives

The Chinese original is [Let Evolution Tune the Learning Rule: How Far Is It From Recursive Self-Improvement?](/posts/2026-10-21-fly-connectome-05-meta-learning/). It holds the two-loop diagram, the parameter table, the forgetting-sweep and re-run figures, the per-seed gain tables for all three conditions, the attribution of the search failure to the sigmoid encoding, and the pointers to the pre-registrations and result files. Its limit is the one stated above: this is one task, one set of five parameters, and three evolution seeds, and the intervals describe world variation rather than the variation of the search itself.

The next note in this series covers [the three traveling-salesman rounds](/en/posts/fly-connectome-06-traveling-salesman/).
