---
sourceId: 2026-10-21-fly-connectome-06-tsp-three-rounds
slug: fly-connectome-06-traveling-salesman
title: "Three Traveling Salesman Rounds, Narrowed Each Time"
editionLabel: ENGLISH OVERVIEW · ANALYSIS NOTE
published: 2026-10-21T00:00:00.000Z
description: "Three reviewed rounds of a traveling-salesman comparison, each finding the learner's benefit within 0.5 percentage points of ant colony optimization."
tags:
  - Traveling Salesman
  - Ant Colony Optimization
  - Feature Mapping
  - Null Result
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 6
lang: en
---

> **Scope note.** This is an English overview of the sixth Chinese article of the Fly Connectome Research series. It reports three successive rounds of a traveling-salesman comparison, each followed by an independent review that narrowed the conclusion. Within the ant colony optimization framework used here, the learner's benefit never exceeded 0.5 percentage points, and the real wiring was equivalent to shuffled wiring, random sparse wiring, and a linear learner. The source's final wording is "no evidence" rather than "no use", because the design could not detect a wiring effect even if one were present: the edge features duplicate distance, the turn information was never expressed, the learning budget was small, tuning noise was about 0.4 percentage points, and there was no positive control. This article does not compare against DeepACO or any other published method, and it does not establish that real wiring is useless.

## Why the traveling salesman, and the ant colony framework

The foraging task has too many hand-written parts — the odor field, the reward schedule, the turning rule, the ant colony behavior — to say whether a conclusion came from the circuit or from the world. The traveling salesman problem has an objective score: given a set of cities, visit each one, and a shorter route is better, and it has mature strong baselines.

The positioning is deliberately narrow. The goal is not to show that a fly algorithm beats DeepACO. Methods of that kind already use a framework in which a cross-instance neural heuristic is learned first and pheromone is then learned within an instance, and all of them are neural networks trained for the task. The question here is narrower still: within that same framework, is replacing the feature mapping with the mushroom body circuit better than random sparse features, shuffled wiring, or a linear learner?

Ant colony optimization in one minute: many "ants" each walk a tour, and at every city the next stop is chosen by a weight that combines three terms.

- **τ, the pheromone.** Traces left by other ants during this search; edges on good routes carry stronger traces. It is learned within a single search, one number per edge.
- **η, the inverse distance.** A fixed prior that favors nearby cities.
- **v, the learned edge value.** The added term, supplied by a mushroom body trained across instances. Setting v to 0 gives plain ant colony optimization.

The "odor" of an edge is hand-written: how many nearest neighbors the edge's destination is of its origin, a distance bin, whether the two are mutual nearest neighbors, the origin's neighbor rank at the destination, and the distances of both endpoint cities from the centroid. Each edge activates 6 glomeruli, drawn from 52 in total. The learner changes only where the last term v comes from; everything else stays fixed.

## Round one: pre-trained edge value

On 100 training instances, the ants walk without pheromone first and the advantage of a route, how much better than the batch average, is used as the dopamine signal that updates the mushroom body. After training, the body is frozen and used on unseen instances, where the frozen v works together with within-instance pheromone.

The first pitfall was the learning rate. Reusing the foraging value of 0.4 saturated every mushroom body variant's values near +0.95, leaving no discrimination between edges, and the zero-shot routes came out longer. Dropping to 0.05 restored discrimination. Learning rate and gain have to be tuned per learner and per task on separate tuning sets.

Results over 100 test instances, expressed as how much longer the route is than a reference solution, in percent:

| Setting | Ant colony | MB real | MB shuffled | MB random sparse | Linear |
| --- | --- | --- | --- | --- | --- |
| R1 uniform 50 cities | 2.90 | 2.69 | 2.70 | 2.68 | 2.93 |
| R2 uniform 100 cities | 3.71 | 3.39 | 3.37 | 3.51 | 3.36 |
| R3 clustered 50 cities | 4.01 | 4.14 | 4.34 | 4.01 | 4.05 |

Real against shuffled, against random sparse, and against the linear learner: equivalent in all three settings. Real against ant colony: -0.21 in R1 with interval [-0.59, +0.14], and -0.33 in R2 with interval [-0.72, +0.05].

The review downgraded three claims.

- "No feature mapping beats plain ant colony optimization by 0.5 percentage points" was too strong: the lower bounds of the two intervals reached -0.59 and -0.72, so larger improvements were not excluded. The accurate statement is "uncertain".
- The R2 advantage on the tuning set, ant colony at 4.84 against 2.68 for the real learner, largely vanished on the test set. Picking the best of 12 configurations overestimates, which is the winner's curse.
- The assignment of features to glomeruli is arbitrary, and real glomeruli fan out very differently, so whether a feature lands on a hub glomerulus or a rare one is random. Detecting wiring specificity was inherently difficult in this setting.

## Round two: four structural changes and 300 instances

One cause of the round-one "uncertain" was too few instances: the standard deviation of the paired difference was 1.4 to 1.8 percentage points, so 100 instances barely sufficed to resolve 0.5. Round two raised the test instances to 300 per setting, ran the ant colony with 3 search seeds as well, and changed four things structurally.

1. **Online learning.** No pretraining. The mushroom body learns during a single search, treated as a generalizable pheromone: pheromone is one number per edge, while the mushroom body lets similar edges share value.
2. **A heterogeneous population.** Four "flies", each with a different circuit, each ant using its own circuit. Individual differences are simulated by randomly dropping 30% of the KCs, and this is explicitly not real individual difference, because no aligned FlyWire matrices are available.
3. **Communication structure.** All circuits sharing the learning signal, by broadcast, against each keeping it private; and four sub-colonies each on its own, against periodically exchanging elite routes.
4. **Local search.** A 2-opt pass on every route. It is industry practice, and without it there is nothing to compare against.

Tuning used 27 configurations for the ant colony and 9 for each online learner. The optima all landed on grid edges, so the grid was widened and tuning was redone. The ant colony settled at ρ = 0.9 on R1 and R2, and 0.5 on R3, meaning the pheromone almost only remembers the previous round's best.

Over 300 new instances, every main comparison fell within plus or minus 0.5 percentage points:

- Online learning with the real wiring against ant colony: -0.05 in R1, -0.04 in R2, and -0.01 in R4 with 2-opt, all equivalent. On the clustered R3 it was -0.26, interval [-0.44, -0.07].
- Heterogeneous against homogeneous: differences from 0.00 to +0.07, equivalent.
- Real against random sparse, shuffled, and linear: differences all at most 0.15.
- Removing the pheromone made routes 5 to 10 percentage points worse. The pheromone is the most important within-instance memory, and the online-learned edge value cannot replace it.
- Four sub-colonies that never exchange were 0.6 to 1.7 percentage points worse than a single colony. Exchanging the best route every 5 rounds recovered part of that: still not better than the whole colony on R1 and R2, where R2 was worse by 0.61, and slightly better on the clustered R3 at -0.33, still inside the bound.

The small clustered gain deserves its own note. Online learning was about 0.3 percentage points better than the ant colony, and it reappeared once on a fresh batch of instances, at -0.28 and -0.35 on the new instances of v3. But random sparse wiring does equally well, so this is a small gain unrelated to whether the wiring is real, and it is below the preset bound of 0.5. The source calls it stable but small.

There was a further episode around the R4 setting. The 2-opt configuration started at n = 50 with 100 constructions, and every arm came in below the reference solution at -0.17%, with no discrimination. It was changed to n = 100 with 30 constructions, which narrowed the equivalence bound to 0.25. The change happened after the tuning results were seen but before the test was run, and it was recorded in the deviation log.

The review found four problems. The edge features nearly duplicate distance: the learned value correlates with the log of inverse distance at 0.85, with a standard deviation of only 0.077, so every circuit is "distance plus a little smoothing bias" and their equivalence is a necessary consequence of the design. Each instance has about 530 distinct edge signatures while learning runs for only 25 rounds, so the learning budget is too small. Equivalence hides small significant effects, since the R3 interval of [-0.44, -0.07] excludes 0 and the gap between exchanging and not exchanging sub-colonies is also clear, so these should be written as small effects below the bound. And the analysis script was modified once after registration, a title change only, so its hash no longer matched.

The final statement for round two: no improvement beyond the bound; the real wiring has no advantage, and this design cannot detect a wiring effect.

## Round three: state-dependent turn information

The review's suggestion was to change the feature rather than add more algorithm. The feature chosen is state-dependent: the turn angle. Knowing the previous stop tells you whether this step is a sharp turn, and neither distance nor pheromone, which look at a single edge, can supply that.

A turn bin is added to the transition signature, with levels for straight, slight turn, right angle, U-turn and others, 7 levels plus "no previous stop", occupying 8 new glomeruli. A hand-written control accompanies it, a fixed turn penalty of the form exp(-γ times (1 minus cosine of the turn angle)), which does not learn and answers whether turn context helps at all.

Over 300 brand-new test instances, the learner against ant colony was -0.01 in R1, +0.03 in R2, and -0.18 in R3, all equivalent, and the hand-written turn penalty was 0.32 worse in R2.

The most consequential review finding looked inside the learner. Within the same static signature, the residual of the learned value after binning by turn angle was only plus or minus 0.01 to 0.02 for the real wiring with no monotonic trend; plus or minus 0.04 for the random wiring with inconsistent signs; and plus or minus 0.1 to 0.26 for the linear learner, whose signs were inconsistent across instances and therefore noise.

The learner had not learned the turn information. The turn accounts for one of the six input glomeruli of an edge, so the 5% KC encoding is nearly the same across turn bins, and with only 25 to 50 learning steps over about 4000 distinct signatures the signal is drowned. Real about equal to random is therefore a necessary consequence of the design and cannot serve as evidence that the wiring does not matter.

The hand-written penalty tests one single-parameter, single-direction hypothesis, a preference for going straight, so it supports only the statement that a straight-preference penalty did not help under these budgets, not the statement that turn context is useless.

## Why the conclusion says no evidence rather than no use

Saying "no use" requires an experiment that would detect use when use is present, and this setup does not satisfy that:

- the features duplicate distance, so the learner has little new information to learn;
- the turn information was never expressed;
- the learning budget is small;
- tuning noise of about 0.4 percentage points is the same magnitude as the effect being measured;
- there is no positive control, meaning no known regularity was planted in the task to show that this setup can detect it.

What can be written is that under these settings no advantage of the real wiring was measured, and that the design itself may not be able to measure it. Getting a conclusion requires a positive control first: implant a structure known to depend on an odor combination, confirm that the circuit learner can detect it, and only then compare real against ablated wiring.

## Where the Chinese original lives

The Chinese original is [Three Traveling Salesman Rounds: Why the Conclusion Says No Evidence Rather Than No Use](/posts/2026-10-21-fly-connectome-06-tsp-three-rounds/). It holds the ant colony diagram with the learned edge values, the per-round result tables, the forest plot of the main comparisons across all three rounds, the review findings with the residual tables, the wording section, and the pointers into the `fly-tsp/`, `fly-tsp/v2/`, and `fly-tsp/v3/` directories with their pre-registrations, result files, and verification reports. Its limit is the one stated above: the comparison varies the feature mapping inside one ant colony framework, and without a positive control the design cannot tell a real absence of advantage from an inability to detect one.
