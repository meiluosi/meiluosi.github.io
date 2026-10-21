---
sourceId: 2026-10-21-fly-connectome-11-summary
slug: fly-connectome-11-negative-result-summary
title: "No Measured Advantage for the Real Wiring: What a Negative-Result Project Leaves Behind"
editionLabel: ENGLISH OVERVIEW · REFLECTION NOTE
published: 2026-10-21T00:00:00.000Z
description: "Across eight lines of work no arm showed the real wiring ahead, and this note states what that null result does and does not license."
tags:
  - Fly Connectome
  - Negative Results
  - Pre-registration
  - Methodology
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 11
lang: en
---

> **Scope note.** This is an English overview of the eleventh and closing article in the Fly Connectome Research series, a Chinese reflection note. It summarises eight lines of work in which the real fruit fly PN→KC wiring was compared against shuffled, random-sparse, and linear controls, and it states what the project can and cannot claim. The headline is a null result: no arm showed the real wiring ahead. That null is a finding about the comparisons that were run on the tasks that were built, not a finding about the fly, and the source is explicit that it must not be read as "the real wiring is useless".

## Eight lines of work, and what each produced

The project began from a larger question — can a system improve its own way of learning — and landed on a learner with real structure, the fruit fly mushroom body. Eight lines of work followed.

| # | Line of work | What it produced |
| --- | --- | --- |
| 1 | Compartments and forgetting | The apparent advantage came mainly from "learning less", and disappeared once the amount learned was matched |
| 2 | Continual-learning audit | FlyModel is surpassed by replay-free SLDA, and most of the paper's sparse advantage came from the control itself |
| 3 | Foraging and fly swarm | The mushroom body does learn, and random-sparse wiring is actually better; more shared information is better and clustering is harmful |
| 4 | Meta-learning | The first evolution only reached +4 to +7, because the encoding of the forgetting rate pinned the starting point on a flat region. After fixing the starting point and re-running, all 9 evolutions chose forgetting, in-distribution performance improved to +14 to +21, and the cost was worse behaviour in worlds without reversal |
| 5 | Traveling salesman, three rounds | Online learning, a heterogeneous population, communication, 2-opt, and turning context: none of the gains exceeded 0.5 percentage points |
| 6 | Positive control and natural odours (article 7) | The non-circular positive control can detect the effect of a wiring instance; no representational advantage was measured on natural odours |
| 7 | Three flies (article 8) | Another fly's wiring is a little better than random, and comparable to the control that keeps only glomerular input mass |
| 8 | Compartments and dopamine (article 9) | The timing effect reproduces; sample efficiency is not better than the base rule; continual learning shows a marginal retention advantage that cannot be attributed to compartmentalisation; foraging shows no advantage |

Two entries are worth separating from the rest, because they are positive findings rather than null ones. In line 3 and line 5 the mushroom body circuit does learn — that much the project established — while the comparison of wirings within those tasks went the other way. In line 4 the meta-learning result changed sign once the starting point was repaired, which is a methodological finding about the search rather than a property of the circuit.

## What can be said, and what cannot

The source assembles an evidence map over the first five lines, with the conclusions of the last three given in the list above and in their own articles. Not a single cell reads "the real wiring is clearly better". In foraging and meta-learning the random-sparse wiring is actually better; in the traveling salesman the different wirings are equivalent; and on natural odours the best difference stays below the minimum meaningful effect.

The map matters because it records where the arms were not close. In the foraging and meta-learning lines, the control was not merely equal to the real wiring, it was ahead of it. That is a different kind of result from a difference of zero, and the source does not soften it into one.

**Supported by this evidence:** on these pre-registered tasks, the real PN→KC wiring shows no measurable difference from degree-preserving shuffled wiring, and is not better than random-sparse wiring. The pipeline can detect the effect when the wiring instance really does contribute, as the positive control with a teacher reshuffled by 30% shows.

**Not supported by this evidence:** that the real mushroom body wiring is meaningless for computation. The tasks were constructed for the purpose of comparison; there is no real behavioural or valence data behind them; and PAM = reward, PPL1 = punishment are only commonly used rough rules.

**Also not supported:** that "Kenyon cell wiring is close to random". Zheng et al. 2022 reported structured, non-random PN→KC wiring. The project's null result is about task performance, not about whether the wiring is random.

The shortest version the source offers: under these settings there is no evidence that the real wiring helps, and in several early settings the design could not even detect such a difference. That is different from "it has been proven not to help".

## How much of this to trust

| More trustworthy | Discount |
| --- | --- |
| The mushroom body circuit does learn, much better than not learning (foraging +35 to +44) | The tasks are toy-like: the teacher–student task measures agreement with the teacher's wiring, and the foraging field is a simplified setting |
| FlyModel is surpassed by replay-free SLDA (consistent across three benchmarks); the original paper has no SLDA baseline | The backpropagation baseline on CIFAR is very weak, and no independent replication of FlyModel was found |
| The non-circular positive control was reproduced by an independent re-implementation (+0.065) | The positive control is itself a teacher–student task, and the 30% reshuffle tested only one level |
| The compartment forgetting advantage mainly comes from learning less; the marginal advantage in continual learning comes from the rule family | The compartment learner's advantage magnitude is +0.04 to +0.05, with interval lower bounds below 0.03 |
| The cross-individual advantage is essentially glomerulus-level input mass | Only three flies, and the numbers depend on the Kenyon cell count (2 to 4 times smaller in the re-implementation) |
| Global multiplicity (67 comparisons) does not change any stage's headline conclusion | A normal approximation back-derived from the intervals; the same hypothesis is tested in multiple stages, so the tests are not independent |

Review and re-implementation were carried out by sub-agents of the same model family, and the literature review read most papers only at abstract level. Those limits are set out item by item in article 10 of the series.

The table is best read as two lists rather than one ranking. The left column is a set of things that survived the project's own adversarial checks, and each of them was checked by something other than the code that produced it: a positive control that was reproduced by an independent re-implementation, a global multiplicity correction over 67 comparisons, a comparison against a control built to be matched on dose rather than on count. The right column is the set of discounts that apply even to those, and it is deliberately specific about what would remove each one.

## Five lessons

1. **Without a positive control, do not write "no effect"; and the positive control must be non-circular.** The three traveling salesman rounds and the first version of stage 0 both failed this test.
2. **Look at the absolute level of controls and baselines first, then at what else differs between the two sides.** A dense control stuck at the random level and a baseline stuck at the random level each mean the comparison itself is broken. Write down whether the match was on count or on effective dose.
3. **Separate representation and classifier in the comparison, and align tuning, budget, and data flow.** If the best point of the tuning grid sits at the edge, relax it and re-run; learning rules need their numerical range checked.
4. **The optimizer's search space can decide the conclusion.** In meta-learning, none of the 9 evolutions chose forgetting, and it nearly got written up as "forgetting is useless"; a manual sweep showed forgetting is very useful in fluctuating worlds, and the encoding of the forgetting rate had put the starting point on a flat region.
5. **A review has to be able to run the code, and every finding has to be verified first.** Same-model review is not an independent error source; its value is in running the code rather than reading the document again.

The fourth lesson is the one that would have been invisible without it. A negative result produced by a search that never visited the relevant region is a statement about the search, and the project nearly published it as a statement about the biology.

## What the project contributes, and what would come next

The scientific contribution is small, and the source says so plainly. "Forgetting depends on how much you learn" is common sense in continual learning, the observation that Kenyon cells in a whole-brain LIF model are not sparse has already been reported, and no systematic literature search was carried out, so none of it can be claimed as new. The model is also a weak proxy for the real mushroom body: "no effect found" says nothing about real flies.

The main value is in process and engineering: pre-registration, hash freezing, independent review, independent re-implementation, corrections to overturned conclusions, a reproducible pipeline, and a checklist for not fooling oneself. The source is careful about the status of this value too: the protocol guarantees that the criteria were not changed after the fact, not that the design could detect the effect it looked for.

If the work continues, the source lists six steps: trace the key citations to the original text, since many remain at abstract level or unverified; repeat the key comparisons on tasks closer to natural behaviour, with real odours and sourced valence annotations; obtain more individuals, since a fly's two hemispheres do not count as independent individuals, and build a comparison that explicitly models the differences in synapse-detection method; test the valence hypothesis "PAM = reward, PPL1 = punishment" against behavioural or activity data; give the learner features that are not duplicates of existing information, as the traveling salesman rounds needed something beyond distance and pheromone; and obtain one external review by a human expert.

Two of those six decide what the next round can claim at all. A task with real odours and sourced valences is the only way to test the compartment hypothesis at all, since the current tasks supply the valences by construction and therefore cannot falsify them. And more independent individuals are what would turn the "mostly glomerular input mass" result from a three-fly observation into something with a sample behind it.

## Where the Chinese original lives

The Chinese original is [The Real Wiring Showed No Advantage: What a Negative-Result Project Actually Leaves Behind](/posts/2026-10-21-fly-connectome-11-summary/). It carries the evidence map, the full project record set, the pointers into the project repository at <https://github.com/meiluosi/fly-connectome-study>, and the reference to the protocol work packaged as reusable skills at <https://github.com/meiluosi/research-writing-skills> (the `preregistered-experiment`, `independent-review`, and `result-writeup-editor` skills, MIT). The first ten articles of the series cover the circuit and comparison framework, the protocol and review, foraging and the fly swarm, the continual-learning audit, meta-learning, the three traveling salesman rounds, positive controls and natural odours, individual differences, compartments and dopamine, and the review checklist.

Its main limit is the one stated above: there is no evidence that the real wiring helps in the settings tested, and in several early settings the design could not have detected such a difference. That is a null result about the comparisons that were run, on tasks built for comparison, with no behavioural or valence data behind them.

The closing distinction is the one the whole series was built to earn. A null result is a finding about the comparison that was run. It is not a finding about the biology, and in this case the more that holds: the tasks were built to compare wirings, the valence rule behind them is a literature convention, and several of the comparisons could not have detected a difference even if one was there.

For that reason the source states the null twice, in two registers, and the second is the one to quote. The long form says that on these pre-registered tasks the real wiring showed no measurable difference from degree-preserving shuffled wiring. The short form says there is no evidence that the real wiring helps in these settings, and in several early settings the design could not even detect such a difference. Neither says that the real wiring does not help.
