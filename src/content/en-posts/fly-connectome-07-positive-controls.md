---
sourceId: 2026-10-21-fly-connectome-07-positive-controls-natural-odours
slug: fly-connectome-07-positive-controls
title: "A Non-Circular Positive Control Detects the Wiring Effect, but Natural Odours Show No Advantage"
editionLabel: ENGLISH OVERVIEW · ANALYSIS NOTE
published: 2026-10-21T00:00:00.000Z
description: "A non-circular positive control detects the wiring effect, but no representational advantage of the real wiring appears on DoOR odours."
tags:
  - Fly Connectome
  - Positive Controls
  - Negative Results
  - Pre-registration
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 7
lang: en
---

> **Scope note.** This is an English overview of the seventh article in the Fly Connectome Research series, a Chinese analysis note. The source reports two experiments: a teacher–student positive control, rebuilt so that teacher and student no longer share the same wiring instance, and a representation-level test on DoOR 2.0 odour response vectors. It establishes that the comparison pipeline can detect a wiring effect when one is genuinely present, and that no representational advantage of the real wiring was measured on natural odours. It does not show that real fruit fly wiring is meaningless, and the model remains a proxy rather than a fly.

## Why a positive control has to come first

The six preceding articles in the series shared one gap: no positive control. Nothing in them had confirmed that the comparison pipeline would detect a wiring effect if one existed. The distinction matters because "no difference was measured" has two possible causes — the wiring has no effect, or the test cannot see one.

The rule the source applies is strict: before writing "no difference", first show that the design would have found a difference known to be there. A comparison that has never demonstrated this can produce a null result for two incompatible reasons, and the result alone cannot tell them apart.

The source inherits the decision rule fixed in article 2 of the series. Intervals are produced by bootstrapping over worlds, and each comparison is judged on a three-tier scale: meaningful, small but real, and equivalent. The minimum meaningful effect is 0.03 accuracy points, chosen in advance. A null result is therefore a statement about a comparison with a stated bound, not a statement about biology.

The three tiers are what keep "equivalent" from becoming a loose synonym for "no difference". A comparison counts as equivalent only when the whole interval falls inside the bound, and an interval that merely touches the bound leaves the verdict at uncertain. This is the machinery this article depends on, because the first positive control it inherits produced a difference that was real and still not attributable to the thing under test.

## The first positive control was circular

The first version used the teacher–student task already in use. A mushroom body teacher with real wiring produced the labels; the students were the real, shuffled, and random-sparse wirings. The real student scored about +0.08 above the other two, clearing the pre-set threshold.

Independent review rejected the result as circular. The teacher drew on exactly the feature set the real student already possessed, so the real student won only because it shared the teacher's features. The comparison showed that the pipeline can detect a feature-matched student, not that it can detect the effect of wiring on a natural task. The same problem affected a companion task in which a real-wiring teacher labelled real odours.

The same review round also found two metric defects, both of the kind that only running the code exposes. The silencing procedure turned a silenced Kenyon cell into a maximally suppressed cell rather than an absent one, so "silenced" and "removed" were being counted as the same intervention. And the quantity labelled "accuracy after 50 trials" was in fact reading trial 40. Both were corrected in the next registration; the superseded results were retained and marked void rather than deleted.

## A non-circular positive control, and what it detected

The rebuilt control separates teacher from student. Two variants were used.

- **Teacher reshuffled by 30%.** The teacher still has real wiring, but 30% of each Kenyon cell's input edges are moved to other olfactory glomeruli. The real student has a largely matching wiring; the shuffled and random-sparse students do not.
- **Teacher drawn as a new shuffled wiring.** The shuffled wiring used by the students is a different instance, sharing only the statistical class: the degree and synapse-count distributions.

The outcomes were as follows.

| Comparison | Difference | Interval evidence | Reading |
| --- | --- | --- | --- |
| Real student vs shuffled student, 30% reshuffled teacher | +0.064 | lower bound of the 99.2% interval above 0.03 | detected |
| Real student vs random-sparse student, 30% reshuffled teacher | +0.066 | lower bound of the 99.2% interval above 0.03 | detected |
| Shuffled vs random-sparse student, statistically matched teacher | +0.001 | judged equivalent | equivalent |
| Shuffled vs real student, statistically matched teacher | +0.018 | no measurable advantage | no measurable advantage |

An independent re-implementation was run as a check on the code itself. A sub-agent was given only the model and task in text, and neither the original code nor its results, and wrote its own implementation. It returned +0.065, +0.064, −0.002, and +0.020, all inside the original intervals, which rules out an artifact of code details.

Two of these lines carry the interpretation. The pipeline has been shown to see a wiring effect, and a shared statistical class does not reproduce it. Neither line says anything about whether the real fly wiring is useful for anything.

## Natural odours: a representation-level test on DoOR

The second experiment drops the teacher and the reward rule, so no feature match is left to exploit. The input is the DoOR 2.0 response matrix over 301 odour molecules and 65 olfactory glomeruli:

$$
X \in \mathbb{R}^{301 \times 65}.
$$

The Kenyon cell encoding keeps the top 5%, and two tasks are run. Both were chosen so that the teacher–student machinery plays no part: there is no teacher, no reward rule, and no label transferred from one wiring to another, so whatever the encoding does here is what the Kenyon cell code alone does.

- **Few-shot class classification.** Nine chemical classes, with 2 or 5 molecules per class used for training and a nearest-centroid classifier. The metric is balanced accuracy; the random level is about 0.11.
- **Neighbour retention.** For 150 odour molecules, the overlap between the 5 nearest neighbours in Kenyon cell encoding space and in DoOR space.

The results do not favour the real wiring.

- On class classification the real wiring was at most about 0.02 above the shuffled and random-sparse wirings. The 5-shot comparison against the random-sparse wiring was +0.020 with an interval of [+0.002, +0.040], short of the 0.03 minimum meaningful effect.
- On neighbour retention the real wiring was slightly worse than the shuffled wiring, at −0.026.
- Expanding and then sparsifying the code was no better than using the 65-dimensional vector directly: on 2-shot the direct vector was slightly higher, and on 5-shot the two were comparable.

The direction of the neighbour-retention difference is worth stating as it stands. The source reports the real wiring as slightly worse, without attaching an interval to it or converting it into a measured disadvantage. Read as a null, it is a null that did not go in the real wiring's favour.

## What these two results license, and what they do not

The source draws the boundary explicitly.

**Supported by this evidence:** the pipeline detects a wiring effect when the wiring instance really does contribute; on natural-odour class classification and neighbour retention, no representational advantage of the real wiring was measured.

**Not supported by this evidence:** that real wiring carries no meaning for natural odours. DoOR covers only about 47 of the glomeruli, and unmeasured entries are treated as zero, so the odour vectors are sparse. 72 of the molecules carry no class label. Class classification uses a nearest-centroid rule rather than a learned readout. The chemical classes were given no ecological valences either: the literature review found no verifiable class-valence table, and random class valence had already been tested in earlier experiments, where all wirings came out equivalent.

Two further limits are recorded. The positive control is a teacher–student task, so it measures agreement with the teacher's wiring rather than natural adaptivity, and the 30% reshuffle was tested at a single level. The natural-odour test rests on one dataset and one encoding rule. The re-implementer and the original implementer are the same model family, working from the same data and the same written specification.

The negative-result framing is the substance of the article, not a caveat attached to it. The measurement that did succeed — detecting a wiring effect — is the one that licenses the null: without it, "no advantage on natural odours" would have been indistinguishable from a test that could not see the thing it was looking for.

## Where the Chinese original lives

The Chinese original is [Positive Controls and Natural Odours: Detecting the Effect of the Circuit, without Finding an Advantage on Real Odours](/posts/2026-10-21-fly-connectome-07-positive-controls-natural-odours/). It carries the experiment records behind both experiments — the pre-registrations for the non-circular control and for the natural-odour stage, the stage notes, and the replication comparison — together with the figures for the positive control and the natural-odour test, and the pointers into the project repository at <https://github.com/meiluosi/fly-connectome-study> (directory `fly-v4/`).

Its main limit is the one stated above: the pipeline was shown able to detect a wiring effect, and no representational advantage of the real wiring was measured on natural odours. That is a null result about the comparison that was run, on one dataset and one encoding rule, not a finding about the fly. The next article in the series takes the individual-difference question: whether the same holds when the wiring comes from a different fly.
