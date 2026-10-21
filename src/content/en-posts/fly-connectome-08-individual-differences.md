---
sourceId: 2026-10-21-fly-connectome-08-individual-differences
slug: fly-connectome-08-individual-differences
title: "Three Flies, Five Hemispheres: The Cross-Individual Advantage Is Mostly Glomerular Input Mass"
editionLabel: ENGLISH OVERVIEW · RESEARCH NOTE
published: 2026-10-21T00:00:00.000Z
description: "Real wiring from another fly beats random sparse only slightly, and matches a control that keeps only glomerular input mass."
tags:
  - Fly Connectome
  - Connectome
  - Negative Results
  - Pre-registration
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 8
lang: en
---

> **Scope note.** This is an English overview of the eighth article in the Fly Connectome Research series, a Chinese research note. The source tests whether the advantage of real mushroom body wiring survives a change of individual, using three flies and five hemispheres from three connectome datasets in a teacher–student comparison. It reports that another fly's real wiring beats a random-sparse control only slightly, and is on par with a control that preserves glomerulus-level input mass while discarding which Kenyon cell connects to which glomerulus. It does not show that the wiring is random, and it does not compare the model against real behaviour.

## Three datasets, five hemispheres, one alignment

Every earlier experiment in the series used one individual: the right hemisphere of MaleCNS. If "real wiring is useful" holds, it should hold across individuals — one fly's wiring serving as the student for another fly's task should also beat random wiring. The source tests that with three flies.

| Hemisphere | Source | Sex | Notes |
| --- | --- | --- | --- |
| M_R, M_L | MaleCNS | male | right and left hemispheres |
| F_R, F_L | FlyWire (v783) | female | right and left hemispheres |
| H_R | hemibrain v1.2 | female | right hemisphere only; 1927 Kenyon cells, of which 1801 have projection neuron input |

Glomeruli are aligned to the 65 of MaleCNS by taking the prefix before the underscore in the projection neuron type name. FlyWire lacks 2 to 3 glomeruli, and hemibrain covers 62 of the 65.

The number of Kenyon cells is not fixed across the experiments reported here, and it turns out to matter. The hemibrain hemisphere contributes 1927 Kenyon cells, of which 1801 have projection neuron input. The first cross-individual experiment ran with 2018 Kenyon cells, the independent re-implementation with 1800, and the experiment that added hemibrain with 1801. The source returns to this discrepancy in its robustness section, because the size of the reported advantage scales with it.

Mean total synaptic weight per Kenyon cell is about 97 in MaleCNS, about 91 in hemibrain, and 57 to 69 in FlyWire. The three datasets use different synapse-detection methods, so those values are not directly comparable. The hemibrain data (45,872,577 bytes, Janelia FlyEM, CC BY 4.0) was downloaded into a separate directory and kept out of the code repository.

## Teacher in one fly, student in another

The teacher–student task is kept and the individual is changed. Labels come from the teacher individual's wiring; the student is another individual's real wiring. Two controls are used.

- **Random-sparse wiring:** each Kenyon cell randomly draws 6 glomeruli with weight 1.
- **Input-mass-preserving control (marg):** keeps each student's Kenyon cell degrees and weights, and samples glomeruli in proportion to the total glomerular weight of the student's hemisphere. It drops the detail of which Kenyon cell connects to which glomeruli and keeps only which glomeruli carry a lot of input.

To keep all wirings the same size, the same number of Kenyon cells is drawn at random from each hemisphere.

Three flies give five usable hemispheres: one male fly contributes both of its hemispheres, one female fly contributes both of its hemispheres, and the hemibrain female contributes only the right. The article treats the two hemispheres of a fly as the same individual, which is also why the project's own closing note lists "more individuals" as outstanding work.

The second control is the load-bearing one, and it is the control that the earlier articles in the series did not have. A random-sparse wiring discards both the assignment of Kenyon cells to glomeruli and the glomerular input totals; the input-mass-preserving control discards only the first. If real wiring were ahead of a random-sparse control, the question would then be whether it was ahead for the reason the control had removed.

## What the cross-individual comparison found

| Comparison | Range of differences | Reading |
| --- | --- | --- |
| Another fly's real wiring vs random-sparse wiring | +0.010 to +0.036 | slightly ahead, with most differences below 0.03 |
| Another fly's real wiring vs the input-mass-preserving control | −0.009 to +0.006 | essentially equivalent; two comparisons were negative |
| Real-odour task (real odours with random class labels), every hemisphere vs both controls | within 0.01 of one another | no separation |

The reading that follows is narrow. What transfers between individuals, in the amount this task can detect, is how much input arrives at glomeruli in total — not the specific assignment of Kenyon cells to glomeruli. The first line of that table can still be read as a weak advantage for real wiring; the second removes most of it, because the control it compares against already keeps the glomerular input totals along with the Kenyon cell degrees and weights.

The third line is a separate check rather than a continuation of the teacher–student comparison. On the real-odour task the labels are random class labels, so no wiring can be better positioned than another by construction, and the finding is that every hemisphere, the input-mass control, and the random-sparse wiring sit within 0.01 of one another. The real-odour task therefore neither supports nor undermines the cross-individual result; it is reported as a case where no arm separated at all.

## Robustness: the numbers depend on the Kenyon cell count

An independent re-implementer, given only the written specification and using 1800 Kenyon cells, obtained cross-individual advantages 2 to 4 times smaller than the first experiment, which used 2018 Kenyon cells: male teacher with female student, +0.036 against +0.010; female teacher with male student, +0.027 against +0.019. The experiment that added hemibrain, at 1801 Kenyon cells, produced values (+0.012 to +0.026) consistent with the re-implementation's magnitude.

The direction agrees across all three implementations. The magnitude does not, so no single one of these numbers should be cited on its own. A related claim behaved the same way: "the other hemisphere of the same fly is better" did not replicate on male teachers, and did replicate on female teachers.

The practical consequence is that the article's headline is a direction, not a number. What can be carried forward is that another individual's real wiring beats a random-sparse wiring by a small amount whose size depends on how many Kenyon cells are in the comparison. Any downstream use of these results has to carry that dependence with them, which is why the source reports the ranges rather than a single headline figure.

## What the literature does and does not settle

- Zheng et al. 2022 reported structured, non-random PN→KC wiring in FAFB: a core community of projection neurons responsive to food, z = 13.8, but a random null model that preserves local spatial information almost erases the effect; in their discrimination model, random glomerulus networks performed best.
- Caron et al. 2013 has limited statistical power and cannot be treated as evidence that the wiring is random.
- The claim that glomerulus-level input mass is conserved across individuals has no direct literature source; it is a result from this project's own data. The mean number of glomeruli per Kenyon cell also differs across datasets: 5.2 in FAFB, 5.6 in hemibrain, 7 in Caron.

The Zheng result is worth stating carefully, because what it shows depends on which null model is used. A core community of food-responsive projection neurons was reported with z = 13.8, yet a random null model that preserves local spatial information almost erases the effect, and in their discrimination model random glomerulus networks performed best. A null result under one null model is not a null result under another, which is the same reason this article keeps the degree-preserving and input-mass-preserving controls apart from the random-sparse one instead of treating all of them as "shuffled".

The null result here is therefore about task performance, not about whether the wiring is random. The source is explicit about that distinction, and the Zheng result is the reason: structured non-random wiring has been reported elsewhere. The literature position of this article is that the question is open, not settled in either direction.

## The limits that bound the claim

Only three flies and two sexes are available, and the three datasets use different synapse-detection methods, so a difference in mean synaptic weight between datasets is not by itself evidence of a difference in wiring. The teacher–student task measures agreement with the teacher's wiring, not natural adaptivity, and neither the teacher nor the student is backed by behavioural or valence data. Kenyon cell identity cannot be aligned across individuals, so only statistical structure can be compared, never Kenyon cell to Kenyon cell: the comparison says that some aggregate property transfers, not that particular cells do. And the input-mass-preserving control keeps degree and weight heterogeneity together; the three were not separated, so the comparison cannot say which of them carries the effect.

A further caution follows from the second line of the results table. The input-mass-preserving control is one this project constructed, not a published benchmark, so the claim it supports — that the transferable structure is mostly glomerular input mass — is a claim about this comparison rather than a survey of what other datasets would show.

The two sexes are also not symmetric in the evidence. The female pair contributes FlyWire and hemibrain, whose synapse detection is weaker, and the "other hemisphere of the same fly" effect replicated on female teachers but not on male ones. Neither observation is large enough to carry a conclusion on its own, and the source does not attach one to either.

It is also worth naming what the design never varied. The teacher and the student always come from different individuals, but the task itself is the same teacher–student task used throughout the series, and the comparison is the same agreement-with-the-teacher measurement. A different task might separate the glomerular input totals from the Kenyon cell assignment in a way this one cannot, and nothing here speaks to such a task.

## Where the Chinese original lives

The Chinese original is [Three Flies with Real Wiring: Shared Structure, Mostly Glomerular Input Mass](/posts/2026-10-21-fly-connectome-08-individual-differences/). It carries the experiment records behind both the original and the replication runs — the stage 3 pre-registrations and notes, the replication comparison, and the script that builds the hemibrain input — together with the figure for the cross-individual comparison, the per-dataset alignment table, the dataset licence terms, and the pointers into the project repository at <https://github.com/meiluosi/fly-connectome-study> (directory `fly-v4/`).

Its main limit is the one stated above: the cross-individual advantage is close to glomerulus-level input mass, the numbers depend on the Kenyon cell count, and the null result is about task performance in a teacher–student comparison built for comparison, not about whether fly wiring is random. The next article in the series turns to the compartment split and the dopamine timing rule.

A reader who wants to reuse the alignment or the control should work from the Chinese original rather than from the summary above. The alignment rule, the Kenyon cell budgets per experiment, and the sampling procedure for the input-mass control are all specified there, and the third of those is the one that decides what a "matched" control means.
