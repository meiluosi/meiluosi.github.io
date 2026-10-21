---
sourceId: 2026-10-21-fly-connectome-09-compartments-and-dopamine
slug: fly-connectome-09-compartments-and-dopamine
title: "Compartments and Dopamine: Reproduced Timing Effects, No Measured Advantage for the Compartments"
editionLabel: ENGLISH OVERVIEW · RESEARCH NOTE
published: 2026-10-21T00:00:00.000Z
description: "Adding PAM and PPL1 compartments with a dopamine timing rule reproduces the timing effect, trails the base rule, and shows no foraging gain."
tags:
  - Fly Connectome
  - Dopamine
  - Mushroom Body
  - Negative Results
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 9
lang: en
---

> **Scope note.** This is an English overview of the ninth article in the Fly Connectome Research series, a Chinese research note. The source adds the mushroom body's reward/punishment compartment split and a dopamine timing rule to the learner, and reports the outcome: the timing effect reproduces, sample efficiency is no better than the base rule, the continual-learning difference is marginal and cannot be attributed to compartmentalisation, and foraging shows no advantage. It does not show that the compartments are useless, and the valence rule it builds on is a rough literature convention with documented exceptions.

## The model: two clusters and a suppression-only rule

The question the source starts from is whether the mushroom body's split of compartments into reward and punishment, together with the timing of dopamine, makes a learner better at continual learning and foraging. The earlier models had neither.

- **Clusters.** By the compartment in which the MBON sits, the PAM cluster (γ3, γ4, γ5, β1, β2, β′1, β′2, α1) carries reward dopamine and the PPL1 cluster (γ1, γ2, α′1, α′2, α′3, α2, α3) carries punishment dopamine. Of the 48 MBONs in the model, 20 fall in the PAM cluster and 28 in the PPL1 cluster, and the one calyx MBON does not learn.
- **Readout.** PAM-cluster MBONs drive avoidance; PPL1-cluster MBONs drive approach.
- **Rule.** Dopamine encodes a prediction error, and the synapses from active Kenyon cells onto that cluster's MBONs are suppressed by dopamine, with no potentiation term and no forgetting term:

$$
\delta_{\text{reward}}=\max(\text{target}-\text{prediction},\,0),\qquad
\delta_{\text{punishment}}=\max(\text{prediction}-\text{target},\,0).
$$

- **Controls.** The base rule (bidirectional and global); the true cluster labels randomly permuted among MBONs; and a permuted version in which the two clusters' total Kenyon cell to MBON synapse mass is held within ±10% of the true values.

The move the source makes is to let the compartment assignment carry the valence and to derive the learning signal from dopamine instead of from a fixed reward or punishment channel. Everything downstream inherits that assumption, and the source keeps it visible: the partition is checked against the data, the literature exceptions are listed, and no behavioural or activity data are used to confirm it.

The source also runs a data check on the partition itself: using direct DAN to MBON synapses in MaleCNS, each MBON was re-assigned to a cluster by whether its dominant DAN input is PAM or PPL1. That assignment agreed with the compartment-label partition 48 out of 48, with the calyx MBON the only difference. The check is weak by the source's own account — the DAN cluster labels come from type names, so the agreement is close to definitional, and it does not verify that PAM means reward.

The literature is treated with the same care. Aso et al. 2014 describe this rule with hedges such as "generally", and there are exceptions: PAM's MB-M3 and γ3 drive aversive memories under some conditions, PPL1-γ2α′1 takes part in the extinction of reward memories, and α′3 is associated with novelty. Transmitter and cluster direction broadly agree — 15 of the 20 PAM-cluster MBONs have inhibitory transmitters, and 19 of the 28 PPL1-cluster MBONs are cholinergic — but GABAergic MBONs occur in both clusters, so transmitter sign cannot be read directly as readout sign.

## The timing effect reproduces, but it does not discriminate

With a single odour and a dopamine pulse after the odour (forward) or before it (backward), the model behaves as follows:

- Reward forward gives a positive value and reward backward a negative one; punishment is the reverse.
- When the pulse sits 8 seconds away from the odour, the effect falls to about 3%.
- Ablating PAM-cluster plasticity destroys only reward learning, and ablating PPL1-cluster plasticity destroys only punishment learning.
- Removing the backward potentiation term drives the backward-pairing effect to zero.

All three cluster assignments — the true one, the permuted one, and the mass-balanced permutation — pass. So this is a self-consistency check of the rule, not a test that can distinguish the true structure. On the literature side, Handler et al. 2019 report that the timing of dopamine relative to the odour determines the sign of the memory, but that reversal requires an explicit temporal term: a plain delta rule on prediction error cannot produce it, and the rule used here carries that term.

## Sample efficiency does not improve

Across four learning tasks — real odours with random class labels, a linear teacher, and two real-wiring teachers — the compartment learner is on par with or slightly worse than the base rule.

| Task | Compartment learner vs base rule | Reading |
| --- | --- | --- |
| Real odours with random class labels | −0.006 | equivalent |
| Linear teacher | −0.018 | slightly worse |
| Real-wiring teachers | −0.027 to −0.065 | worse |

The compartment learner is clearly better than the cluster-label-permuted control, but independent review pointed out that the permuted control mainly differs in learning level: among permuted versions with a similar just-learned accuracy, the final accuracy showed no difference. The comparison therefore does not isolate the compartment labels from the learning level.

## Continual learning: a marginal difference that does not survive attribution

Each world has 40 odours with random ±1 labels, split into 4 tasks trained in sequence, and all 40 odours are tested at the end.

| Learner | Just-learned | Final |
| --- | --- | --- |
| Base rule, own tuning | 0.940 | 0.802 |
| Base rule, matched learning amount | 0.902 | 0.798 |
| Compartment rule, true clusters | 0.905 | 0.846 |
| Permuted cluster labels | 0.727 | 0.645 |

The compartment rule's final accuracy is +0.044 above the base rule and +0.048 above the matched-learning base rule, but the lower bounds of the 99.3% intervals are +0.024 and +0.029, both below the 0.03 minimum meaningful effect, so the difference is marginal at best. The matched-learning row is the fairer of the two comparisons, because it fixes the amount of learning instead of letting each rule find its own best tuning; the compartment rule matches it on just-learned accuracy but not on final accuracy. At the points where just-learned accuracy matches, the advantages are +0.028, +0.048, +0.041, and +0.004, and the reported +0.048 is the most favourable of them. An independent re-implementation returned +0.0395 and +0.0425, consistent with the original.

The review's conclusion is the part that constrains the claim: it cannot be said that compartmentalisation reduces forgetting. The difference comes from the rule family. The new rule only suppresses synapses and never potentiates, so old traces are not overwritten; and a version that suppresses every MBON for any error never learns on any task, so clustering is a precondition for a suppression-only rule to work at all. The two are confounded. The true partition is not special either: across 200 mass-balanced permutations, about 7.5% are no worse than the true partition, and among permutations with a close just-learned accuracy the final accuracy shows no advantage. What can be said is narrow: the true partition belongs to the class of partitions that can learn.

## Foraging: the advantage disappears once the update is bounded

The compartment learner was placed in the foraging field: 200 brand-new worlds, reward reversal at step 3000, in both consistent and conflict worlds.

1. **Narrow grid, learning-rate cap 0.8.** The compartment learner's net reward was only +4.9 and +13.4 against +51.9 and +36.2 for the base rule. All three compartment learners' best learning rates sat on the grid's upper bound.
2. **Relaxed grid.** In the consistent world the compartment learner reached +53.8 against +8.6 for the base rule; in the conflict world −9.3. Both are only marginal.
3. **Review finding.** The compartment learner's suppression has no upper bound. At learning rate 4.0 the synapse values land in [−6.9, +9.5], which is no longer suppression but sign-flipping amplification. In the control where suppression is clipped to [0, 1], the advantage in the consistent world disappears, roughly 8 against 54.
4. **Bounded re-run.** In the consistent world the compartment learner is 22.3 below the base rule; in the conflict world −2.6, indistinguishable. The true cluster partition and the permuted partitions are indistinguishable in both kinds of world.

There is therefore no evidence that the compartment learner beats the base rule in foraging. The stage 4 result is unaffected by this problem: comp_real's learning rates were all below 0.05 with a suppression step of no more than 0.1.

Three further limits bound everything above. PAM = reward and PPL1 = punishment is a commonly used rough rule with exceptions in the literature and no behavioural or activity data verifying it. The rule has suppression only, no contralateral potentiation and no forgetting term, and its temporal constants were chosen rather than fitted. And the claim that prediction error is encoded by dopamine is stronger than the evidence: the only two fly blocking experiments are negative, and Zhao et al. 2021 is a model fit on 28 data points, so it should be written as an error-driven updating hypothesis. The continual-learning task measures memory over random labels rather than generalisation, and the foraging field is a toy setting with substantial per-world reward noise.

## Where the Chinese original lives

The Chinese original is [Compartments and Dopamine: Putting Valence Specialisation into the Model, and What Came Out](/posts/2026-10-21-fly-connectome-09-compartments-and-dopamine/). It carries the experiment records behind the continual-learning and foraging stages — the stage 4 and stage 6 pre-registrations and notes, the two independent review reports, the replication comparison, and the literature review — together with the continual-learning frontier figure and the pointers into the project repository at <https://github.com/meiluosi/fly-connectome-study> (directory `fly-v4/`).

Its main limit is the one stated above: the timing effect reproduces, and the compartments bring no measured advantage — sample efficiency no better, a marginal continual-learning difference that cannot be attributed to compartmentalisation, and no foraging advantage once the update is bounded.
