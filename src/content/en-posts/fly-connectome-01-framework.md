---
sourceId: 2026-10-21-fly-connectome-01-circuit-and-framework
slug: fly-connectome-01-framework
title: "A Real Fly Mushroom Body Circuit as a Fixed Learning Skeleton"
editionLabel: ENGLISH OVERVIEW · RESEARCH NOTE
published: 2026-10-21T00:00:00.000Z
description: "How a real fruit fly mushroom body circuit becomes a fixed learning skeleton, and how five ablation arms isolate wiring from rule."
tags:
  - Connectomics
  - Mushroom Body
  - Ablation Study
  - Null Result
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 1
lang: en
---

> **Scope note.** This is an English overview of the first Chinese article of the Fly Connectome Research series. The wiring is real — the Janelia MaleCNS v1.0 connectome, CC BY 4.0 — while the learning rule built on top of it is a hand-written abstraction, not a biophysical model of the fly. The article reports the properties of the extracted matrices and the fairness checks every later arm must survive, but it reports no head-to-head outcome of its own. The comparison holds the rule, the task, the budget, and the tuning procedure fixed and varies only the wiring. It does not show that the mushroom body wiring beats any control, and it refuses the opposite claim too: a null reading here is a statement about one abstract setting, not about the biology.

## From connectome to three matrices

The source is the Janelia MaleCNS v1.0 connectome: an electron-microscopy reconstruction of one male fruit fly brain, recording each neuron's type and the number of synaptic contacts between each pair of neurons (CC BY 4.0). Only the right hemisphere's mushroom body is used.

From the raw "who connects to whom, and with how many synapses" table, three steps produce the model's inputs:

1. Select the mushroom body neurons by cell type — projection neurons (PN), Kenyon cells (KC), and mushroom body output neurons (MBON).
2. Merge the PNs by the olfactory glomerulus they belong to, yielding a glomerulus-to-KC matrix.
3. Give each MBON a sign from its predicted neurotransmitter: acetylcholine excitatory, GABA and glutamate inhibitory.

What survives is three small objects:

| Object | Shape | Reported values |
| --- | --- | --- |
| Glomerulus to KC weights `P` | 65 × 2045 | 10,800 non-zero connections; about 5.3 glomeruli per KC on average; up to 89 synapses in one connection |
| KC to MBON weights `N` | 2045 × 49 | 24,091 non-zero connections; about 11.8 MBON per KC on average |
| MBON signs | 49 | 25 predicted excitatory, 24 predicted inhibitory |

One property of the real wiring shapes everything that follows: glomerulus fan-out is very uneven. The glomerulus reaching the fewest Kenyon cells reaches 2, the median reaches 164, and the most reaches 473. The article returns to this when it explains why "real is about equal to shuffled" cannot be read as "the fan-out structure does not matter".

## The learning rule, and which parts are hand-written

The rule is an abstract rate model, not a biophysical one. An odor is a 65-dimensional activation vector `u` over glomeruli, and each Kenyon cell's input is

$$
\psi = u P, \qquad \mathrm{code} = \text{top } 5\% \text{ of } \psi
$$

so only the strongest 5% of Kenyon cells are active. That approximates the sparse, high-dimensional code the real mushroom body uses, where a gating inhibitory neuron (APL) keeps the active set small.

Each KC-to-MBON synapse carries a strength `q` bounded in [0, 1] and initialized at 0.5. The value read out is the signed weighted sum over MBONs, with the initial value subtracted and the result normalized to [-1, 1]:

$$
V = \frac{\sum_j s_j w_j q_{ij} - V_0}{\mathrm{scale}}, \qquad s_j \in \{+1, -1\}
$$

The normalization is what lets a foraging turn rule and a traveling-salesman weight of the form weight times exp(κ times value) share one scale.

Learning is driven by a prediction error, not by the reward itself:

$$
\delta = r - V, \qquad q_{ij} \leftarrow q_{ij} + \eta |\delta| (t - q_{ij}), \qquad t \in \{0, 1\}
$$

A positive `delta` strengthens synapses onto excitatory MBONs and weakens synapses onto inhibitory ones; a negative `delta` does the reverse. The source notes that this is the same move earlier work made when writing the mushroom body as a gambling-machine learner (Bennett et al., 2021), adopted here so the rule stays comparable to published work.

The article separates what is real from what is assumed:

| Real, taken from the connectome | Hand-written assumption |
| --- | --- |
| Glomerulus to KC wiring and synapse counts | How odors are produced and how the reward schedule runs |
| Only the strongest 5% of KCs kept (the ratio is biological; taking the top 5% is the method) | The form of the learning rule |
| KC to MBON wiring and neurotransmitter signs | How value becomes an action |
| MBON types and compartments | The ant colony and every baseline's rule |

Two decisions are worth separating from the rest because they are choices, not findings. The spiking (LIF) engine was tried first and never produced an encoding that was simultaneously sparse, broadly covering, and odor-specific, so it was dropped at a pre-set stopping line in favor of the rate model, which is simpler and easier to control. And the step that turns value into behavior has no biological basis at all; it is the minimal interface that lets value influence action.

## Five ablation arms, one variable at a time

The question the series asks is whether the real wiring has anything special about it. Answering it by control requires the learning rule, the task, the budget, and the tuning procedure to be identical across arms, with only the glomerulus-to-KC connections replaced.

| Arm | Construction | Question it answers |
| --- | --- | --- |
| Real | The original connections | Baseline |
| Shuffled | Per KC, randomly permute which glomeruli it receives, keeping the input count and the synapse counts | Does the specific topology matter? |
| Random sparse | Each KC picks 6 random glomeruli, all synapse counts set to 1 | Is sparse expansion alone enough? |
| Real, binarized | Keep the real topology, set every synapse count to 1 | Is it the topology or the input-strength differences that matter? |
| Linear learner | No sparse expansion; learn a linear value directly on the 65-dimensional glomerulus vector | Does sparse expansion itself help? |
| No learning | Value constantly 0 | Sanity check: does learning do anything at all? |

The source states the reading rules in advance, and they are asymmetric on purpose:

- Real about equal to shuffled: what matters is not which glomeruli a cell receives.
- Real about equal to random sparse: sparse expansion is enough, and the real wiring adds nothing.
- Real better than random sparse: this is the case that would be evidence of real wiring specificity.
- Real about equal to linear: sparse expansion brings no benefit.

## Four fairness traps

A comparison that produces a null is only informative if the arms are genuinely comparable. The article names four ways this framework could have produced a null, or a difference, for reasons unrelated to the wiring.

**Tie-breaking at the cutoff.** Under random-sparse wiring a KC's input score is an integer between 0 and 6, so about 152 KCs tie at the cutoff, against about 10 under the real wiring. `argpartition` picks among ties by implementation detail, so if it favored certain indices the random arm would gain an unearned advantage. The implicitly selected mean index was essentially uniform, showing no visible bias; the arm was nevertheless changed to break ties explicitly with a tiny random jitter, every run was repeated, and the result did not change.

**Learning-rate saturation.** Reusing the foraging learning rate in the traveling-salesman setting saturated every variant's value near +0.95, leaving no discrimination between edges. Learning rate and gain have to be tuned per learner and per task, each on its own tuning set with the same grid.

**Controls and baselines have to be tested first.** In the continual-learning audit, the paper's dense control collapsed to chance level because of weight saturation, which made the sparse advantage look large; replacing it with the same classifier shrank the advantage to within about 3 percentage points.

**Features that duplicate a prior.** In the traveling-salesman task the edge features correlated with distance at 0.85, so the learner was effectively "distance plus a little smoothing bias", and the arms were equivalent by construction.

## What this framework cannot measure

The shuffled arm preserves each KC's input count and synapse counts but destroys the uneven glomerulus fan-out structure, from 2 KCs at the low end to 473 at the high end. Real about equal to shuffled therefore does not mean the fan-out structure is unimportant.

Which feature attaches to which glomerulus is assigned arbitrarily, so whether a feature lands on a hub glomerulus or a rare one is partly random. Testing wiring specificity is hard by construction in that setting.

The circuit is a simplification of the mushroom body: there are no valence-specialized compartments, no temporal rules for dopamine, and no feedback or recurrence.

The article's own conclusion is therefore deliberately narrow: in this abstract setting, no advantage of the real wiring was measured. That is not the statement that real wiring is useless.

## Where the Chinese original lives

The Chinese original is [How Can a Real Fly Brain Circuit Be Compared Fairly With Other Approaches?](/posts/2026-10-21-fly-connectome-01-circuit-and-framework/). It holds the diagram of the four-step circuit, the extraction table with the MaleCNS download script, the full design-rationale table, the per-arm construction table, the four fairness checks with their recorded numbers, and the pointers to the code. Its limit is the one stated above: the series measures whether real wiring beats its own controls inside one abstraction, and the framework cannot see advantages that would require valence compartments, dopamine timing, or feedback.

The next note in this series covers [the protocol every experiment in the series shares](/en/posts/fly-connectome-02-preregistration/).
