---
sourceId: 2026-10-21-fly-connectome-03-foraging-and-swarm
slug: fly-connectome-03-foraging-and-swarm
title: "The Fly Learner Does Learn, and Random Wiring Does Better"
editionLabel: ENGLISH OVERVIEW · RESEARCH NOTE
published: 2026-10-21T00:00:00.000Z
description: "In a foraging arena the mushroom body learner learns across reward reversal, while random sparse wiring outperforms the real wiring in both worlds."
tags:
  - Mushroom Body
  - Foraging
  - Swarm Behavior
  - Null Result
category: Fly Connectome
series: Fly Connectome Research
seriesOrder: 3
lang: en
---

> **Scope note.** This is an English overview of the third Chinese article of the Fly Connectome Research series. It reports two pre-registered experiments: a single fly foraging in an arena whose rewards invert halfway, and a swarm of 20 flies sharing one world under four communication modes, with an ant colony as a side-by-side demonstration. The numbers below come from 200 new worlds for the single fly and 100 for the swarm. The comparison holds the learning rule, the world, and the tuning procedure fixed and varies only the wiring. It establishes that the learner learns and that the real wiring was not better than random sparse wiring in this model and task. It does not establish that the real mushroom body is suboptimal: valence compartments, dopamine timing, and realistic odor statistics are not modeled here.

## A world built to punish rote memorization

The arena is a 100 × 100 plane holding 6 food sources in two odor families: three in family A (A, A′, A″) and three in family B. The variants and the prototype share 4 of the 6 olfactory glomeruli, so they "smell alike". Odor decays exponentially with distance, and the fly compares concentrations with its left and right antennae before turning.

The rewards are deliberately unstable: good food pays +1, bad food pays -1, and at step 3000 of 6000 every family's reward flips at once, so what was good becomes harmful and the other way round.

Two worlds sit on top of that:

- **Similar odors, same reward.** The A family shares a sign and B is opposite. Learning that A is good should generalize to A′, so generalization pays.
- **Similar odors, opposite rewards.** A and A′ have opposite signs. Appearance alone leads to learning the wrong thing, so generalization hurts.

The pair is there to separate generalizing from memorizing.

## Three participants and four controls, with two accounting cautions

The fly uses the circuit from the first note: odor activates glomeruli, the real glomerulus-to-KC wiring produces a sparse code, KC-to-MBON synapses read out a value, and a prediction error updates those synapses after eating. How value becomes action is hand-written: the strength of the turn is proportional to an innate attraction plus the learned value.

The ants do not smell. An ant walks on a grid, finds food, carries it back to the nest, and leaves pheromone on the way; later ants favor pheromone-dense paths. Harmful food costs points and sends the ant back to the nest.

The controls are shuffled wiring, random sparse wiring, a linear learner, and a fly that does not learn at all.

Two cautions apply to every number that follows.

1. **The accounting is not symmetric.** A fly scores on eating; an ant scores only after carrying food back to the nest, which works against the ants, and the ant rules are hand-written as well. The ant colony is a parallel demonstration, not a strict opponent.
2. **The innate component earns no points.** The fly's turn has a hand-written component that pulls it toward odor. The fly that does not learn ends with a net reward of only +0.5, so the score has to come from learning.

All parameters were chosen on separate tuning worlds, and the test uses brand-new worlds.

## Single-fly results across 200 pre-registered worlds

Average net reward over 6000 steps in 200 new worlds:

| Brain | Similar odors, same reward | Similar odors, opposite rewards |
| --- | --- | --- |
| Real wiring | +44.0 | +35.9 |
| Shuffled wiring | +44.4 | +32.4 |
| Random sparse wiring | +50.9 | +44.1 |
| Linear learner | +46.2 | +31.5 |
| No learning | +0.5 | +0.9 |

Four readings follow, and they pull in different directions.

- **It learns.** The real wiring earns 35 to 44 more than no learning, and it relearns after the mid-run inversion of every reward.
- **Real is about equal to shuffled.** Randomizing which glomeruli each KC receives leaves the score almost unchanged: -0.4, interval [-3.6, +2.8].
- **Random sparse is better.** It is ahead by 6.9 to 8.3 in both worlds, with both intervals on the positive side of 0.
- **The linear learner is not bad either.** It matches the mushroom body in the same-reward world and is slightly worse in the opposite-reward world, where the real wiring is ahead by +4.3.

Sparse expansion therefore did not give the mushroom body a clear advantage in the world where generalization pays; it was only somewhat steadier in the world where generalization hurts, and by a small margin.

### Ruling out an implementation artifact

Under random sparse wiring a KC's input score can only be an integer from 0 to 6, so about 152 KCs tie at the cutoff, against about 10 under the real wiring. `argpartition` breaks ties by implementation detail, so if it favored certain indices the random arm would gain an unfair advantage. The selected indices averaged around 990 with the first half occupying 51%, showing no visible bias. The arm was still changed to an explicit random tie-break with a tiny jitter, everything was re-run, and random sparse remained better.

A second candidate explanation was that the real wiring has uneven input strengths. The control that keeps the real topology but sets every synapse count to 1 came out slightly better than the real wiring: +3.3, with intervals [+0.1, +6.4] and [-0.1, +6.7]. That is uncertain, and the source records it only as an unconfirmed lead.

## The swarm: four communication modes and the cost of crowding

Real flies do not talk to each other. The four modes below are hand-designed hypotheses about how much sharing is worth, not descriptions of fly behavior.

Next, 20 flies forage in one world at the same time. Each has its own mushroom body and learns separately, but they eat the same food: a source that has been eaten pays nothing for 30 steps, so they compete. The modes differ only in whether another fly's meal affects me.

| Mode | In plain terms | Mechanism | Real fly behavior? |
| --- | --- | --- | --- |
| Independent | 20 strangers, each exploring alone | Each fly learns only from its own food | Yes, closest to reality |
| Cluster | Follow the crowd, exchange nothing | When choosing a direction a fly is also drawn toward where peers are; movement only, no learning | Hand-written hypothesis |
| Observational learning | Seeing a neighbor eat well, learn a little of it | When one fly eats, flies within 20 distance units update their mushroom body at half their own learning strength, as if they had eaten too | Hand-written hypothesis |
| Broadcast | 20 people sharing one notebook | When any fly eats, all 20 learn at half strength regardless of distance | Not a biological practice; the upper bound on shared information |

Results over 100 new worlds in the similar-odors-same-reward world, total net reward for the 20 flies:

| Group | Total | Per fly |
| --- | --- | --- |
| Independent | +274.2 | +13.7 |
| Cluster (hand-written) | +211.0 | +10.5 |
| Observational learning | +353.5 | +17.7 |
| Broadcast | +411.6 | +20.6 |
| Ant colony | +187.9 | +9.4 |
| Single fly | +42.9 | +42.9 |

- **More shared information is better.** Observational learning adds +79.3 over independent, and broadcast adds another +58.1 over observational learning. The source's reading, which was not tested separately, is that food sources are few, a fly takes a long time to find one on its own, and after the reward inversion everyone has to relearn; if a neighbor's experience counts as your own, relearning is much faster.
- **Clustering is harmful, -63.2.** It gives no information at all, so it only makes everyone crowd around the same source and compete, while a depleted source pays nothing for 30 steps.
- **The independent swarm beats the ant colony, +86.3, interval [+78.0, +94.3].** Because ants are disadvantaged in the accounting, this should not be read as "flies beat ants".
- **Crowding is a real cost.** Under the independent mode a fly earns 41.7 on average; with 5 flies it is 32.9 each; with 20 flies 13.8 each; with 40 flies 6.7 each, and the total even falls from 275.6 to 267.6. More agents do not produce a proportional gain in total.

## Credibility and the boundary of the conclusion

- **Where the numbers come from.** The page's measured data come from the pre-registered 200 new worlds, and 100 for the swarm. The web simulation is a line-by-line port of the Python version that passes 37 checks in Node and is statistically consistent with the Python 200 worlds (real wiring 44.1 against 44.0, no learning 0.6 against 0.5). The two sides use different random numbers, so only the statistics are compared.
- **Exploratory numbers have to be re-run.** In the earliest 20 worlds a hint appeared that the real wiring was slightly stronger. It vanished on a re-run: within the same batch of worlds, net reward fluctuates by about 5 because the agent's internal random stream differs.
- **The boundary of the conclusion.** "Random sparse is better" says only that in this model and this task, this real wiring encoding has no advantage. It cannot say the real mushroom body is suboptimal, because this model has no valence-specialized compartments, no dopamine timing, and no realistic odor statistics.

## Where the Chinese original lives

The Chinese original is [It Learns, but Random Wiring Does Better: Foraging with Flies and Ants](/posts/2026-10-21-fly-connectome-03-foraging-and-swarm/). It holds the world and swarm diagrams, the full result tables with both worlds, the implementation-artifact check with the tie counts and index averages, the crowding figures at 1, 5, 20, and 40 flies, the interactive page, and the pointers to the result files. Its limit is the one stated above: the comparison varies the wiring inside one abstraction, and the abstraction omits the compartments, dopamine timing, and odor statistics that a real mushroom body would need.

The next note in this series covers [the continual-learning audit](/en/posts/fly-connectome-04-continual-learning/).
