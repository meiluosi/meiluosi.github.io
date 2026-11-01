---
sourceId: 2026-11-01-network-evolve-lab-negative-results
slug: network-evolve-lab-negative-results
title: "Three Pre-Specified Hypotheses, Zero Passes: Two Reusable Checks from an LLM–Social-Network Co-Evolution Simulation"
editionLabel: ENGLISH EDITION · SOCIAL SIMULATION
published: 2026-11-01T00:00:00.000+08:00
draft: false
description: "Results from NetworkEvolveLab: modularity saturates the bipolar ceiling at 0.49999997; asynchronous scheduling alone manufactures one-sided unfollowing (speaker-frequency bias ruled out); naive scaling to N=1000 collapses modularity to 0.077 as a pure density confound. Plus four retracted claims, a 2.5%-coverage LLM null, and the evidence boundary."
tags:
  - Social Simulation
  - LLM Agents
  - Negative Results
  - Polarization
  - Complex Networks
category: Social Simulation
lang: en
---

<!-- Mirrors the Chinese edition 2026-11-01-network-evolve-lab-negative-results. -->

> **Boundary first: every number here comes from my own simulator. None comes from real-platform stance data.** The only "real" network is the SNAP Higgs edge list, which carries no stance labels; its two camps are a proxy derived from the topology (Louvain communities). The code is currently in a private repository, so you cannot independently re-run these numbers yet. Please discount accordingly.

## The short version

I built a simulation where cognitive updates and unfollow/block decisions feed back on each other (NetworkEvolveLab) and wrote down three hypotheses first. **None passed**: H1 and H2 were falsified, H3 was not supported. What is worth keeping is not the three nulls themselves but two **reusable checks** that surfaced while producing them:

1. **Asynchronous scheduling manufactures the appearance of one-sided unfollowing**, even when every agent speaks with equal probability.
2. **Naively scaling a network confounds size with density.** Modularity falls from 0.45 to 0.077 at N=1000, which looks like the mechanism breaking, but mean degree simply grew tenfold.

## The model in one paragraph

N agents, each with a belief, a tolerance, and an archetype: 2% Hub, 8.5% Bridger, 89.5% Long-tail (Bridger base tolerance 0.7, others 0.4). Each round has N single-speaker events: a listener reads a statement, updates their belief, and **unilaterally** decides to reinforce, decay, or sever the directed edge. Severed edges go on a blocklist so triadic closure cannot restore them. The key metric $A_{\text{rupture}}$: among initially mutual dyads where at least one side severed, the fraction where **only one** side did.

## Finding 1: modularity hits the ceiling (H1)

Pre-specified H1: under unconstrained severing, modularity $Q$ jumps from <0.25 to >0.65 within 20 rounds.

- For any partition into $K$ communities, $Q \le 1-\sum_c a_c^2 \le 1-1/K$. So $K=2$ gives $Q\le0.5$, and $Q>0.65$ needs $K\ge3$. This is textbook; I do not present it as novel.
- 12 configurations, 20 seeds, T=100: initial $Q=0.213$, final $Q\in[0.174,0.487]$, **maximum 0.4999999655**, H1 pass rate 0%.
- In the Phase 5 grid, 70–90% of runs are $K=2$ and the rest $K=3$; nothing larger.

The maximum is **not "didn't jump high enough" but "hit the K=2 ceiling"**: at low tolerance the network splits into two disconnected camps of near-equal volume. The falsification is empirical (the dynamics produce only $K\le3$), not a theorem. I also checked all 42 (dataset × method) cells in the benchmark table: no violations of $Q\le1-1/K$. Caveat: a numerical "audit" of a theorem only shows the implementation is not buggy; it is not evidence.

## Finding 2: asynchronous scheduling creates one-sided rupture (RQ1)

I had initially attributed one-sided rupture to "multi-dimensional semantics". Controlling the variables showed otherwise. Full factorial: **schedule** (synchronous / async uniform speaker / async archetype-weighted speaker) × **cognition** (scalar / multi-d isotropic / multi-d anisotropic, i.e. agents weight issues differently). 20 seeds, N=60, 30 rounds; within a seed all arms share the initial graph and beliefs; mutual-edge weights are symmetrized.

| $A_{\text{rupture}}$ | Scalar d=1 | Multi-d isotropic | Multi-d anisotropic |
|---|---|---|---|
| Synchronous | 0.000 | 0.000 | **0.216** ± 0.051 |
| Async, uniform speaker | **0.053** ± 0.018 | **0.041** ± 0.022 | 0.219 ± 0.058 |
| Async, weighted speaker | 0.059 ± 0.021 | 0.060 ± 0.020 | 0.228 ± 0.062 |

![10-arm factorial attribution: left, A_rupture per arm; right, seed-paired decisive contrasts](/network-evolve/fig7_attribution_ablation.png)

(The figure also shows a reference arm with the original tolerance rule, A=0.134; omitted from the table.)

- **Under synchronous scheduling, scalar and isotropic multi-d models are exactly symmetric (A≡0 in 20/20 seeds).** This is an algebraic identity, not a finding: $|b_i-b_j|$ is symmetric and all edges are evaluated simultaneously.
- **Anisotropy produces genuine one-sided rupture**: +0.216 ($p=1.9\times10^{-6}$, 20/20 seeds positive), and a robust +0.178 under async uniform speaking.
- **Pure timing is enough**: async with **uniform** speaking gives +0.053 in the scalar model ($p=6.5\times10^{-4}$, 15/20 positive) and +0.041 in the isotropic multi-d model ($p=0.002$).
- **Speaker-frequency bias does nothing**: weighted vs uniform, +0.0065, $p=0.71$. I guessed wrong here: Hub/Bridger/Long-tail speaking weights are 10/3/0.5 in the code, and I suspected that was the driver.
- Not a T=30 truncation artefact: scalar uniform-speaker $A(t)$ is 0.53 (t=5), 0.087 (t=10), then flat at 0.053 from t≥15. The large early values come from very few severed dyads.
- The two sources are **not shown to be additive**: adding async on top of anisotropy gives almost nothing (+0.003). The interaction point estimate is −0.039, 95% CI [−0.088, +0.010], $p=0.14$. With n=20 I cannot detect it; "descriptively attenuated" is as far as I can go.

**Why this matters for LLM social simulation**: many systems use "one random speaker per step". If you report that agents "unfollow one-sidedly" in such a system, run a synchronous control first; otherwise you cannot tell whether it is agent cognition or the scheduler. I only demonstrated this on rule-based agents, not LLM agents.

## Finding 3: the "size effect" is a density effect

The engine's default initial graph is Erdős–Rényi with fixed $p=0.08$. As N grows, $|E|=O(N^2)$ and mean degree grows linearly, so scaling to N=1000 changes size and density together. Two arms: **density-matched** (mean in-degree fixed at 7.92, $p(N)=7.92/(N-1)$) and **fixed-$p$** (naive).

| N | Mean deg. (matched) | Q (matched) | Mean deg. (fixed-p) | Q (fixed-p) | Π (fixed-p) |
|---|---|---|---|---|---|
| 60 | 8.0 | 0.427 | 4.8 | 0.444 | 0.261 |
| 100 | 8.0 | 0.446 | 8.0 | 0.446 | 0.163 |
| 250 | 8.0 | 0.455 | 19.9 | 0.273 | 0.000 |
| 500 | 7.9 | 0.456 | 39.9 | 0.141 | 0.000 |
| 1000 | 7.9 | 0.453 | 79.9 | **0.077** | 0.000 |

10 seeds per cell, **except fixed-p at N=1000, which has only 3 seeds** ($O(N^2)$ edges make it slow).

![Left: paired LLM − rule differences; right: density-matched vs fixed-p scaling](/network-evolve/fig8_llm_arm_and_scaling.png)

- Under density matching $Q$ barely moves, but not exactly zero: the regression on $\log N$ has slope +0.0082 (95% CI [+0.0006, +0.0159]), under 0.03 total over a sixteen-fold range.
- Under fixed $p$, $Q$ collapses to 0.077 and polarization Π goes to 0, meaning everyone converges to consensus. **That is not "the mechanism failing"; it is a different parameter regime.** Someone scaling naively would wrongly conclude the mechanism breaks beyond a few hundred nodes.
- Real Higgs subgraph (N=1000, mean degree 14.9, not density-matched): $Q=0.489$, giant component 0.990.

This lesson then came back to bite me, as the next section shows.

## Claims I retracted

The part I trust most is that the work repeatedly overturned my own statements:

| Original claim | Why it was retracted |
|---|---|
| The critical tolerance $\theta^*$ is topology-invariant, in [0.50, 0.55] | Only 1 of 40 estimates (10 seeds × 4 topologies) falls in that band |
| "Then $\theta^*$ is topology-dependent" | Initial mean out-degree of the four topologies is 7.96 / 6.41 / 6.15 / 5.94 and reciprocity 0.08 / 0.75 / 0.75 / 1.0 (ER / WS / BA / EB order); $\theta^*_{\text{frag}}$ is 0.334 / 0.357 / 0.400 / 0.520, **strictly reversed** against degree (only 4 points: no causal claim, but enough to sink "topology-dependent"). Matching degree to 6.6 and reciprocity to 0.65 gives 0.374 / 0.329 / 0.360 / 0.417 (same order), CIs largely overlapping, spread down from 0.186 to 0.088. So only "the band [0.50, 0.55] fails" survives |
| "Bridgers are structural suspension cables that prevent fragmentation" | Next section: the advantage comes from the higher tolerance I gave them |
| "Asymmetric rupture comes from multi-dimensional semantics" | Finding 2: synchronous multi-d isotropic is exactly symmetric |
| Early draft: "$Q>0.65$ is mathematically impossible" | True only for $K=2$; for $K\ge3$ it is reachable |

One more wording fix: I first called the hypotheses "pre-registered", but there is no externally time-stamped registry and the plan document contains no numeric thresholds. They are now "pre-specified", and I no longer claim they preceded the simulations.

## Two more negative results

### H2: Bridgers are not more fragile

Pre-specified: Bridgers, squeezed by dissonance from both sides, retain at least 40% fewer ties than others ($p<0.01$). Define $\text{gap}=(\text{others}-\text{Bridger})/\text{others}$; H2 needs gap ≥ +0.40. 20 seeds, T=60, seed as the statistical unit.

| Bridger assignment | Tolerance | Bridger retention | Others | gap (95% CI) |
|---|---|---|---|---|
| By node index | Bridger 0.7 / others 0.4 | 0.664 | 0.511 | −0.299 [−0.359, −0.238] |
| By node index | Equalized | 0.514 | 0.497 | −0.034 [−0.092, +0.014] |
| By betweenness | Bridger 0.7 / others 0.4 | 0.678 | 0.531 | −0.277 [−0.316, −0.244] |
| By betweenness | Equalized | 0.490 | 0.499 | +0.019 [−0.023, +0.061] |

The gap is negative, so Bridgers are *more* stable. But once tolerance is equalized the advantage disappears (CI spans 0). The "resilience" was **written in by me**: Bridgers get a base tolerance 0.3 higher. "Bridger" is just an archetype label, not a network position. I added a betweenness-based control and the verdict is unchanged, but honestly: in this random graph the camps are set by beliefs, independent of topology, so high-betweenness nodes are not nodes that bridge opposing camps. **That control tests betweenness, not cross-camp brokerage.** The falsification stands; the "structural hole" mechanism was never actually tested.

### H3: platform interventions

Baseline $Q=0.442$ (N=100, T=60, 20 seeds). H3 requires a ≥35% gain in cross-camp reach **and** $\Delta Q\le-0.20$.

- 15% diversified recommendation: $\Delta Q=-0.038$ ($p=7.4\times10^{-8}$). Dose-response at 5/10/15/20/30%: −0.014/−0.029/−0.038/−0.055/−0.078. **Monotonic**, but nowhere near −0.20.
- Combined three-policy: $\Delta Q=-0.041$. Bridger subsidy and hub throttling alone do almost nothing (+0.002, +0.001).
- The reach limb is **statistically inconclusive**: combined governance +19.4% (paired one-sided $t$ $p=0.040$, Wilcoxon $p=0.105$), bootstrap 95% CI [−0.1%, +50.1%], which **includes 35%**. Reach is close to bimodal: 7 of 20 seeds have baseline reach <0.05. So the $\Delta Q$ limb alone is what sinks H3.
- An observation I did not pre-specify: diversified recommendation crushes **belief polarization Π** (0.123 → 0.068 → 0.030 at 0% / 15% / 30%) and edge count falls (1008 → 961 → 921), while modularity barely moves. Community structure is far more inert than beliefs. Exploratory only.

## The LLM part: a null at 2.5% coverage

The title says "LLM cognition", but this is my weakest evidence, so it gets its own section.

Setup: a real LLM (MiniMax-M3.1-Flash, temperature 0) is mounted in the flagship engine. Each round, 6 listener–speaker events go in one batched call; the model returns a belief update and keep / mute / unfollow / block, applied to the graph. 10 seeds × 30 rounds = 1800 requests, 1721 decisions applied, i.e. **2.52%** of 68,404 interactions. The rest follow the rule backend.

- Result: $\Delta Q=-0.0040$ ($p=0.42$, 95% CI [−0.015, +0.007]); the $A_{\text{rupture}}$ difference is exactly 0 in every seed.
- Actions: keep 1631, mute 61, unfollow 28, block 1. About 3 edges per seed (≈0.9%) were removed by the LLM.
- **Placebo**: delete the same number of random edges (~2.9 per seed) in the rule backend: $\Delta Q=+0.0018$ ($p=0.57$). LLM vs placebo: −0.0058 ($p=0.19$). **The LLM effect is indistinguishable from random severance.**
- **Per-event agreement**: LLM and rule agree on 96.3% of 1721 events (1657/1721), but that is swamped by the "mostly reinforce" base rate: Cohen's $\kappa\approx0.62$. On the 85 events where the rule would sever, the LLM agrees on only 37 (43.5%), and it severs less (29) than the rule would (53).
- $A_{\text{rupture}}$ is 0.089 ± 0.078 in the rule, LLM, and placebo arms alike: that endpoint is saturated and cannot count as evidence.

**What I can say**: in this engine, with this prompt, at 2.5% coverage, replacing the rule backend with an LLM has no measurable structural effect; this design cannot detect one. **What I cannot say**: that "LLM cognition doesn't matter". Answering that needs far higher coverage than this experiment had.

## Evidence boundary

- **Simulation only**: rule-based agents mostly, LLM on 2.5%. The five benchmark networks are generated proxies, **not** the datasets they are named after; the real Higgs subgraph's camps are topology-derived.
- **Small scale**: main experiments at N=60–100, only Phase 7 reaches N=1000; $\theta^*$ is measured only at N=100 with no finite-size scaling, so "phase transition" is at best a crossover.
- **One LLM, one prompt, temperature 0.**
- **n=20 has limited power**: failing to detect the interaction does not mean it is absent.
- I used several AIs (Gemini, DeepSeek, Claude) for implementation, auditing, and writing. Four rounds of "review" were done by Claude, which also helped iterate on the manuscript, so it was **not independent review**. It made the argument more rigorous; it is not peer review.

## If you keep two sentences

- When simulating agent societies, **run a synchronous control before reporting "one-sided unfollowing"**.
- When scaling up, **pin density first**, or the "size effect" you see may be a density effect.

As for the three nulls: they did not make the world more complicated. They showed me that many "phenomena" in this model were parameters I had written in myself (the Bridger tolerance, the scheduler, the mean degree). Once those are pinned down, what remains is worth trusting.

## Reproducibility

- Every number is generated from frozen result JSONs by script, with fixed seeds;
- a 451-entry SHA256 manifest covers results, benchmark graph caches, and the LLM call archive; I recomputed all of them and they match;
- LLM calls are archived by request-content hash and replay with zero new calls (300 used by the experiment, 323 in the archive; the other 23 are earlier exploratory calls);
- 46 unit tests pass.

The code is currently private; the above is my own check of my own environment.
