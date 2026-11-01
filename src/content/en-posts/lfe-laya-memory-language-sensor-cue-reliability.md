---
sourceId: 2026-11-01-lfe-laya-memory-language-sensor-cue-reliability
slug: lfe-laya-memory-language-sensor-cue-reliability
title: "Teaching Simulated Creatures to Remember an Old World: A Memory Structure, an LLM as Language Sensor, and Why Innate Trust Must Evolve Among Lies"
editionLabel: ENGLISH EDITION · ARTIFICIAL LIFE
published: 2026-11-01T00:00:00.000+08:00
draft: false
description: "Twenty-one preregistered simulations from LFE/Laya Life Lab: a table-per-environment memory lets individuals remember the old world (replicated on 16 fresh seeds and independent data); an LLM is useless as a decision-maker but the best cue sensor, until it errs; innate trust must evolve in a history with unreliable cues, and lifetime re-estimation did not help."
tags:
  - Artificial Life
  - LLM Agents
  - Continual Learning
  - Bayesian Inference
  - Preregistration
  - Negative Results
category: Artificial Life
lang: en
---

<!-- Mirrors the Chinese edition 2026-11-01-lfe-laya-memory-language-sensor-cue-reliability. -->

> **Boundary first: every number comes from my own simulator, none from real animals or environments.** The world has six foods whose effects keep reversing; the cue sentences were written by me or by the same LLM; the LLM judgements are precomputed and cached. The claim is "what holds and what does not in this kind of simulation", not "AI learned to adapt to the real world".

## The short version

Simulated foragers live in a world where the rules suddenly reverse (what was edible turns toxic) and later revert. I wrote the criteria first and ran 21 preregistered evaluations. **Three fairly solid findings and several honest negatives:**

1. **A memory structure lets them remember the old world.** An ordinary learner is *worse* when the old world returns than the first time it met it. Adding a structure that opens a new belief table on prediction error turns that negative number positive, and the result replicated on 16 fresh training seeds, on independently sourced data, and with twice as many foods.
2. **An LLM is a useless decision-maker but an excellent language sensor.** Letting it decide when to switch memory did not help. Asking whether two sentences describe the same kind of weather, and fusing the answer as a noisy sensor into Bayesian inference, gave the largest gain, until the sensor errs 30% of the time.
3. **Innate trust must evolve in a history where cues lie.** Individuals evolved where cues were always right trust them completely and collapse when cues mislead; those evolved in a mixed history keep some doubt and are robust everywhere. **Re-estimating reliability within a lifetime did not help**, contrary to my prediction.

## What was done (briefly)

- World: K foods, half nutritious, half toxic; the individual only reads a one-sentence description. Every other phase, all food effects reverse. A frozen pretrained decision model (Laya) gives a prior on "edible?"; individuals learn within a lifetime; **innate parameters (the genome) are evolved**; innate, acquired and Laya-perceptual sources are labelled separately.
- Key metric, **savings**: the share of nutritious foods eaten in the first two windows when the old world returns, minus the same share on first entering it. **Positive = remembers; negative = new experience overwrote the old knowledge.**
- Discipline: each comparison freezes its protocol and criteria first, then trains, hashes the genomes (SHA-256), then evaluates **once**; git commits are the timestamps. Deviations, interruptions and one naming accident are recorded.

## Finding 1: a table-per-environment memory remembers

![Savings: ordinary learner, a probabilistic comparator, environment-table memory](/lfe-laya/fig1_retention.png)

The environment-table memory (regime) keeps up to four belief tables and opens or switches on prediction error. On independently sourced mushroom data (UCI Mushroom, CC BY 4.0, labels from real records) with 16 fresh training seeds:

| Comparison (16 seeds) | Difference (preset threshold) | Same direction |
|---|---|---|
| A/B/A: regime savings − ordinary learner | **+0.250** (≥ +0.15) | 16/16 |
| Repeated reversals A/B/A/B/A | **+0.205** (≥ +0.15) | 16/16 |
| Partial reversal | **+0.141** (≥ +0.05) | 16/16 |
| Twice as many foods (K=12) | **+0.365** (≥ +0.10) | 16/16 |
| regime not worse than a probabilistic comparator (non-inferiority) | **+0.131** | 16/16 |

Two notes: **Laya's prior has no discriminating power on this data (AUC 0.505)** and the retention gain is still there, so the credit belongs to the memory structure, not to the language model; and the probabilistic comparator (a simplified online approximation of latent-cause inference) saved about zero or less, which only says something about *my approximation*, not about that family of methods.

## Finding 2: use the LLM as a sensor, not as the decision-maker

First I tried letting the LLM decide: on a trigger, hand it the individual's recent meals and each table's expectations and ask it to keep, switch or open a table. In two development rounds (2916 and 1216 calls) its savings were 0.05 to 0.20 *below* the same trigger with a rule action — no sign of benefit. Scrambling the outcomes hurt it, so it did read the experience; it just did not read it better than an ordinary learner.

Then a different question: **do these two sentences describe the same kind of weather?** On 18 held-out wordings the LLM passed all three preset gates (accuracy 1.0, pair AUC 1.0, AUC against unrelated sentences 1.0); Laya's hidden-state vectors (0.33) and question features (0.83/0.89) did not.

I precomputed the LLM's pairwise judgements (630 pairs of 36 cue sentences) and cached them, and used them as a **noisy sensor** in Bayesian context inference: when the cue sentence changes, each context gets a likelihood factor $f=r\,s+(1-r)(1-s)$ ($s$: pair similarity, $r$: sensor reliability, set by evolution). In this world the same state is worded differently every time it returns, so literal matching is useless.

![Language sensor: savings and fitness](/lfe-laya/fig2_language_sensor.png)

| Condition (mushroom data, 16 seeds, A/B/A) | Savings | Fitness |
|---|---|---|
| ordinary learner | −0.11 | 0.73 |
| environment-table memory | +0.11 | 0.81 |
| literal cue-keyed memory | +0.07 | 0.90 |
| Bayesian inference, sensor off | −0.11 | 0.73 |
| **Bayesian inference + LLM sensor** | **+0.25** | **0.90** |
| oracle sensor (true state; upper bound only) | +0.25 | 0.90 |
| shuffled sensor | −0.09 | 0.71 |
| **30% of judgements inverted** | **+0.02** | **0.74** |

How to read it: (1) **with the sensor off it is exactly an ordinary learner**; all the gain is language evidence; (2) **a shuffled sensor degrades it sharply**, so "any cue change helps" is not the explanation; (3) the literal cue-keyed memory also has high fitness because every new wording opens a new table — it relearns each phase and **retains nothing**; (4) eight of nine hypotheses held; **the one that failed is "with 30% of judgements inverted it is still no worse than not using cues"**: +0.008 against a +0.02 threshold.

## Finding 3: innate trust must evolve among lies

What if the sensor errs? Biology's answer is presumably that evolutionary history contained unreliable cues, so doubt is innate. I tested that analogy in a dense-alternation world (effects reverse every 300 ticks) with two cue channels, each independently in one of: reliable, 30% misleading, uninformative, or absent — unknown to the individual — and compared evolutionary histories:

![Innate trust: clean history versus mixed history](/lfe-laya/fig3_reliability.png)

- Individuals evolved where cues were always right evolve an innate trust of **0.99**; when cues mislead (30%) fitness is only **0.68** (0.91 for the mixed-history individual under the same condition), and **with no cue at all it falls to 0.26 — worse than ignoring cues altogether (0.40)**.
- Individuals evolved in a mixed history lower their trust to **0.91**, score 0.91 under misleading cues, still 0.82 with no cue, and pay only **0.009** when the cue is reliable; it also holds for error rates never seen in training (15%, 45%; +0.528).
- **A counter-example to my own prediction**: I expected within-lifetime re-estimation of reliability to help when reliability varies across worlds. It was on average slightly *worse* than innate trust alone (−0.031, 7 of 8 seeds negative). Possible reasons are untested: numeric evidence already corrects a misleading cue in the posterior; my consistency check is noisy. **This holds only for the one update rule I implemented.**
- A second independent cue channel helped (+0.047, 7 of 8 seeds positive) but **missed the preregistered +0.05 by 0.003**, and is recorded as not met.

## What did not work

- **The advantage did not carry over to reproducing populations**: with reproduction and inherited innate types, the memory structure gave no detectable advantage in persistence, size or recovery (establishment +0.016). A heritable-mutation probe found no selection direction.
- **No evidence that Laya understands environmental change**: two probes missed their preset gates.
- **An early large effect did not replicate** across training seeds.
- The prior's value is conditional: +0.049 / +0.073 at delay 40 / 80 (8/8), about zero under immediate feedback, and zero in a world simple enough for experience to suffice.

## Three sentences to take away

- To make a creature remember an old environment, **give it a memory that can "open another page"**; that works better than a stronger prior.
- **Let an LLM do what it is good at (judging whether two sentences mean the same) and not what it is bad at (sequential judgement on a few noisy samples)**, and treat its output as a noisy sensor, not as a command.
- **Innate doubt has to evolve among lies**; re-estimating reliability within a lifetime did not help in my setup.

## Limitations

- The world is artificial: six foods, two alternating states; the mushroom descriptions are a fixed template over real record attributes, not natural text.
- Cue sentences were written by me or by the same LLM; the sensor is one LLM's cached judgement (no other models or prompts tried); "two independent channels" means independently sampled faults, not independent information sources.
- No comparison with neural continual-learning methods such as EWC. Here beliefs are per-description tables without shared weights, so "protecting important weights" corresponds to lowering forgetting for well-evidenced entries — an argument, not an experiment.
- 8–16 seeds; Holm's correction is severe with 8; several explanations are hypotheses, not tested.
- Evaluations ran with heavy machine load, which does not affect the results; one naming accident (I accidentally overwrote an older learner's source file) was restored from version control before any evaluation used it and changed no hypothesis.

## Reproducibility and disclosure

- Code (Apache-2.0), data and docs (CC BY 4.0): [github.com/meiluosi/lfe-laya](https://github.com/meiluosi/lfe-laya). Each preregistration's protocol, frozen commit, genome hashes, raw results and evaluation script are under `docs/PREREGISTRATION_*.md` and `experiments/`; the three key ones are [20](https://github.com/meiluosi/lfe-laya/blob/main/docs/PREREGISTRATION_20.md), [21](https://github.com/meiluosi/lfe-laya/blob/main/docs/PREREGISTRATION_21.md) and [22](https://github.com/meiluosi/lfe-laya/blob/main/docs/PREREGISTRATION_22.md). A paper draft is in `paper/`; **it has not been uploaded to arXiv**.
- Model weights are not redistributed. The LLM sensor uses cached judgements from DeepSeek's `deepseek-flash` (thinking off); evaluation makes no new LLM calls, every call is archived by content hash and strictly replayable; LLM calls had no budget cap (about ten thousand distinct calls in total).
- **A large part of the code, experiments and documents was produced by an AI coding assistant (Claude) under my direction**: I set the questions, made the trade-offs and reviewed protocols and results; every number traces back to a raw result file in the repository. References were checked at the metadata level; specific claims were checked against abstracts only.
