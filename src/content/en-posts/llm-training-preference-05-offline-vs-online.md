---
sourceId: 2026-10-20-llm-training-preference-05-offline-vs-online
slug: llm-training-preference-05-offline-vs-online
title: "Offline Preference Optimization versus Online Sampling: The Fixed Answer Pool and Regeneration"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-20T00:00:00.000Z
description: "How a fixed preference pool and on-policy sampling differ in feedback coverage, distribution drift, sampling budget, and verification cost."
tags:
  - Online Preference Optimization
  - Off-Policy Sampling
  - Distribution Shift
  - Sampling Budget
category: LLM Training Research
series: LLM Training Research Notes · Preference Volume
seriesOrder: 5
lang: en
---

> **Scope note.** A four-category softmax example, finite differences, sampling probabilities, and event counts were executed. No language model was trained or timed.

## Separate feedback collection from the update rule

Offline training reuses fixed comparisons; online training collects new feedback, sometimes in batches that are reused. [OAIF](https://arxiv.org/html/2402.04792v2#S3) samples two current-model responses and asks an annotator to choose. [TRL 0.23.1](https://huggingface.co/docs/trl/v0.23.1/en/online_dpo_trainer) supports a judge or reward model.

Online does not imply PPO. Old-policy ratios and clipping belong to specific update methods, not every online preference procedure. Historical data need not move monotonically farther from the current distribution; asynchronous online collection can also use stale samples.

## Uncompared answers can change probability

Consider independent logits with probabilities [0.05,0.15,0.30,0.50], a uniform reference, and two preferences: y1 over y3 and y1 over y4. Sum the two DPO losses at beta=0.1.

The negative logit gradient is approximately [0.1102,0,−0.0545,−0.0557]. These are logit changes per unit learning rate, not log-probability changes. The latter subtract the probability-weighted mean logit change, approximately −0.0387. Thus y2's log probability increases to first order even though its own logit is unchanged. A small numerical update verifies this.

A fixed pool provides no direct quality comparison for y2, but normalization and shared neural parameters can still change unseen answers. Offline generalization is possible; online feedback adds evidence about newly sampled behavior.

## Sampling counts are not gradient multipliers

With the distribution fixed, 200 independent samples have expected counts 10,30,60,100. Seeing y1 at least once has probability 1−0.95^N: about 64.2% at N=20 and 99.9965% at N=200.

A two-sample pair contains exactly one y1 with probability 0.095. Whether that pair gives useful feedback depends on the other answer and the evaluator. No universal twentyfold gradient reduction follows from the 5% marginal probability.

## Count comparable units and budgets

For 100 prompts with one pair each, each online round generates 200 answers and requires 100 pairwise judgments, absent retries. Fifty rounds yield 10000 answers and 5000 judgments. A toy offline pool of 100 pairs requires only the initial 200 answers and 100 judgments, but both methods still perform training computation.

These are events, not forward-pass or API-call counts. Per-answer scoring, batched requests, repeated judging, and generated token lengths change cost. New online responses need their own reference scores.

Equal training-pair counts isolate a different question from equal total cost. Report collection, scoring, training, and failures. More distinct outputs plus higher scores alone cannot establish exploration as the cause.

## Keep feedback definitions explicit

The evaluator supplies feedback; it is not the generation policy. Online optimization can reinforce evaluator errors. Original outcome-supervised [GRPO](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS2) standardizes group rewards rather than retaining only their ranks: [0,1,2] and [0,1,100] have the same ranking but different advantages.

The [Chinese article](/posts/2026-10-20-llm-training-preference-05-offline-vs-online/) gives the worked derivation, cost ledger, and controlled-comparison design. No result here establishes that online or offline preference optimization is superior.
