---
sourceId: 2026-10-20-llm-training-preference-06-score-vs-correctness
slug: llm-training-preference-06-score-vs-correctness
title: "Why a Higher Preference Score Does Not Necessarily Mean a Better Task Result"
editionLabel: ENGLISH OVERVIEW · ANALYSIS NOTE
published: 2026-10-20T00:00:00.000Z
description: "How length bias and reward over-optimization separate preference win-rate from independent correctness, using one constructed pair set."
tags:
  - Reward Over-Optimization
  - Length Bias
  - LLM-as-a-Judge
  - Independent Evaluation
category: LLM Training Research
series: LLM Training Research Notes · Preference Volume
seriesOrder: 6
lang: en
---

> **Scope note.** All pairs and checkpoint values are constructed teaching data. Arithmetic was checked in Python; no reward model, evaluator, or policy was trained or tested.

## Separate preference from correctness

In five retained teaching pairs, choosing the longer answer matches four preference labels. Length differences [12,13,7,−12,4] and correctness differences [1,1,1,1,0] have sample Pearson correlation about 0.044. This does not prove independence or that length is unrelated to correctness in other tasks.

In a separate four-pair construction, system A wins every comparison against B, yet only one A answer is factually correct. The longer-answer rule agrees with the judge on three pairs but selects a correct answer on none. These are different metrics, not interchangeable estimates of one capability.

The choices alone do not establish why the judge selected A. Length, position, content, or another factor could explain them; selecting the correct answer once does not reveal the judge's reasoning.

## Proxy optimization can diverge from the intended target

[Gao, Schulman, and Hilton](https://arxiv.org/abs/2210.10760v1) study a synthetic setup using a fixed gold reward model to label data for a proxy. The gold model is not infallible human utility, and the reported behavior is not a universal law of degradation.

The Chinese article's five constructed checkpoints have rising proxy rewards and preference wins but falling correctness. Correlations of about 0.987 between reward and wins, and −0.949 between reward and correctness, describe those points only. They do not identify length as a cause.

[MT-Bench](https://arxiv.org/html/2306.05685v4#S3) discusses judge limitations including position and verbosity. [Park et al.](https://arxiv.org/abs/2403.19159v2) study length exploitation in DPO. These sources motivate checks; they do not establish the cause of our invented table.

## Protect both data independence and metric meaning

Keep all candidates for a prompt and all prompts derived from one source in the same split. Final test data must not guide training or model selection. Changing annotation rules tests sensitivity to the criterion, not whether the data are independent.

Declare unknown-outcome treatment before evaluation. Report confirmed-correct share over all cases, conditional correctness among judged cases, and judgment coverage where useful. Do not silently discard hard cases or relabel unknown facts as false.

Length changes do not automatically invalidate win-rate comparisons. Report the distribution and the intended task, then add stratified or controlled analyses when needed.

## A better length-bias investigation

Length stratification is diagnostic, not causal proof. For a stronger comparison, create concise and expanded versions of the same answer, independently verify preserved facts and key information, randomize presentation order, and blind the evaluation against a fixed baseline. Report failed content-preservation checks and aggregate repeated variants by source prompt.

Even controlled rewriting may alter readability or information density. Limit the conclusion to the intervention actually tested, rather than claiming that any remaining difference proves a particular bias.

The [Chinese article](/posts/2026-10-20-llm-training-preference-06-score-vs-correctness/) contains the complete tables and reporting design. Preference improvement can be meaningful under its stated criterion without proving improved factual performance.
