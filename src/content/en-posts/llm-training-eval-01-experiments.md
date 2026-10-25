---
sourceId: 2026-10-23-llm-training-eval-01-experiments
slug: llm-training-eval-01-experiments
title: "Training Comparisons That Answer a Question: Hypotheses, Baselines, and Repeats"
editionLabel: ENGLISH OVERVIEW · EXPERIMENT DESIGN
published: 2026-10-23T11:00:00+08:00
description: "Define interventions, budgets, paired repeats, failure rules, and the limits of training conclusions."
tags:
  - Experiment Design
  - Reproducibility
category: LLM Training Research
series: LLM Training Research Notes · Evaluation and Reproducibility Volume
seriesOrder: 1
lang: en
---

> **Scope note.** All scores and thresholds are constructed. Only arithmetic checks were run; no training comparison was executed.

## Make the hypothesis falsifiable

Ask whether verified recovery examples improve recovery without unacceptable first-attempt regressions. Define the intervention as replacing some clean supervision under a fixed target-token budget. Specify the primary metric, guardrails, and decision thresholds before viewing test outcomes.

Equal target tokens do not imply equal input tokens, attention work, or elapsed time. Separate fixed-budget substitution from adding data and compute. Keep both an unchanged starting model and a clean-data training baseline.

## Check interactions rather than attributing everything to one change

A constructed two-factor table has scores 60 and 65 without recovery training, versus 62 and 74 with it, across no-feedback and feedback inference conditions. Training effects are 2 and 9 percentage points; their difference is 7. The full 14-point change cannot be attributed solely to training.

[NIST's randomized-block guidance](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm) explains controlling nuisance variation within blocks. Paired training repeats can adopt that principle, while recognizing that an identical seed number does not guarantee identical random events across implementations.

## Preserve repeated runs and their units

Constructed paired scores A=(60,70,65), B=(64,72,66) yield differences (4,2,1), averaging about 2.33 percentage points. Comparing the best B with the worst A produces 12 points and misrepresents the average comparison. Three repeats are an illustration, not a universal sample-size recommendation.

Repeated decoding of one checkpoint does not measure training-seed variation. Preserve task-family and run grouping when estimating uncertainty. Record out-of-memory failures, divergence, and invalid infrastructure measurements separately under predefined rules; do not silently report only successful runs.

## Keep the evaluation population comparable

A constructed system A succeeds on 81/90 easy and 2/10 hard tasks, totaling 83%. B succeeds on 88/99 easy and 0/1 hard tasks, totaling 88%. B is worse within each category but benefits from a different mixture. Use shared instances or fixed declared weights rather than attributing this aggregate change to training.

An experiment card should specify the question, intervention, baseline, budgets, metrics, versions, repeats, failures, stopping, and checkpoint selection. The [Chinese article](/posts/2026-10-23-llm-training-eval-01-experiments/) explains how outcomes should guide the next decision without claiming unrun gains.
