---
sourceId: 2026-10-23-llm-training-rl-06-evaluation
slug: llm-training-rl-06-evaluation
title: "Interpreting RL Results: Sampling Gains, Reward Loopholes, and Generalization"
editionLabel: ENGLISH OVERVIEW · EXPERIMENT DESIGN
published: 2026-10-23T09:05:00+08:00
description: "Separate parameter updates from inference budgets, candidate coverage from selection, and proxy rewards from independent task success."
tags:
  - Reinforcement Learning
  - Evaluation
category: LLM Training Research
series: LLM Training Research Notes · Reasoning RL Volume
seriesOrder: 6
lang: en
---

> **Scope note.** Every evaluation table and probability below is constructed. This volume validates teaching calculations and checker code, not PPO or GRPO training gains. The controlled comparisons described here have not been run.

## Separate training and inference changes

Compare an unchanged and an updated model under both single-response and four-response settings, with the same selector in the latter. Track actual generated tokens and selection cost, since equal candidate counts need not mean equal inference budgets. The presence of a correct candidate does not guarantee that a practical selector delivers it.

A controlled parameter comparison fixes the model starting point, prompts, task split, output contract, checker, and decoding settings. If multiple components change, attribute the result to the combined intervention rather than one algorithm.

## Candidate coverage is not delivered accuracy

For N sampled responses containing C passing candidates, a standard pass@k estimate is one minus the ratio of all-failure k-subsets to all k-subsets, requiring N≥k. With N=10, C=2, and k=4, it is 2/3. This is distinct from assuming a known true success probability of 0.2, which gives 0.5904 for four independent attempts.

The [Codex evaluation paper](https://arxiv.org/html/2107.03374v2#S2.SS1) derives the estimator and explains the bias of substituting an empirical pass@1 into the power formula. Compute per task before averaging; do not pool candidates across different questions. Unbiased estimation does not make a single observed value the true probability.

## Count repairs and regressions

A constructed paired table has 50 tasks correct before and after, 20 repaired, 10 regressed, and 20 wrong in both. Accuracy changes from 60% to 70%, but ten regressions remain important. Separate training-seed variation from repeated decoding of one checkpoint, and preserve task or source grouping when evaluating uncertainty.

Reward can rise because a checker accepts nonminimal integers. An independent evaluator should add different evidence rather than repeat the same flawed predicate. [Reward-model overoptimization research](https://arxiv.org/abs/2210.10760v1) studies divergence between a proxy and a model-based gold standard in a synthetic setting; the gold model is not a universal substitute for human correctness.

## Entropy and generalization need context

Entropy falls from about 1.3863 for four equally likely candidates to 0.9404 for probabilities (0.7,0.1,0.1,0.1). Whether this is useful depends on which candidate dominates. Token entropy averaged over sampled prefixes is also a different quantity from this response-level illustration.

Report same-range held-out inputs, larger-number inputs, and prompt variations separately. Record formatting failures, numerical failures, invalid infrastructure measurements, and exact denominators. Reserve the test set from configuration and checkpoint selection.

## Report the evidence that was actually produced

Include generation, scoring, update, retry, and selection costs. Predefine stopping rules and preserve failed runs. A successful arithmetic check supports a statement about the calculation, not a claim of stronger reasoning from RL.

The [Chinese article](/posts/2026-10-23-llm-training-rl-06-evaluation/) closes the volume with a reporting protocol and questions that distinguish sampling gains, parameter gains, reward failures, and limited generalization. The next volume addresses tool use and multistep interactions.
