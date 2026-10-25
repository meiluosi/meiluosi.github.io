---
sourceId: 2026-10-23-llm-training-rl-01-formulation
slug: llm-training-rl-01-formulation
title: "Formulating Language Model Reinforcement Learning: Responses, Rewards, and Updates"
editionLabel: ENGLISH OVERVIEW · METHOD NOTE
published: 2026-10-23T09:00:00+08:00
description: "A small integer task separates token actions, terminal rewards, sampling budgets, and parameter updates."
tags:
  - Reinforcement Learning
  - Reasoning Training
category: LLM Training Research
series: LLM Training Research Notes · Reasoning RL Volume
seriesOrder: 1
lang: en
---

> **Scope note.** The task and probability table are constructed teaching examples. No model was trained or evaluated. The experimental procedure below is a proposal, not a report of measured capability gains.

## A task with an explicit contract

Given nonnegative integers a and b, return the smallest nonnegative integer n satisfying 3n+a≥b, using exactly one JSON field. For a=2 and b=12, `{"n":4}` is valid. The value 5 satisfies the inequality but violates minimality; the text `The answer is 4` violates the output contract. This deliberately simple task makes verifier mistakes visible; it is not a benchmark for advanced reasoning.

At token step t, the state consists of the prompt and the generated prefix. The action is the next token sampled from the policy. A complete response defines a trajectory. The terminal task reward is one for a valid answer and zero otherwise. Normal termination and a generation-length cutoff must remain distinguishable in the logs.

One can also treat the entire response as a single action in a contextual bandit. [InstructGPT](https://arxiv.org/html/2203.02155v1#S3.SS5) uses this prompt–response–reward framing. The token-level and response-level descriptions connect through the product of conditional token probabilities, including the normal end action under the chosen convention.

## Sampling success is not a training result

Suppose a constructed distribution gives probabilities 0.20, 0.30, 0.40, and 0.10 to the correct JSON, an overlarge integer, an insufficient integer, and a non-JSON answer, respectively. Single-sample expected reward is 0.20.

Four independent draws have probability 1−0.8⁴=0.5904 of containing at least one passing response. This improvement requires no parameter update. It spends more generation budget, and delivering the successful candidate also requires a selector that recognizes it. Repeated deterministic decoding does not justify the independence calculation.

## Separate generation, selection, and optimization

Sampling creates candidates, selection retains some of them, and optimization changes parameters. Supervised training on accepted generations and reward-based policy optimization are different update rules, even when both use generated answers. [DeepSeekMath](https://arxiv.org/html/2402.03300v3#S4) provides a concrete sample–score–update procedure; later articles will examine its GRPO objective rather than treating the loop alone as a complete algorithm.

The single-sample objective averages reward over a specified prompt distribution and response policy. A best-of-several score, a reward augmented with KL regularization, and a task success rate are different quantities. Old sampling policies, the current trainable policy, and a reference policy must also be distinguished.

## Verify the loop before interpreting gains

A verifier that only checks the inequality rewards arbitrarily large integers. A dataset dominated by b≤a rewards a constant-zero answer. All-zero task rewards provide no successful examples and give zero task-reward contribution to the simplest reward-weighted policy-gradient estimator, although other losses or baselines can still produce updates.

The proposed experiment first checks boundary cases, malformed responses, and deterministic rescoring. It then compares an unchanged model with one sample, the unchanged model with multiple samples and selection, and an updated model with one sample, adding matched inference-budget comparisons. Evaluation groups must cover different input conditions and include an untouched test partition. These comparisons have not been run.

The [Chinese article](/posts/2026-10-23-llm-training-rl-01-formulation/) contains the complete derivation, examples, failure cases, and questions for designing the next comparison.
