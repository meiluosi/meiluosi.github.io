---
sourceId: 2026-10-23-llm-training-rl-03-policy-gradient
slug: llm-training-rl-03-policy-gradient
title: "How Policy Gradients Use Rewards: Four Responses and a Baseline"
editionLabel: ENGLISH OVERVIEW · METHOD NOTE
published: 2026-10-23T09:02:00+08:00
description: "An enumerated softmax policy explains reward-weighted gradients, baseline conditions, variance, and implementation boundaries."
tags:
  - Policy Gradient
  - Reinforcement Learning
category: LLM Training Research
series: LLM Training Research Notes · Reasoning RL Volume
seriesOrder: 3
lang: en
---

> **Scope note.** All probabilities and updates belong to a constructed four-response policy. Numerical checks enumerate outcomes and use finite differences; no language model was trained. The tensor snippet in the Chinese article is interface pseudocode, not a completed training implementation.

## Differentiate probabilities, not the checker

The running integer task has four candidate answers with probabilities (0.2, 0.3, 0.4, 0.1) and rewards (1, 0, 0, 0). Parameterizing these probabilities with four independent softmax logits gives expected reward 0.2. Its exact gradient with respect to the logits is (0.16, −0.06, −0.08, −0.02). A gradient-ascent step of size 0.1 raises the first probability to approximately 0.203624.

This finite policy is deliberately simpler than a shared autoregressive network. It illustrates probability normalization rather than guaranteeing improvement across real prompts.

The log-derivative identity expresses the gradient as an expectation of reward times the gradient of log probability, under the current policy. The reward need not be differentiable, but this expression assumes no explicit parameter dependence in the reward and the usual differentiation conditions. The [official Spinning Up derivation](https://github.com/openai/spinningup/blob/master/docs/spinningup/rl_intro3.rst) supplies the general formulation.

## Baselines preserve an expectation under conditions

Subtracting a baseline independent of the sampled action preserves the expected gradient because the expectation of the score function is zero. It does not mean every baseline reduces variance. For the first logit in this example, single-sample variances are 0.1024, 0.0576, and 0.2304 for baselines 0, 0.2, and 2.

For G independent samples, subtracting their inclusive reward mean from every sample scales the expectation of this simple estimator by (G−1)/G. A leave-one-out mean avoids including the sample's own reward and recovers the expectation when G>1 under the stated independence assumptions. Standard-deviation normalization introduces another random weight. These calculations are not an analysis of the full GRPO objective.

## Sequence implementation choices matter

Sequence log probability is the sum over generated-token conditional log probabilities, with termination handled consistently. Prompt and padding positions should not accidentally enter the action sum. Dividing each sequence sum by its own length changes relative weighting; it is not automatically the gradient of the original task-success objective.

The sampling distribution must match the distribution assumed by the estimator or receive an appropriate correction. Temperature, top-p filtering, and reusing old samples all require explicit treatment. A negative reward-weighted log-probability loss converts ascent into descent, and weights are treated as fixed for that policy update. A learned baseline can have its own training objective.

## Check a small policy before a large network

The numerical validation compares analytic gradients with finite differences, verifies baseline expectations and coordinate variances, and enumerates all two-sample pairs for inclusive and leave-one-out baselines. It establishes arithmetic consistency in the finite example, not improved model capability.

The [Chinese article](/posts/2026-10-23-llm-training-rl-03-policy-gradient/) includes the full derivation, tensor interface sketch, and implementation checks. Later articles address controlled sample reuse and clipped updates.
