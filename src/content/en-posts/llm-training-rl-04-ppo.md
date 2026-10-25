---
sourceId: 2026-10-23-llm-training-rl-04-ppo
slug: llm-training-rl-04-ppo
title: "What PPO Limits: Probability Ratios, Clipping, and Value Estimates"
editionLabel: ENGLISH OVERVIEW · METHOD NOTE
published: 2026-10-23T09:03:00+08:00
description: "Positive and negative advantage examples explain PPO-Clip, its limits, reference policies, and terminal handling in GAE."
tags:
  - PPO
  - Reinforcement Learning
category: LLM Training Research
series: LLM Training Research Notes · Reasoning RL Volume
seriesOrder: 4
lang: en
---

> **Scope note.** The probability ratios, advantages, and three-step trajectory are constructed arithmetic examples. No model was trained. The article discusses PPO-Clip rather than treating every PPO variant as identical.

## Clip an incentive, not the policy itself

The ratio compares the current probability of a sampled action with its probability under the policy that generated the batch. Old log probabilities stay fixed while that batch is reused. A ratio of 1.4 means a relative increase of 40%, not an increase of 40 percentage points.

With epsilon 0.2, PPO-Clip maximizes the minimum of the ordinary ratio-times-advantage term and its clipped counterpart. For advantage 2 and ratio 1.4, the contribution is 2.4. For advantage 2 and ratio 0.6, it is 1.2. For advantage −2, ratios 0.6 and 1.4 give −1.6 and −2.8 respectively.

Favorable movement eventually reaches a flat branch; unfavorable movement is not flattened in the same way. Shared parameters, other samples, and optimizer state can still move ratios beyond the interval. Clipping is not a hard probability or KL constraint. See the [official PPO explanation](https://github.com/openai/spinningup/blob/master/docs/algorithms/ppo.rst) and [implementation](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py).

## Old policy and reference policy have different roles

The old policy explains how the current batch was sampled. A reference model provides a separate anchor, often the initial SFT model in language-model training. Small updates relative to successive old policies do not guarantee proximity to the initial model. [InstructGPT](https://arxiv.org/html/2203.02155v1#S3.SS5) includes a separate SFT-reference KL penalty in its RL setup.

Tokenwise ratios and a complete response's product of ratios also differ. Report the direction, reference, and sampling distribution for each KL metric.

## Estimate advantage with explicit boundaries

A value function predicts future return from a prefix; it does not certify a mathematical statement. GAE accumulates discounted TD residuals, introducing an estimation tradeoff described in the [GAE paper](https://arxiv.org/abs/1506.02438v6).

In a constructed three-step terminal trajectory, rewards are (0,0,1), values are (0.2,0.4,0.7), terminal value is zero, and gamma is one. Residuals are (0.2,0.3,0.3). Lambda 0.5 gives advantages (0.425,0.45,0.3); lambda one gives (0.8,0.6,0.3).

True termination and collection cutoffs require different boundary treatment. A generation budget limit may be a real terminal failure if the task defines it that way, or an artificial collection boundary in a continuing trajectory. The field name alone does not decide.

## Inspect metrics before interpreting them

A reported clip fraction may count every out-of-range ratio, even when the sample is not on the flat branch. Check the implementation. Track task success, reward components, old-policy and reference-policy KL, entropy, length, and value error separately.

The [Chinese article](/posts/2026-10-23-llm-training-rl-04-ppo/) contains the full numerical table, GAE calculation, failure cases, and proposed controlled training comparison. Only the arithmetic checks have been run.
