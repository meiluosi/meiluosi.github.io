---
sourceId: 2026-10-23-llm-training-rl-05-grpo
slug: llm-training-rl-05-grpo
title: "How GRPO Compares Responses: Group Advantages, Zero Variation, and Length Weights"
editionLabel: ENGLISH OVERVIEW · METHOD NOTE
published: 2026-10-23T09:04:00+08:00
description: "Constructed group calculations explain reward normalization, homogeneous groups, sampling cost, and sequence-versus-token aggregation."
tags:
  - GRPO
  - Reinforcement Learning
category: LLM Training Research
series: LLM Training Research Notes · Reasoning RL Volume
seriesOrder: 5
lang: en
---

> **Scope note.** All rewards, probabilities, and length comparisons are constructed examples. Standard-library calculations were checked, but no GRPO model was trained. Library options are described for a fixed documentation version, not as tested recommendations.

## Compare answers to the same question

In the outcome-supervision formulation of [DeepSeekMath](https://arxiv.org/html/2402.03300v3#S4.SS1), multiple responses to one question supply a group-relative signal without a separately trained value model. The response-level advantage is applied across its generated positions; this does not verify each token independently.

For rewards (1,0,0,0), the mean is 0.25. Using population standard deviation gives approximately (1.732,−0.577,−0.577,−0.577). Using sample standard deviation gives (1.5,−0.5,−0.5,−0.5). Record the convention and numerical stabilization instead of assuming every implementation produces identical values.

All-correct and all-incorrect groups both have zero centered rewards. Numerical handling of zero standard deviation does not create comparative information. Other loss terms or other groups can still update shared parameters. The two homogeneous cases need separate diagnostics.

## More responses cost more and alter the batch

Under independent draws with fixed success probability p, the probability of a mixed group is 1−p^G−(1−p)^G. At p=0.2 it is 0.5888 for four responses and about 0.832225 for eight. This is not single-response accuracy or final-selection accuracy. More responses also spend generation and scoring budget, potentially reducing prompt diversity under a fixed budget.

Group standardization changes cross-group weighting. Rewards (1,0,0,0) and (0.51,0.50,0.50,0.50) have the same standardized advantages under the stated convention, despite different absolute gaps. Small scoring noise can therefore acquire substantial relative weight.

## Length aggregation is part of the objective

The original objective combines clipped probability-ratio contributions with a reference term and averages within each response before averaging responses. Removing the value model does not remove sampling, reward evaluation, or reference-policy choices.

For two responses of lengths two and eight, sequence averaging gives per-token weights 1/4 and 1/16; token averaging gives 1/10 to both. If all short-response contributions are +1 and all long-response contributions are −1, the aggregate values are zero and −0.6. This establishes different weighting, not a guarantee of a particular learned length or a zero gradient.

[TRL 0.23.1](https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer) exposes multiple loss and reward-scaling options and documents `dapo` as its default loss type. A trainer name alone does not specify the original paper's objective. Record effective masks, normalization, accumulation, and distributed reduction as well.

## Diagnose the signal before reporting gains

Track all-correct, all-incorrect, mixed, and failed-scoring groups separately. Preserve prompt-to-response group membership through collection and filtering. Distinguish probability-ratio clipping from generation truncation. Mean-zero advantages follow from centering and are not independent evidence of healthy training.

The proposed experiment varies standard-deviation scaling first and length aggregation separately, with fixed task splits and matched budgets. No training comparison has been run. The [Chinese article](/posts/2026-10-23-llm-training-rl-05-grpo/) provides the calculations, implementation boundaries, and independent-evaluation questions.
