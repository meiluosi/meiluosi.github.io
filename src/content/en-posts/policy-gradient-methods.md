---
sourceId: 2024-07-05-策略梯度方法详解
slug: policy-gradient-methods
title: Policy Gradients and REINFORCE
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-07-05T00:00:00.000Z
description: An overview of the policy-gradient theorem, REINFORCE, variance reduction, and the CartPole code examples in the source article.
tags:
  - Reinforcement Learning
  - Policy Gradient
  - REINFORCE
  - CartPole
category: Reinforcement Learning
lang: en
---

> **Scope note.** This overview follows a Chinese technical article containing derivations and implementation examples. The examples are not presented here as independently reproduced runs; the source does not provide an experiment artifact sufficient to verify performance claims.

## Optimize the policy directly

Value-based methods first estimate action values and then use them to choose actions. Policy-gradient methods instead represent a policy $\pi_\theta(a\mid s)$ and adjust its parameters to increase expected return.

The policy-gradient theorem gives a way to estimate the direction of improvement from sampled experience without requiring direct access to the environment's transition probabilities. In a score-function form, the update is proportional to the log-probability gradient of an action, weighted by an estimate of its return.

## REINFORCE and variance reduction

REINFORCE collects a trajectory, computes returns, and updates the policy in the direction that makes actions associated with higher returns more likely. This simple Monte Carlo approach can have high variance because each update depends on sampled outcomes.

The source discusses several ways to make the learning signal more useful:

- Subtract a **baseline**, such as a state-value estimate, without changing the expected policy gradient.
- Use an **advantage estimate** to compare an action's outcome with an expected baseline.
- Combine bootstrapped temporal-difference residuals with **Generalized Advantage Estimation (GAE)**, trading bias against variance through its parameters.

## Implementation examples and limits

The Chinese article includes a PyTorch CartPole implementation and discusses extensions to continuous-control settings such as MuJoCo. Those code listings illustrate the algorithmic workflow; the article does not attach run logs, environment versions, or evaluation artifacts that would establish a reproducible result.

When evaluating a policy-gradient method, record the environment, random seeds, training budget, reward definition, and evaluation protocol. A successful example on one control task would not by itself establish general learning ability or recursive self-improvement.

See the Chinese original for the full derivation, pseudocode, CartPole listing, and implementation discussion.
