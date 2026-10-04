---
sourceId: 2024-06-09-强化学习笔记记录
slug: reinforcement-learning-notes
title: "Reinforcement Learning: Core Ideas and Practice Notes"
editionLabel: ENGLISH OVERVIEW · STUDY NOTE
published: 2024-06-09T00:00:00.000Z
description: A map of Markov decision processes, value- and policy-based methods, model-based learning, and practical algorithm selection.
tags:
  - Reinforcement Learning
  - MDP
  - DQN
  - Policy Optimization
category: Reinforcement Learning
lang: en
---

> **Scope note.** This is an English overview of an introductory Chinese study note. It organizes concepts and algorithm families; it is not an experiment report and does not claim that a particular agent was trained or evaluated.

## The interaction loop

Reinforcement learning studies how an agent can choose actions from experience to increase cumulative reward. The article frames the problem as a Markov decision process (MDP), commonly written as $(S,A,P,R,\gamma)$: states, actions, transition dynamics, rewards, and a discount factor.

An agent follows a policy, receives observations and rewards from an environment, and uses those interactions to improve its decisions. Value functions estimate expected return; policies specify how actions are selected.

## Three broad approaches

The note groups methods into three families:

1. **Value-based methods** learn values such as $Q(s,a)$ and derive an action choice from them. The source discusses dynamic programming, temporal-difference learning, Q-learning, SARSA, and DQN.
2. **Policy-based methods** optimize a policy directly. The article introduces policy gradients and REINFORCE, then points toward Actor–Critic methods.
3. **Model-based methods** use or learn a model of how the environment changes, then use that model for planning or policy improvement.

These families trade off assumptions, action-space support, sample use, and implementation complexity. The note's selection advice is introductory guidance, not a controlled comparison.

## From concepts to practice

The source also reviews exploration, reward design, replay and target networks, and practical challenges such as unstable training and high-variance returns. Before comparing methods, an experiment still needs a defined environment, baselines, repeated runs where appropriate, and recorded configurations.

For the original explanations, equations, and code examples, see the Chinese study note.
