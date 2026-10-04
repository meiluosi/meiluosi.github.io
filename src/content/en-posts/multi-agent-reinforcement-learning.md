---
sourceId: 2024-07-20-多智能体强化学习
slug: multi-agent-reinforcement-learning
title: "Multi-Agent Reinforcement Learning: Cooperation and Competition"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-07-20T00:00:00.000Z
description: An overview of non-stationarity, credit assignment, CTDE, value decomposition, and coordination in multi-agent RL.
tags:
  - Multi-Agent Reinforcement Learning
  - Coordination
  - CTDE
category: Reinforcement Learning
lang: en
---

> **Scope note.** This is an English overview of a longer Chinese survey. It summarizes the algorithms and example setups in the source. The three-robot delivery scenario is presented as pseudocode, not as a reported run; the article provides no trained policy, benchmark results, or experiment logs.

## Why study multiple learning agents?

In a multi-agent environment, each agent's outcome depends in part on what other agents do. The setting may be cooperative, competitive, or mixed: robots can share a task, players can compete, and teams can cooperate internally while competing with one another.

The source article describes a stochastic game with a set of agents, a shared state, an action space for each agent, a joint state transition, and a reward for each agent. Because the transition depends on the joint action, the number of possible action combinations can grow quickly as agents are added.

Three recurring difficulties follow:

- **Non-stationarity:** while one agent learns, the policies of the others also change, so its effective environment changes over time.
- **Credit assignment:** a team reward does not immediately reveal which agent or action contributed to it.
- **Coordination:** individually reasonable actions may not combine into a good joint behavior.

## Centralized training, decentralized execution

One common design is **Centralized Training with Decentralized Execution (CTDE)**. During training, a value function or critic may use global state and information about multiple agents. At execution time, each policy acts using its own observation. This separates the information available for learning from what an individual agent can access when deployed.

The guide surveys several algorithm families rather than reporting a controlled comparison:

| Family | Main idea in the source | Typical trade-off |
| --- | --- | --- |
| Independent Q-Learning (IQL) | Each agent learns its own action-value function and treats the others as part of the environment | Simple and decentralized, but coordination and convergence can be difficult |
| VDN and QMIX | Learn local values and combine them into a team value | Designed for cooperative tasks; QMIX uses a more expressive monotonic mixing function than VDN's additive combination |
| MADDPG | Train an agent's critic with joint information, while its actor uses local observations | Supports continuous actions, with greater training-time information and computation requirements |
| MAPPO | Use PPO-style policy updates with a centralized value function | A practical CTDE variant; the source's recommendation is not a current benchmark ranking |

The right choice depends on the environment, observability, action spaces, and what behavior needs to be evaluated. A survey or algorithm name alone does not establish which method performs best for a particular task.

## Communication and coordination signals

Some systems give agents an explicit communication channel. The article sketches CommNet, where agents exchange encoded observations, and an attention-based QCOMM pattern, where an agent learns to weight information from other agents.

Communication can help coordinate actions, but it adds design choices: what to send, to whom, and when. The source also proposes inspecting trajectories and monitoring coordination signals such as action diversity, action variance, and value-function variance. These signals can help describe behavior, though none alone proves that a team is solving its task better.

## An illustrative cooperative task

The article sketches a scenario in which three robots move a large box toward a target. The example defines local positions, box and target positions, movement and grasping actions, a team reward for moving closer, and a penalty for collisions. It then gives a PyMARL-style QMIX configuration as pseudocode.

This is a proposed experiment setup, not an empirical result in the article. It includes no environment run, trained policy, success rate, or comparison against another algorithm. A reproducible report would need the environment implementation, training details, repeated seeds, evaluation protocol, and measured task outcomes.

## Connection to RSI and AGI

Multi-agent learning is relevant to questions about collective capability: how agents coordinate, share information, and improve behavior from feedback. To claim an improvement, however, a system needs measurements at both the team and agent levels—for example, task completion alongside collisions, resource use, and robustness when other agents or conditions change.

This survey introduces tools for studying cooperation and competition. It does not show that a multi-agent system recursively improves itself or reaches AGI. The full Chinese article contains the algorithm sketches, additional pseudocode, debugging suggestions, and references.
