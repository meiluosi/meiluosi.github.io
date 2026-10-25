---
sourceId: 2026-10-23-llm-training-tool-03-rewards
slug: llm-training-tool-03-rewards
title: "Rewards for Multistep Tasks: Outcomes, Call Costs, and Recovery"
editionLabel: ENGLISH OVERVIEW · REWARD DESIGN
published: 2026-10-23T10:02:00+08:00
description: "Trace returns through tool interactions and examine how costs, shaping, and retries change the objective."
tags:
  - Tool Use
  - Reward Design
category: LLM Training Research
series: LLM Training Research Notes · Tool Interaction Volume
seriesOrder: 3
lang: en
---

> **Scope note.** All trajectories, probabilities, and value estimates are constructed. The companion script checks arithmetic and counterexamples; no reinforcement-learning run was performed.

## Terminal feedback may evaluate the whole trajectory

The inventory task requires correct items, warehouse order, evidence-based answers, and no reservation claim. A terminal reward can inspect the entire log while being delivered only at the end. Guessing the right warehouse is not sufficient under this contract.

With immediate rewards (0,0,1) and no discounting, all three decisions have return-to-go 1. This does not assign equal causal contribution. Subtracting hypothetical history values (0.6,0.7,0.9) gives advantages (0.4,0.3,0.1), which compare outcomes against expectations rather than prove which action caused success. [Spinning Up](https://github.com/openai/spinningup/blob/master/docs/spinningup/rl_intro3.rst) derives reward-to-go for policy gradients.

## Costs and discounting change preferences

Charging 0.05 per lookup gives rewards (-0.05,-0.05,1) and undiscounted returns (0.90,0.95,1). A hypothetical policy with success probability 1 and two expected calls scores 1−2c; another with probability 0.9 and one expected call scores 0.9−c. Their ordering reverses at cost c=0.1. These are illustrative policies, not measured inventory results.

With discount 0.9, the first return becomes 0.715. Define whether a step means a decision, token, or time interval. Report timeouts instead of silently dropping them, and distinguish task termination from collection truncation.

## Dense rewards can reward the wrong activity

A 0.1 bonus for any valid call combined with a 0.05 call charge still pays 0.05 for a useless repetition. Ten such calls earn 0.5 without task success. A step cap limits exploitation but does not repair the objective.

[Potential-based shaping research](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) analyzes policy-preserving transformations under specified MDP conditions. The article separately derives a finite-episode telescoping identity for extra rewards γΦ(next)−Φ(current). With terminal potential zero, a fixed potential, consistent discounting, and a fixed start distribution, the added return is policy-independent. Dropping terminal adjustments invalidates that argument. This says nothing by itself about neural-network convergence speed.

## Recovery needs a new environment contract

If queries can time out, specify retries and correct unresolved-status reporting. Truthfully reporting failure is not the same as obtaining the requested inventory answer. Deployment faults and broken evaluation infrastructure also require different accounting rules.

Under the explicit assumption of independent 0.8 success probability per attempt, at most two attempts yield 0.96 completion probability and 1.2 expected calls. The gain comes from additional attempts; it does not establish better learned decisions. Persistent service failures violate independence.

Keep success, costs, faults, constraint violations, and process scores separate before combining them into a versioned reward. The [Chinese article](/posts/2026-10-23-llm-training-tool-03-rewards/) proposes controlled comparisons but reports no training gains.
