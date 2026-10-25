---
sourceId: 2026-10-23-llm-training-rl-02-verifiers
slug: llm-training-rl-02-verifiers
title: "What Does the Reward Actually Check? Answers, Formats, Tests, and Process Feedback"
editionLabel: ENGLISH OVERVIEW · METHOD NOTE
published: 2026-10-23T09:01:00+08:00
description: "An executable integer checker illustrates verifier coverage, malformed outputs, incomplete tests, and the difference between composite reward and task success."
tags:
  - Reinforcement Learning
  - Reward Design
  - Verification
category: LLM Training Research
series: LLM Training Research Notes · Reasoning RL Volume
seriesOrder: 2
lang: en
---

> **Scope note.** The integer task and reward comparison are constructed examples. The checker in the Chinese article was executed against synthetic cases; no model training or capability evaluation was performed. The proposed reward comparison remains an experimental design.

## Define what acceptance means

The running task asks for the smallest nonnegative integer n satisfying 3n+a≥b, returned as a JSON object with exactly one integer field. For a=2 and b=12, the value 4 passes. The value 5 satisfies the inequality but fails minimality. Returning a string, an extra field, or non-JSON text violates this particular interface contract, even if another application might choose a more permissive contract.

The checker verifies both that n satisfies the inequality and that n−1 fails when n is positive. Monotonicity makes these sufficient for minimality. A ceiling-based formula and exhaustive enumeration provide alternative ways to check small cases.

The article rejects duplicate keys, non-finite constants, Boolean values, floating-point values, truncated JSON, and excess fields. Python's default decoder accepts repeated names and non-finite constants, so successful parsing alone is insufficient for this contract. See the [Python 3.13 JSON documentation](https://docs.python.org/3.13/library/json.html#standard-compliance-and-interoperability). Invalid task inputs raise an exception rather than becoming model failures.

## Test the evaluator itself

The validation runs the actual published Python block. It checks 18 output cases and 10,584 candidate answers across 441 integer input pairs, using enumeration as the answer oracle. It also checks three invalid input cases and counterexamples to two deliberately wrong predicates. These are finite synthetic checks, not proof of unrestricted robustness or training effectiveness.

A checker that omits minimality accepts overlarge answers. A strict inequality substituted for a non-strict one rejects an exact boundary solution. Tests should catch these plausible defects rather than only exercise happy paths. [EvalPlus](https://arxiv.org/html/2305.01210v3#S2) motivates expanded code tests and discusses mutation-based testing requirements; this article does not transfer its benchmark results to the teaching checker.

## Outcome and process feedback cover different claims

A final answer can be correct despite an incorrect intermediate equation. An outcome check does not validate the derivation. Process feedback can inspect intermediate steps, but step segmentation, annotation errors, and score aggregation introduce further choices. Summing positive step rewards may favor splitting a statement into more steps; taking a minimum may make a single scoring error dominate.

[Let's Verify Step by Step](https://arxiv.org/html/2305.20050v1#S2) trains outcome- and process-supervised reward models and evaluates solution selection. Its findings do not establish that adding a process reward necessarily improves every reinforcement-learning policy.

For generated code, passing a finite test suite establishes agreement on those tests, not universal correctness. Test inputs must respect the specified domain. Program timeouts and evaluation-infrastructure failures should remain distinct; the article's text checker executes no generated code.

## A composite reward is not a success rate

Suppose valid formatting earns 0.2 and correctness adds 0.8. A constructed system A has valid formatting on every output and complete success on 50%. System B succeeds completely on 60% and is unparseable otherwise. Both average 0.6 reward, despite different complete success rates. This follows from the chosen weights, not from a measured training run.

Report component scores and complete success separately. Independent evaluation should add genuinely different checks or information, rather than copying the training checker. The proposed comparison freezes the task and verifier, then compares complete-success-only feedback with added format feedback under matched training and generation budgets. No such model comparison has been run.

The [Chinese article](/posts/2026-10-23-llm-training-rl-02-verifiers/) includes the executable checker, boundary cases, reward calculations, and experiment questions.
