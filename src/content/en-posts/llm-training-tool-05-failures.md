---
sourceId: 2026-10-23-llm-training-tool-05-failures
slug: llm-training-tool-05-failures
title: "Learning from Failed Trajectories: Repairs, Supervision, and Selection Bias"
editionLabel: ENGLISH OVERVIEW · DATA CONSTRUCTION
published: 2026-10-23T10:04:00+08:00
description: "Separate retries from training, validate repaired actions, and avoid reinforcing errors or hiding selection bias."
tags:
  - Tool Use
  - Training Data
category: LLM Training Research
series: LLM Training Research Notes · Tool Interaction Volume
seriesOrder: 5
lang: en
---

> **Scope note.** Examples and counts are constructed. A tiny local fixture checks repaired requests and supervision boundaries; no model self-correction or parameter training was evaluated.

## Diagnose before generating reflection

Distinguish malformed output, wrong arguments, environment faults, answer inconsistency, and evaluator errors. Record the earliest verified divergence and final stopping reason separately. A model's explanation is a diagnostic hypothesis, not proof of causation.

Retries provide more attempts. Contextual correction changes available feedback or memory. Parameter training changes weights. [Reflexion v4](https://arxiv.org/html/2303.11366v4#S3) uses textual reflection and memory without weight updates; its terminology should not erase this distinction.

## Build two different kinds of examples

If a user requests A17 but the model queries A18, an `unknown_sku` result does not itself reveal A17. The correction must use the original request.

A first-action example maps the original request and tool definition directly to A17. A recovery example retains the A18 call and error as context but supervises only the verified correction. Applying positive SFT loss to the entire failed history would also reinforce A18.

Execute the revised action in the appropriate environment. Do not attach the old error result to a changed request or edit observations to justify an action. Preserve parent trajectory, branch point, repair source, snapshot, and validation scope. Write operations require real state restoration or compensation, not text edits.

## Success filtering changes the sample distribution

In a constructed pool of 50 easy and 50 hard tasks, 45 easy and 10 hard successes yield a retained set that is 45/55, or about 81.8%, easy. Preserve per-family retention and attempt budgets. Deduplication does not undo this selection effect.

Preference pairs require comparable task conditions, information, budgets, and initial environments. An answer produced after seeing corrective feedback is not automatically a fair alternative under the original no-feedback context. Different actions can legitimately produce different observations, provided the environment and training representation handle them consistently.

## Validate recovery independently

Use execution evidence, task rules, or human review rather than relying solely on the same model to propose and certify repairs. Hidden test feedback changes the evaluation protocol; training on those instances requires a new independent test set.

The companion fixture rejects A18, returns HZ stock 2 for A17, checks first-action versus recovery target boundaries, and detects a stale result attached to a revised request. This validates data construction only.

Compare first-attempt completion, recovery on fixed erroneous histories, closed-loop outcomes, and costs. Recovery rates over each model's own failures have different denominators. The [Chinese article](/posts/2026-10-23-llm-training-tool-05-failures/) proposes these experiments without claiming they have been run.
