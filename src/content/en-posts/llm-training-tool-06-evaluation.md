---
sourceId: 2026-10-23-llm-training-tool-06-evaluation
slug: llm-training-tool-06-evaluation
title: "Evaluating Tool Models: Success, Reliability, and Cost"
editionLabel: ENGLISH OVERVIEW · EVALUATION PROTOCOL
published: 2026-10-23T10:05:00+08:00
description: "Separate candidate coverage from repeated reliability and report costs, constraints, and paired regressions."
tags:
  - Tool Use
  - Evaluation
category: LLM Training Research
series: LLM Training Research Notes · Tool Interaction Volume
seriesOrder: 6
lang: en
---

> **Scope note.** Probabilities, tables, and cost calculations are constructed. No tool model or training intervention was evaluated.

## Define the system and success contract

Fix task snapshots, tools, budgets, retries, decoding, and scoring before comparing results. Parsers, format repair, memory, and selectors belong to the system under evaluation. A deterministic inventory baseline can expose fixture problems without claiming to replace general language capability.

Success must include evidence and applicable constraints. Reporting insufficient stock can correctly complete a query. Reporting an unresolved timeout is distinct from obtaining an inventory answer. [τ-bench v1](https://arxiv.org/html/2406.12045v1#S3) combines outcome and required-output checks while explicitly acknowledging that these do not capture every process-policy violation.

## Distinguish coverage from reliability

For a known per-task independent success probability p=0.8, at least one success in three attempts has probability 0.992; all three succeeding has probability 0.512. The former is pass@k, while τ-bench calls the latter pass^k. Coverage also requires a practical selector to deliver a successful result.

Given n observed trials with c successes and n≥k, subset estimators are one minus C(n−c,k)/C(n,k) for pass@k and C(c,k)/C(n,k) for pass^k. With n=5,c=4,k=3, they are 1 and 0.4. These differ from treating observed c/n as known p. A finite estimate of 1 is not a guarantee.

Compute per task before averaging. Tasks with true success probabilities 1 and 0 have mean three-run reliability 0.5, not the cube of their mean success rate. Shared memory, persistent faults, and changing state can violate independent-trial assumptions.

## Count failures in cost accounting

A constructed four-task batch has calls (1,2,5,4), durations (2,4,20,12), and only the first two tasks succeed. All-task averages are 3 calls and 9.5 seconds; success-only averages are 1.5 calls and 3 seconds. Total calls divided by successes is 6, a batch amortization measure rather than mean successful-trajectory cost.

Report tokens, repair, scoring, retries, elapsed time, and timeout rates with clear units. Separate attempted violations from operations blocked by the executor. A final answer can falsely claim a reservation even when the tool cannot create one.

## Preserve paired outcomes and independence

A constructed paired table has 60 both-success, 15 repairs, 5 regressions, and 20 both-failure tasks. Success rises from 65% to 75%, but regressions remain visible. Use task-family grouping when assessing uncertainty instead of treating repeated runs as unrelated tasks.

Keep development and test decisions separate. Document simulated-user limitations and exact system scope. The [Chinese article](/posts/2026-10-23-llm-training-tool-06-evaluation/) closes this volume with a reporting protocol and introduces the final volume on evaluation and engineering reproducibility.
