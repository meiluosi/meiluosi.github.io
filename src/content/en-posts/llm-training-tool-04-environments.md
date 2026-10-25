---
sourceId: 2026-10-23-llm-training-tool-04-environments
slug: llm-training-tool-04-environments
title: "Independent Tool Evaluation: Resets, Randomness, and Replay"
editionLabel: ENGLISH OVERVIEW · ENVIRONMENT VALIDATION
published: 2026-10-23T10:03:00+08:00
description: "Detect cross-task cache contamination, separate random sources, and distinguish replay from fresh execution."
tags:
  - Tool Use
  - Evaluation
category: LLM Training Research
series: LLM Training Research Notes · Tool Interaction Volume
seriesOrder: 4
lang: en
---

> **Scope note.** The local checks use a tiny serial in-memory fixture, not a model, warehouse service, or Gymnasium implementation. They do not validate production concurrency or write transactions.

## A task includes its environment

An inventory request alone does not define an evaluation instance. Include the stock snapshot, tool contract, budget, fault schedule, and scoring rules. Different stock states can change outcomes without any model change.

A read-only tool can retain mutable cache state. If task A has HZ stock 2 and task B has stock 4, a cache keyed only by item and warehouse may return 2 in B. Reset per-episode caches or use valid snapshot-aware cache semantics. Intentional warm-cache evaluation should be labeled explicitly.

## Reset all behavior-relevant state

Clear counters, caches, fault positions, logs, and completion flags. Return independent observation objects so caller mutation cannot rewrite environment state or stored history. [Gymnasium's API documentation](https://gymnasium.farama.org/api/env/) separates reset, step, termination, and truncation, and explains that `reset(seed=None)` does not reseed an existing generator. These are interface lessons, not a claim that this fixture implements Gymnasium.

## Match randomness to the question

Separate task generation, faults, and model sampling. A shared global random stream can shift when an unrelated operation consumes a draw. Seeds require implementation, version, ordering, and initial-state context to be useful.

The fixture keys faults by instance, operation, arguments, and attempt count. HZ's first query times out and its second succeeds regardless of an intervening SZ query. This is a deliberate environment model, not a universal fairness guarantee: real outages may be time-correlated or depend on load. Hidden fault schedules must not enter model observations.

## Replay is not fresh execution

Replay can validate parsing and scoring of a recorded trajectory. Fresh execution allows the model to choose new actions. A replay engine must reject an unmatched request instead of returning the next result by position; otherwise it can attribute HZ stock to an SZ query.

Clean environment resets also do not ensure independent data. Split task families before generating paraphrases, specify held-out dimensions, and exclude hidden expected answers from model-visible diagnostics.

## Evidence from the local checks

The companion script reproduces stale-cache contamination, verifies corrected task-order independence and observation copying, checks keyed faults and counter resets, and rejects mismatched or exhausted replay requests. These are environment-contract checks, not model success rates.

The [Chinese article](/posts/2026-10-23-llm-training-tool-04-environments/) lists the artifacts needed to reconstruct an instance and the limits of this serial fixture.
