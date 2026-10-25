---
sourceId: 2026-10-23-llm-training-tool-01-trajectories
slug: llm-training-tool-01-trajectories
title: "Tool-Use Samples: From Inventory Lookups to Interaction Trajectories"
editionLabel: ENGLISH OVERVIEW · CONTROLLED ENVIRONMENT
published: 2026-10-23T10:00:00+08:00
description: "Separate requested actions, executed results, and final answers in a controlled inventory task."
tags:
  - Tool Use
  - Training Data
category: LLM Training Research
series: LLM Training Research Notes · Tool Interaction Volume
seriesOrder: 1
lang: en
---

> **Scope note.** The inventory, trajectories, and checks are constructed teaching examples. No warehouse service or language model was invoked. Passing the checks validates fixture consistency, not model performance.

## Define success before collecting examples

A user needs three units of item A17. Query warehouse HZ first; query SZ only if HZ lacks enough units. Do not split the order or reserve stock. With available quantities HZ=2 and SZ=5, the correct report selects SZ and states that no reservation was made.

The read-only `get_stock` contract takes a nonempty item identifier and a warehouse enum. A successful lookup can still report insufficient stock. An error means the quantity is unknown, not zero.

## Preserve actions and observations separately

The teaching log records a call with identifier c1, its result, another call c2, its result, and a final answer. Call identifiers associate results with requests. Calls and final answers are model outputs; results originate in the environment. The full training record must also retain the user request, tool definitions, and relevant instructions.

The array in the [Chinese article](/posts/2026-10-23-llm-training-tool-01-trajectories/) is a custom audit format, not a universal model API. [Transformers 4.57.1 documentation](https://huggingface.co/docs/transformers/v4.57.1/en/chat_extras) demonstrates rendering tool definitions and returning execution results to the conversation. Adapting a log requires checking the selected model's chat template and serialization conventions.

Removing the first result hides why the second query was needed. Swapping results can preserve fluent prose while corrupting the decision evidence. A final answer alone cannot establish that a lookup occurred.

## History is not complete environment state

A policy conditions its next action on visible interaction history. After a tool request, an observation extends that history. [ReAct's method section](https://arxiv.org/html/2210.03629v3#S2) describes context-dependent actions and observations; its prompting experiments should not be mistaken for parameter training. This example does not require a verbose reasoning trace before each call.

The fixture freezes stock at `stock-v1`. Real inventory may change between lookup and reservation, so an earlier observation cannot guarantee later availability. External notes remain task data rather than new authorization. Adding a state-changing operation would require a new execution contract.

## Check more than parsability

Validate argument types, requested item, unique call IDs, result association, event order, snapshot consistency, and final structured selection. Reject orphan results, missing results, duplicate results, premature completion, and claims that a read-only lookup reserved stock.

The companion checker covers all 36 stock pairs with each quantity from zero through five, plus deliberately corrupted trajectories. It checks structured final fields, not arbitrary natural-language factual consistency. The displayed sentence is reviewed manually. Nor can fixture consistency prove that a remote service actually executed: real evidence must come from execution-side records.

## Prepare supervision without leaking future observations

Calls and final answers can be prediction targets while tool results remain conditioning context. Masking result tokens from loss does not mean removing them from input. Conversely, a later result must not become visible while predicting an earlier request.

Keep examples where no lookup is needed, and later test held-out items, phrasings, and states. Complete trajectories make the learning problem inspectable; they do not by themselves demonstrate successful training.
