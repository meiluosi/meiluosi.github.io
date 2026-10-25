---
sourceId: 2026-10-23-llm-training-tool-02-supervision
slug: llm-training-tool-02-supervision
title: "SFT for Tool Interfaces: Syntax, Arguments, and Task Completion"
editionLabel: ENGLISH OVERVIEW · SUPERVISION DESIGN
published: 2026-10-23T10:01:00+08:00
description: "Inspect decision prefixes, target masks, repeated supervision, and closed-loop evaluation before training tool-use models."
tags:
  - Tool Use
  - Supervised Fine-Tuning
category: LLM Training Research
series: LLM Training Research Notes · Tool Interaction Volume
seriesOrder: 2
lang: en
---

> **Scope note.** Counts, losses, and evaluation rates are constructed examples. The companion checks verify supervision arithmetic and event prefixes; no tokenizer, trainer, or model experiment was run.

## Separate three kinds of correctness

A call can satisfy its schema yet query the wrong item or violate the requested warehouse order. Correct calls can still end with a false reservation claim. Report syntax, argument semantics, and complete task outcomes separately.

The inventory trajectory provides three decisions: query HZ, query SZ after observing insufficient HZ stock, then report SZ availability without claiming a reservation. Each target may depend only on earlier observations.

## Choose the supervision unit explicitly

A full causal conversation can supervise assistant spans while retaining tool outputs as unscored context. Alternatively, split it into history-to-next-decision examples and supervise only the new decision. Supervising all historical assistant spans in every split repeats early targets: the first call appears three times, the second twice, and the final response once.

[TRL 0.23.1](https://huggingface.co/docs/trl/v0.23.1/en/sft_trainer) documents completion-only and assistant-only settings, template-mask requirements, and tool data columns. Inspect actual labels rather than assuming a configuration switch guarantees the desired coverage.

## Repeated labels change the objective

Construct three target spans with lengths 4, 6, and 10 and mean negative log-likelihoods 0.5, 1.0, and 0.2. Counting each target once yields 20 positions and a token mean of 0.5. Repeating historical targets yields 34 positions and 20/34, approximately 0.5882. Equal weighting of the three decision means instead yields approximately 0.5667.

These are different weightings, not evidence of model improvement or regression. The lengths are hypothetical, not measured tokenization of the inventory example.

## Inspect rendered inputs and branch coverage

[Transformers chat-template documentation](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating_writing) describes how templates render tools and messages. Verify that call fields survive preprocessing, results retain their roles, future observations remain inaccessible, and stopping tokens receive the intended supervision.

Changing HZ stock from two to three must change the target behavior: stop after HZ. Include insufficient-stock, no-tool-needed, and missing-information cases. Group near-duplicate trajectory families before train/test splitting. Truncation must not silently disconnect calls from results or remove required tool definitions.

## Do not teach failed actions as successful demonstrations

Recovery data may retain an erroneous history as context while supervising a verified correction. Applying ordinary positive SFT loss to every assistant message would also reinforce earlier errors. Validate repairs against the environment rather than altering results to justify a bad action.

## Measure execution as well as next-step prediction

A correct reference history makes offline next-decision prediction easier to interpret, but hides errors the model may make earlier. Closed-loop evaluation lets the model generate calls and consume actual environment returns.

In a constructed example, 90/100 first calls satisfy syntax, 72/100 also meet semantic requirements, and 54/100 full tasks succeed. The conditional 72/90 rate does not replace either unconditional rate. Multiple calls, retries, and no-tool tasks prevent arbitrary multiplication of call-level and task-level metrics.

A proposed experiment compares an unchanged model, final-answer supervision, and call-plus-answer supervision while holding visible context and evaluation rules fixed. It remains unrun. The [Chinese article](/posts/2026-10-23-llm-training-tool-02-supervision/) explains budget accounting and the artifacts needed before drawing training conclusions.
