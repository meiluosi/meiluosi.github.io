---
sourceId: 2026-10-23-llm-training-eval-03-costs
slug: llm-training-eval-03-costs
title: "Accounting for Training Cost: Tokens, Compute, Time, and Peak Memory"
editionLabel: ENGLISH OVERVIEW · COST ACCOUNTING
published: 2026-10-23T11:02:00+08:00
description: "Separate token denominators, approximate FLOPs, measured runtime, memory components, and end-to-end research costs."
tags:
  - Training Cost
  - Reproducibility
category: LLM Training Research
series: LLM Training Research Notes · Evaluation and Reproducibility Volume
seriesOrder: 3
lang: en
---

> **Scope note.** Every number is a teaching calculation. No hardware benchmark, cloud price lookup, or training run was performed.

## Count what the denominator represents

Two examples with 800 prompt and 200 answer tokens each contain 2,000 nonpadding input positions and 400 supervised targets. Padding each to 1,024 produces 2,048 tensor positions. In a hypothetical 0.5 seconds, throughput is respectively 4,000 input tokens/s, 800 targets/s, or 4,096 positions/s. Unscored context is not free computation.

## Treat compute formulas as approximations

For an appropriate dense full-training approximation, C≈6ND gives 6×10^15 FLOPs at N=10^8 and D=10^7. [Kaplan et al., Section 2.1](https://arxiv.org/html/2001.08361v1#S2.SS1) explains the counting assumptions and omitted terms. State parameter conventions, padding, and attention limits. Trainable adapter parameter count cannot simply replace N, because frozen base layers still perform work.

Two 1,024-token sequences and one 2,048-token sequence have equal token totals but different sums of squared lengths: the latter is twice as large. This describes a dense-attention component, not necessarily twice the total runtime.

## Itemize memory rather than quoting one constant

[Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_memory_anatomy) separates weights, gradients, optimizer state, activations, and temporary buffers. An explicitly assumed ledger of 2-byte weights, 2-byte gradients, 4-byte master weights, and two 4-byte Adam states totals 16 bytes per parameter: 1.6 billion bytes, about 1.49 GiB, for 100 million parameters, before activations and other overhead.

Implementations differ. Sharding, quantization, offload, master-weight choices, and gradient dtype change the ledger. Report per-device peaks with measurement scope, not just checkpoint size.

## Measure the whole experiment

If training occupies 70 minutes of a 100-minute job, doubling that part's speed yields 65 minutes total, or about 1.54× overall speedup. Separate warmup and compilation from steady-state measurements, and account for asynchronous execution correctly.

Include data preparation, rollout generation, feedback, evaluation, storage, and failed runs. Ten two-hour searches cost twenty hours even if the selected final run took two. GPU-hours are not interchangeable across devices, and tokens are not currency without verified billing information.

The [Chinese article](/posts/2026-10-23-llm-training-eval-03-costs/) recommends separate columns for estimates and measurements and comparison at matched quality or cost.
