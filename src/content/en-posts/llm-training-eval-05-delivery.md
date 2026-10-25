---
sourceId: 2026-10-23-llm-training-eval-05-delivery
slug: llm-training-eval-05-delivery
title: "Why Model Behavior Changes After Delivery: Templates, Quantization, and Regression"
editionLabel: ENGLISH OVERVIEW · EXECUTED TOY EXAMPLE
published: 2026-10-23T11:04:00+08:00
description: "Separate input contracts, weight conversion, generation, parsing, and task-level regression."
tags:
  - Model delivery
  - Regression testing
category: LLM Training Research
series: LLM Training Research Notes · Evaluation and Reproducibility Volume
seriesOrder: 5
lang: en
---

> **Scope note.** Custom token sequences, decimal rounding, stopping rules, and paired counts were checked in Python. No language model, real quantizer, or inference server was evaluated.

## Preserve the input contract

Identical messages need not yield identical model inputs. Record rendered text, token IDs, masks, truncation, and any server-inserted instructions. A toy sequence [1,2,10,3,4] differs from both [1,1,2,10,3,4] and [1,2,10,3], even though its content placeholder is unchanged.

[Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/chat_templating) warns against duplicating special tokens after template rendering. Generation prefixes depend on the model's template. Check single-request and mixed-length batching separately.

## Identify the entire model package

[PEFT 0.17.0](https://huggingface.co/docs/peft/v0.17.0/developer_guides/checkpoint) explains that adapter checkpoints depend on the original model. Pin the base revision, adapter configuration, tokenizer, and any changed vocabulary parameters.

For ordinary additive LoRA, merging represents W+ΔW in ideal arithmetic; finite precision and unsupported combinations still require validation. Compare the unconverted reference, merged model, and quantized model separately.

## Test numerical and task behavior

Let Q round to the nearest decimal tenth. With w=0.14 and d=0.04, Q(w+d)=0.2, Q(w)+d=0.14, and Q(Q(w)+d)=0.1. This executed counterexample shows that conversion order can matter; it is not an implementation of NF4 or GPTQ.

Scores [1.001,1.000] and [0.999,1.000] select different greedy tokens despite small changes. Numerical tolerance alone does not establish task equivalence.

## Record generation and stopping semantics

[Generation strategies](https://huggingface.co/docs/transformers/v4.57.1/generation_strategies) distinguish greedy selection from sampling. Record effective decoding settings, output limits, stop rules, and termination reasons. Greedy decoding does not guarantee cross-platform identity.

A three-token toy JSON sequence becomes incomplete when capped at two tokens or stopped on its middle token. Real parsing failures should distinguish model termination from truncation, timeout, and postprocessing. Report retries and repairs as system behavior with their cost.

## Use paired regression evidence

Compare input IDs, fixed-prefix forward outputs, generated tokens, parsed results, then complete tasks. Once generated prefixes diverge, later logits no longer compare the same input.

In a constructed 100-task table, 70 succeed in both versions, 5 are repaired, 5 regress, and 20 fail in both. Both totals are 75%, hiding five regressions. Inspect failure slices and predeclare acceptance limits. A small passing suite does not prove equivalence.

The [Chinese article](/posts/2026-10-23-llm-training-eval-05-delivery/) proposes staged delivery checks and a package containing immutable identities, configurations, per-task results, known limits, and a recoverable reference version.
