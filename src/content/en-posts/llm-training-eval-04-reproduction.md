---
sourceId: 2026-10-23-llm-training-eval-04-reproduction
slug: llm-training-eval-04-reproduction
title: "Why Training Results Change: Precision, Data Order, and Resume Semantics"
editionLabel: ENGLISH OVERVIEW · EXECUTED TOY EXAMPLE
published: 2026-10-23T11:03:00+08:00
description: "Distinguish numerical variation from changed training semantics and test full-state checkpoint restoration."
tags:
  - Reproducibility
  - Checkpointing
category: LLM Training Research
series: LLM Training Research Notes · Evaluation and Reproducibility Volume
seriesOrder: 4
lang: en
---

> **Scope note.** A Python scalar momentum example and arithmetic counterexamples were executed. No language model, GPU, or distributed training was run.

## Define the required level of reproduction

Bitwise identity, numerical closeness, and reproduction of an experimental conclusion are different requirements. [PyTorch 2.14](https://docs.pytorch.org/docs/2.14/notes/randomness.html) does not guarantee complete reproducibility across releases or platforms. A seed alone cannot fix all variation.

Floating-point addition can depend on order: with Python floats, (10^16−10^16)+1 is 1, while 10^16+(−10^16+1) is 0. This exaggerated example explains a mechanism, not the expected size of training differences.

## Check semantics before blaming rounding

Record storage, multiplication, accumulation, gradient, and optimizer precision separately. Overflow skips, scheduling, distributed sampling, tail batches, and gradient normalization can change the actual update.

Two microbatches with target counts 10 and 30 and mean losses 0.2 and 1.0 have an unweighted mean of 0.6 but a target-token mean of 0.8. These are different objectives, not tiny numerical discrepancies.

## Restore more than weights

For the defined momentum recurrence v'=0.9v+(w−1), w'=w−0.1v', starting from zero gives w=0.28,v=−1.8 after two steps. Full restoration gives w=0.514 on the next step; resetting momentum gives 0.352. Changing the next target to 2 gives 0.614.

[PyTorch's checkpoint tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) includes optimizer state for resumed training. Real jobs may also require scheduler, scaler, random-generator, sampler, and data-consumption state. Save at complete update boundaries or explicitly restore unfinished accumulation.

## Compare uninterrupted and resumed execution

The companion test serializes the scalar checkpoint to JSON and resumes it, matching uninterrupted execution while detecting missing momentum and changed data. This does not validate any real framework's resume path.

For actual jobs, compare the first resumed batch, learning rate, gradients, and weights under declared tolerances. Preserve consistent distributed shards and distinguish prefetched data from data already consumed by an optimizer update. Loading successfully is not proof of equivalent continuation.

The [Chinese article](/posts/2026-10-23-llm-training-eval-04-reproduction/) outlines a staged investigation from fixed inputs through forward outputs, loss, gradients, and updates to full-run conclusions.
