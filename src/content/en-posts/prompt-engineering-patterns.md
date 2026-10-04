---
sourceId: 2024-09-29-提示工程最佳实践
slug: prompt-engineering-patterns
title: Prompt Engineering Patterns and Their Limits
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-09-29T00:00:00.000Z
description: A practical overview of prompting patterns, iterative evaluation, and the limits of the results reported in the Chinese guide.
tags:
  - Prompt Engineering
  - LLMs
  - Evaluation
  - AI Applications
category: NLP
lang: en
---

> **Scope note.** This is an English overview of a Chinese prompt-engineering tutorial. It summarizes the techniques and code patterns discussed in the source. The article includes a GSM8K accuracy table, but does not give enough model, prompt, sampling, or evaluation details to reproduce the comparison; the values are therefore not presented here as verified results. No prompt pattern is universally best across models and tasks.

## Make the task and expected output explicit

A useful prompt states the task, provides relevant context, and describes the expected answer format. Role instructions can establish a useful point of view; examples can demonstrate the desired input-output pattern; constraints can narrow the answer. These directions make the request easier to interpret, but they do not guarantee correctness. Important facts and output requirements should be checked independently.

The source distinguishes zero-shot prompts, which rely on instructions alone, from few-shot prompts, which add examples. Examples are especially useful for demonstrating a format or convention. They can also bias the model toward their mistakes or toward a narrow pattern, so they should reflect the intended range of inputs.

## Ask for structured work when a task has parts

The tutorial describes chain-of-thought (CoT) prompting and zero-shot variants that ask a model to work through a problem step by step. For practical use, it is often more useful to request an answer structure—such as assumptions, calculation, and final result—than to treat a fluent explanation as proof that the answer is correct. An explanation can be persuasive and still contain an error; verify important intermediate claims or calculations with appropriate tools.

The source also discusses **self-consistency**: sample multiple candidate solutions and choose the most common extracted answer. This can reduce sensitivity to one sampled response on some tasks, but adds inference cost and depends on reliable answer extraction. Agreement among samples is not independent evidence of truth, particularly when the samples share the same model and prompt.

## Combine reasoning with tools where useful

The guide presents ReAct-style prompting, which alternates between deciding what to do, taking an action through a tool, and observing its result. A calculator, search system, or code runner can supply information that text generation alone may not reliably provide. The application should validate tool names and inputs, handle failures, and treat tool output as data to inspect rather than instructions to follow.

For tool-using prompts, make the available actions and their input formats explicit. Keep the model's proposal separate from the application's execution of that proposal. This makes it possible to apply permissions, validation, limits, and logging before an external action occurs.

## Iterate with evaluation, not intuition alone

Prompt development can be treated as an experiment: define a representative set of cases, change one part of the prompt, compare outputs against a consistent rubric, and inspect failures. The article introduces A/B testing and prompt-optimization ideas, but its GSM8K table lacks the details needed to reproduce its values. It also makes a broad performance-improvement claim that is not supported by a specified benchmark in the article. Neither that claim nor the table should be treated as a general guarantee.

An evaluation set should include ordinary cases, edge cases, and requests the system should refuse or clarify. Record both quality and operating cost, such as latency and tokens. If a prompt is optimized repeatedly against a small fixed set, performance may simply overfit to those examples; keep separate cases for follow-up evaluation.

## Choose patterns for the task

The source surveys prompting for coding, data analysis, and creative writing, along with role, format, contrastive, and meta-prompting patterns. The practical choice depends on what is failing:

- If outputs are inconsistent in shape, add a clear schema or a concise example.
- If a task is underspecified, ask for missing inputs or make assumptions explicit.
- If factual answers need current or private information, connect retrieval or another trusted data source and check citations.
- If arithmetic or deterministic transformations matter, use a suitable tool and validate its output.
- If a prompt change appears helpful, compare it on the same cases and against a baseline.

More prompt text is not automatically better. Long instructions can compete with one another, consume context, and make failures harder to diagnose. Keep the prompt focused on requirements that affect the task.

## Connection to RSI and AGI

Prompt optimization can be one component in a model-improvement loop: propose a change, evaluate it on a defined task set, and retain it only if it improves the chosen objective without unacceptable regressions. The improvement claim depends on the quality and independence of the evaluation. A better prompt on a narrow benchmark is not, by itself, evidence of broad capability growth or recursive self-improvement.

For the full Chinese tutorial—including zero-shot and few-shot examples, CoT variants, self-consistency, ReAct, task-specific patterns, and prompt evaluation code—see the original article.
