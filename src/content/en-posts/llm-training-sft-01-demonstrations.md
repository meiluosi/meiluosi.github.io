---
sourceId: 2026-10-20-llm-training-sft-01-demonstrations
slug: llm-training-sft-01-demonstrations
title: "What Supervised Fine-Tuning Actually Teaches a Model"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of what supervised fine-tuning changes, using structured extraction from an experiment log as the running task."
tags:
  - Supervised Fine-Tuning
  - Instruction Tuning
  - Experiment Design
category: LLM Training Research
series: LLM Training Research Notes · SFT Volume
seriesOrder: 1
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on supervised fine-tuning (SFT). Its running example — the "Cedar" model and the experiment log it is extracted from — is constructed teaching material written by the source author, not a record of a training run. The article derives a loss formulation and proposes a comparison design. It reports no training log, no model checkpoint, and no benchmark score, and it does not establish that SFT improves any particular task. Every number below comes from the source's own illustrative counts, not from a measured run.

## What the task actually asks for

The article runs on one narrow task: given an experiment log, return three fields as JSON — `model` (string or `null`), `learning_rate` (number or `null`), `steps` (integer or `null`). A field the material does not state explicitly must be `null`, with no extra explanation.

The reference material is one sentence: a run used Cedar-30M at a learning rate of 2e-4, and the record does not state the number of training steps. The conforming demonstration is therefore:

```json
{"model":"Cedar-30M","learning_rate":0.0002,"steps":null}
```

The source separates the ways this can go wrong, because they need different evidence and different fixes:

| Output behavior | What is wrong |
| --- | --- |
| A paragraph explaining that the record used Cedar-30M | Content may be right, format does not follow the contract |
| Valid JSON, but `steps` filled with 10000 | Format passes; the model invented a fact the material does not contain |
| Learning rate written as 0.002 | Numerical conversion error |
| Every field set to `null` | Over-conservative; available information is dropped |

The second and fourth rows show opposite errors: inventing an unknown value and dropping known values, and a single accuracy number cannot distinguish them. Collapsing all four into "the model does not obey instructions" removes the signal that a data intervention would need.

## From continuation to demonstration, and what a demonstration teaches

The autoregressive objective from the earlier volume is unchanged in form. With input $x$ and demonstration answer $y=(y_1,\ldots,y_m)$, the conditional probability of the whole answer factorizes as

$$
p_\theta(y\mid x)=\prod_{t=1}^{m}p_\theta(y_t\mid x,y_{<t}),
$$

and supervising only the answer gives the per-example negative log-likelihood

$$
\begin{aligned}q_t&=p_\theta(y_t\mid x,y_{<t}),\\\ell(x,y)&=-\sum_{t=1}^{m}\log q_t.\end{aligned}
$$

The difference from pretraining is what counts as the target and how much of the sequence carries a target. Here $x$ includes the serialized prompt and the necessary control tokens, not just the natural-language material a reader sees. When several examples are combined, one further decision is required — averaging per token or per example — and the source deliberately defers that to the third note in this volume, keeping only the single-example sum here.

One consequence is worth stating plainly. The model is not first generating a complete JSON object and then receiving a "correct/incorrect" score. Basic SFT uses per-token supervision on the demonstration. Whether the JSON parses is a task check the article designs separately, not a business rule that cross-entropy learns by itself.

### What a demonstration teaches at once

For the sample above, a demonstration can carry several signals at the same time:

- **Field extraction** — locating the value that belongs in each field.
- **Format adherence** — the key names, braces, and value types the contract requires.
- **Missing-value handling** — emitting `null` when there is no evidence.
- **Output boundary** — stopping after the answer instead of adding a note or a new question.

These signals are not cleanly separable. If every example opens with a long fixed preamble, much of the supervision budget goes into learning that preamble. If missing values always co-occur with very short inputs, the model can learn "short text" as a shortcut for "value is absent".

A demonstration is therefore not finished when the right answer is written into a file. It defines what the model imitates, and it can contain regularities nobody intended to teach. SFT can teach new information as well as new patterns, so it cannot be reduced to "it only changes format". But if the goal is to use the input material reliably, memorizing the numbers in the training set is not task completion: a model that still extracts correctly from a record with a different model name and learning rate is behaving the way the task intends.

### Why published results do not transfer automatically

The InstructGPT method has stages: demonstrations for SFT first, then a reward model, then PPO. The paper discusses those models and comparisons separately. Reading the end result of the full pipeline and concluding that SFT alone yields the same benefit inverts the design ([InstructGPT, Section 3](https://arxiv.org/html/2203.02155v1#S3)).

A second line of work writes many tasks as natural-language instructions and then fine-tunes, studying how that training transfers to unseen tasks ([FLAN](https://arxiv.org/abs/2109.01652)). What transfers from it is a question about designing the task distribution of instruction data, not a claim that any mixture of tasks improves any objective.

For a small extraction task, what is worth borrowing is the question design and the comparison structure: which behaviors do the demonstrations cover? Does the evaluation include unseen templates and unseen materials? Did the model gain a more stable format, or did content accuracy improve as well? Replicating a paper's data scale and epoch count is not a precondition for answering a small, well-posed question.

## Whether to train at all, and how to settle ambiguity

Before collecting data, check whether the input format is fixed and the fields always sit in the same position; a rule-based parser may already be enough. Even if a model is used, first measure what the base model achieves with a clear prompt.

The source proposes three arms, and it is careful about what they isolate:

| Arm | What changes | What it answers |
| --- | --- | --- |
| Base model + explicit instruction | No parameters change | Whether existing capability already suffices |
| Base model + few-shot demonstrations | Examples added to the context | Whether the task is merely underspecified or still unstable |
| SFT model + fixed prompt | Parameters updated on training examples | Whether a parameter update improves the task within budget |

These are not a clean single-variable experiment by construction. Adding demonstrations changes context length, and SFT adds training cost. A report should say explicitly whether it is comparing deployable options or isolating the effect of the parameter update. To study SFT alone, keep at least the base model and the SFT model on the same prompt, the same template, and the same generation settings; a few-shot arm can be an extra baseline, but not the place where every difference is attributed.

Ambiguity in the specification multiplies with more examples. "Learning rate" can be ambiguous: if a record says the peak learning rate was 2e-4 and it has decayed to 1e-5, which value belongs in the field? That is not common sense the model must guess; it is a specification that has not been written down. If half the annotators pick the peak and half pick the current value, training receives contradictory demonstrations.

The choices are to extract only a value explicitly labeled as this run's learning rate, or to split the field into peak and current. Either way the decision comes first, and the description, the demonstrations, and the evaluation change together. Accepting both answers during training while recognizing only one at evaluation is not a workable arrangement. For the simple version used in this volume, each material describes exactly one experiment, and records with multiple candidates that cannot be resolved by a rule go into a separate ambiguity set rather than being forced into the normal samples.

## Falling loss is not field correctness

Cross-entropy rewards the probability of the demonstration sequence, but field errors and token errors do not map one to one. Writing 0.0002 as 0.002 may differ by very few tokens and still be a serious numerical error; changing the key order produces a different string without changing the structured content.

The source therefore separates three measurements:

1. **Parse rate** — is the output valid JSON?
2. **Structural conformance** — are the keys, types, and absence of extra text as specified?
3. **Field accuracy** — is each value supported by the material, and is absence handled correctly?

The article's own illustration: of one hundred examples, ninety-eight parse and seventy have every field right. The 98% is not the task success rate, because the application needs the whole record. That count is a hypothetical example written to make the denominator visible, not a score achieved by any model here.

For an extraction task, validation loss helps observe fit, while task metrics answer the question of actual use. When the two disagree, save the outputs and classify the error type instead of picking whichever number looks better.

### Designing the first comparison

Prepare a small batch of manually checked examples covering ordinary values, scientific notation, missing fields, and format variations. Split by source group so that several rewrites of one material never straddle the training and test sides. Fix the model and tokenizer versions, keep the base-model result, and train one SFT baseline with an explicit configuration. Do not vary the LoRA rank, the data generator, and the learning-rate schedule at the same time; a result would then not be attributable.

Write the prediction down first. A plausible one is that SFT reduces extra explanation and improves `null` handling, while numerical conversion may or may not improve; count format errors, omissions, fabrications, and conversion errors separately to match. If the model improves only on the training template and degrades on rephrased input, the next step is coverage and split inspection, not more epochs. And if the prompted base model is already stable, fine-tuning may carry no maintenance cost worth paying.

This is an executable comparison design, not a report of a completed training run.

## The Chinese original and the limits of this note

The full Chinese article carries the JSON examples, the error taxonomy tables, the derivation of the per-example loss, the three-arm comparison plan, the reference list with links to InstructGPT and FLAN, and two comprehension questions about attributing gains to parameter updates — material summarized but not reproduced here. The original is at [the Chinese source article](/posts/2026-10-20-llm-training-sft-01-demonstrations/); the next note in this volume covers [what one instruction sample must contain](/en/posts/llm-training-sft-02-messages-and-templates/). The main limit is unchanged: no fine-tuning was run for this volume, and nothing reported in either version is a measurement of a trained model.
