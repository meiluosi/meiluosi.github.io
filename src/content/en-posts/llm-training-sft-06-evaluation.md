---
sourceId: 2026-10-20-llm-training-sft-06-evaluation
slug: llm-training-sft-06-evaluation
title: "Deciding What SFT Improved: Format, Accuracy, and Capability Regression"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of layered evaluation for supervised fine-tuning: format adherence, field accuracy, paired changes, and capability regression checks."
tags:
  - Supervised Fine-Tuning
  - Evaluation
  - Experiment Design
category: LLM Training Research
series: LLM Training Research Notes · SFT Volume
seriesOrder: 6
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on evaluating supervised fine-tuning (SFT). The article says at the outset that every number in it is a self-written counting example used to explain metric definitions, not the result of a trained model. It reports no training run, no evaluated checkpoint, no benchmark score, and no model-judge score. The Cedar extraction task and its logs are constructed teaching material. The paired counts, the 98/94/78 illustration, and the all-null field accuracy figure illustrate how to define and read metrics; they are not measurements of anything that was run.

## Three layers, from parseable output to a usable record

### First layer: can the output be read at all

First, check whether the complete output is valid JSON, rather than extracting a pair of braces from the middle of an explanation. If the application requires JSON only, an extra opening paragraph is itself a format failure.

Then check the structure: it must be an object, contain the three specified keys, have the right types, and carry no disallowed extra fields. `steps` should be an integer or `null`, `learning_rate` a number or `null`.

Language-level details matter here. In Python a boolean inherits from integer, so a naive integer type check can accept `true` as a step count. The checker should follow the task definition rather than letting a programming language's default behavior define the specification.

Parsers can also be lenient about duplicate keys and non-standard numeric forms, so the acceptance range used for evaluation should be stated and the checker should be tested with both normal and abnormal outputs. **Parse rate** and **structural conformance** can be reported separately; a high parse rate does not guarantee a high conformance rate.

### Second layer: were the fields extracted correctly

Once the structure passes, compare field meaning. For material written as `2e-4`, an output of `0.0002` should count as the same value, and a change in JSON key order should not change the result.

Which normalizations are permitted has to be specified in advance: whether model-name capitalization, whitespace, or unit conversion may be ignored cannot be loosened after seeing the model's answers. Numeric comparison should not use one very wide tolerance everywhere, since a learning rate moving from 0.0002 to 0.002 is an order-of-magnitude error that an unsuitable absolute tolerance could mark correct. Use exact decimal parsing for the task, or state the relative and absolute error rules explicitly.

Field correctness is judged against the current material and the task specification; outside knowledge about how many steps a given model usually trains for cannot replace a value the material does not record.

### Third layer: is the whole record usable

The third layer asks whether the whole record is usable. Suppose that of one hundred outputs, ninety-eight parse, ninety-four conform structurally, and seventy-eight have all three fields correct. Those are three different metrics: 98%, 94%, and 78%.

If the application needs a fully correct record, 98% cannot be presented as the task success rate. **Whole-record success rate** and **per-field accuracy** can be reported together: the former is stricter, the latter helps locate whether the error came from the model name, the learning rate, or the step count.

Failed parses must stay in the overall denominator, because computing accuracy only over the ninety-eight outputs that parsed lets failures disappear from the report. Error types can also co-occur on one record, so the counts per error type may overlap and need not sum exactly to the number of failed records; a mutually exclusive classification requires a separately defined priority rule.

## Missing values, failure modes, and loss versus task success

### When missing values dominate, overall field accuracy misleads

Suppose all one hundred records clearly contain a model name, while the learning rate and the step count are both missing. There are three hundred fields in total, and two hundred of them should be `null`. A system that always emits three `null` values reaches a field accuracy of $200/300\approx66.7\%$, yet not one record is completely correct and no model name was ever extracted.

Absence handling therefore needs at least two sides:

| Metric | What it observes | Failure it corresponds to |
| --- | --- | --- |
| Accuracy on known fields | Whether a present value is given correctly | Omission, blanking, miscopying |
| Absence rate on missing fields | Whether an absent value is left empty | Writing something without support |

One can also report the fabrication rate among missing fields and the omission rate among known fields, with the denominator stated explicitly as either the number of missing fields or the number of all fields.

This split explains what a data change did: if adding many absence samples reduced fabrication but increased omissions, the behavior traded one error for another, which is not an unconditional improvement.

### Separating cannot-answer, unclear, and refusal

Different error sources also need different names. In this volume's task, `null` means the material does not state the value; it does not mean the model refused to do the task, and on ordinary material with clear values, setting every field to `null` is a task failure.

If an input contains several experiments and the specification does not say which is the target, it should enter an ambiguity-handling path rather than letting the evaluator quietly pick one as the single answer. If a passage in the material instructs the model to ignore the task or output something else, the model should be tested on whether it keeps processing the material by the agreed contract.

The range of inputs that should be refused or excluded belongs in the task specification; a single blended "refusal rate" cannot represent them. For broader assistant tasks, safety boundaries and helpfulness on ordinary requests need their own evaluation, but such general requirements should not be mechanically transplanted into a field-extraction task, where they would create irrelevant metrics.

### Why validation loss and task success can diverge

Validation loss and task success can diverge. The supervision objective averages over many tokens, so large amounts of punctuation, fixed key names, and common text become easier to predict, which is enough to lower the average loss, while a critical number inside a field can still be wrong.

In the other direction, a demonstration supplies only one way of writing a thing, so an output that is semantically equivalent but tokenized differently may be correct under the task check and still not match the per-token objective exactly.

Loss is suited to observing fit and numerical behavior, while a fixed task evaluation answers the application question more directly; neither may fully replace the other. When generating for evaluation, fix the template, the input handling, the maximum output length, and the sampling settings, because if one configuration has too short a generation limit the JSON is cut off and the difference may come from the inference configuration rather than from the training method.

## Eight percentage points: which samples were repaired

Suppose the base model succeeds on seventy of one hundred records and the new model on seventy-eight. The total shows a net gain of eight, but it does not show whether new errors appeared.

Pairing the records one to one can reveal: both succeed in sixty-six, previously failing and now succeeding in twelve, previously succeeding and now failing in four, and both failing in eighteen. The new model's 78 comes from $66+12$ and the old model's 70 from $66+4$.

The net gain is eight, but the four regressions still deserve inspection, especially whether they concentrate on an important boundary.

Differences in small samples can also be unstable. Repeat or quantify uncertainty over independent units; ten rewrites of one source are not ten fully independent samples, and treating them as such produces an over-optimistic interval.

Evaluation-sample uncertainty and training-seed variance are two different things. Resampling a fixed model on the test set does not replace repeating with several training seeds, and several training seeds do not repair a test set drawn from a single source.

## Unseen splits, model judging, and regression checks

### Unseen templates and unseen sources should be measured separately

A random split of question-and-answer pairs can put rewrites of the same original material on both sides. Good performance then does not show that the model handles new experiment logs.

Several sets of different difficulty can be kept: new sources for the same task, unseen phrasing templates, longer material, and boundary samples containing key distractors. Each set keeps an explicit specification and is reported separately.

A "hard set" does not necessarily represent the real traffic distribution; it exists to find fragile points, and reporting overall use still requires an evaluation set that matches the actual input distribution. Once a hard set has been tuned against until the results look good, it has become a development set and should no longer be called an unseen test, so separate material must be kept for the final check.

### What model judging can add

For field extraction, JSON and deterministic numbers can usually be checked directly, and there is no need to hand every judgment to another model. Model judging is more useful where there is complex coreference, a material-support relation, or a more open-ended answer.

But a judge model is affected by phrasing, length, position, and its own preferences. The research on MT-Bench and Chatbot Arena discusses both such judging capability and such bias, so a judge score is not error-free ground truth ([Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).

Rule-verifiable instruction evaluation is a separate direction. IFEval turns a class of instruction requirements into checkable constraints, which helps separate formal compliance from open-ended quality judgment; passing such constraints does not mean the content is factually correct ([Instruction-Following Evaluation](https://arxiv.org/abs/2311.07911)). When a judge is actually used, keep the judge prompt, the model version, the input order, and the raw judgments, and sample for human review. If a judge conflicts with a deterministic check, investigate the reason instead of averaging the disagreement away.

### Do not check only the target task

After SFT on an extraction task, the model may become more inclined to answer briefly in JSON, and it may also start emitting fields during ordinary explanation tasks. Whether such a change is acceptable depends on what the model is for.

If it is a dedicated extractor, a general-chat regression may not affect the deliverable; if it is still a general assistant, a regression set should cover basic explanation, language understanding, and other capabilities that were promised. So define the product boundary first and then decide the regression set.

No model has to remain unchanged on every task, but a target-task gain does not cancel capabilities that were already committed to. When comparing LoRA against full fine-tuning, the same regression standard applies; different options must not be judged on different favorable tasks.

## What a report should contain, and the Chinese original

A reviewable SFT report can include: the fixed task specification and model version, the training data and splits, the template and supervision rules, the actual training configuration, the inference settings, the layered metrics, paired changes, and typical errors.

Typical outputs should cover repairs, regressions, and persistent failures, not only the best cases. On resources, report actual time and the measurement method, and do not present a low-rank parameter share as a cost result.

If the experiment has not been run, deliver the evaluation scripts and the plan, and state clearly what is missing. This article completes the evaluation design and the counting explanation; it does not invent a training result for this volume.

The full Chinese article carries the three-layer evaluation procedure, the worked 98/94/78 example, the 200/300 all-`null` analysis, the paired change table with its arithmetic, the judge-model discussion with the MT-Bench and IFEval references, and the reporting checklist — material summarized but not reproduced here. The original is at [the Chinese source article](/posts/2026-10-20-llm-training-sft-06-evaluation/), and the first note in this volume returns to [what SFT is being asked to teach](/en/posts/llm-training-sft-01-demonstrations/). The main limit is unchanged: every count here is a self-written illustration of a metric definition, and no model was trained or evaluated to produce any of them.
