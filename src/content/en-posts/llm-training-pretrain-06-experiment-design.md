---
sourceId: 2026-10-20-llm-training-pretrain-06-experiment
slug: llm-training-pretrain-06-experiment-design
title: "How to Run an Interpretable Pretraining Experiment: Baselines, Logs, and Resumption"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of smoke tests, tiny-sample fitting, held-out comparison, checkpoint resumption, and the cost accounting of a pretraining run."
tags:
  - Experimental Design
  - Evaluation
  - Pretraining
category: LLM Training Research
series: "LLM Training Research Notes · Pretraining Volume"
seriesOrder: 6
lang: en
---

> **Scope note.** This is an English overview of a Chinese technical note in the pretraining volume. The protocol described here has not been executed as a set of controlled research comparisons, and the levels of evidence it distinguishes are not results. The only measured artifact the source points to is an archived short engineering record of a 138,560-parameter model; that record supports engineering scope only, and it is not a rerun of this note.

## Four levels of evidence, and the inputs of a run

The temptation after the previous five notes is to pick a larger model, let it train overnight, and look at the result the next day. The problem is that a bad result still leaves it unclear whether the data, the code, the budget, or the model size was responsible. A good starting experiment is not "make everything big at once" but "let every step answer one clear question".

"Train a good model" is too broad to decide what to look at next. "Verify that this code really updates the parameters" is much narrower. The source splits the first round into four levels:

| Stage | Question it answers | What it still cannot show |
| --- | --- | --- |
| Engineering smoke test | Can it read, backpropagate, save, and load? | That the model learned language |
| Tiny-sample fitting | Can it learn a few fixed records? | That it works on new text |
| Held-out comparison | Does prediction on unseen data improve? | That the task requirement is met |
| Task check | Is it better under a specified task and conditions? | That every capability improved |

Low-level errors are excluded before a larger budget is spent. These are four different evidence requirements rather than four tidy labels, and a training loop that exits normally proves only a very small part of the first row.

For the Chinese technical-corpus scenario that runs through the series, a candidate research question is: under a fixed set of source documents and a fixed effective-target budget, does a particular cleaning change improve prediction on Chinese technical validation text without clearly harming the English part? The sentence fixes the change, the budget, and the breakdown, which makes it much more specific than "the cleaned data works better". Before any such comparison, the training pipeline itself has to pass the first two levels.

The data records established earlier have to be connected to a model configuration. At minimum, record the data version and split, the tokenizer file and its id mapping, the model structure and initialization, the context length, the packing and label rules, the optimizer and schedule, and the software and device information. Keeping only a set of weights makes it impossible to reconstruct how they were produced.

Controlled comparisons also have to distinguish "fixed" from "identical": two configuration files that look the same may order the data differently, and the same file path may hold rewritten content. Saving content checksums for important inputs helps detect such changes, although matching checksums do not prove that the data design is sound. The source suggests starting from a short record rather than building a platform first:

| Field | Meaning in the example |
| --- | --- |
| Research question | Change one cleaning rule and compare held-out prediction |
| Data identity | Training, validation, and test manifests with content checksums |
| Objective definition | Causal prediction, effective label rule, token averaging |
| Fixed items | Initialization, tokenizer, model, budget, evaluation procedure |
| Changed item | The cleaning rule under test this time |
| Failure criteria | Non-finite loss, inconsistent resumption, overlapping validation inputs |
| Conclusion scope | The observation under the current data, budget, and model |

The cleaning study in that example is only a candidate question. While the training code is first being checked, the data should stay unchanged and that variable should not be introduced at the same time.

## The three gates: from one batch to a held-out comparison

Take a very small batch and complete the following chain:

```text
document → token → inputs and labels → logits → loss
         → backpropagation → parameter update → save → load in a new process
```

Every arrow has something to check. After tokenization, check restoration and length. During alignment, inspect the prefix and the label item by item. In the forward pass, check the shapes and the causality. In the loss, check the effective count and the finiteness of the values. Across the update, compare whether the trainable parameters actually changed.

A parameter change does not mean the direction is right. For a stronger check, a fixed small example can be compared against an independently hand-computed loss or gradient direction; the three candidate scores introduced in the second note serve this purpose.

After saving, do not read the file back in the same process. Start a clean process, restore the configuration, the tokenizer, and the weights from disk, and compute outputs for a fixed input, because leftover in-process state can otherwise hide a problem.

Take two or three short texts and train on them repeatedly. The goal here is deliberate memorization of a small amount of data, so improvement on those samples cannot be claimed as generalization. Watch whether the training loss drops clearly, and confirm that the original objective was not rewritten. If difficult positions were removed from the loss mask, the average loss can also fall without the model predicting better. When this gate fails, check first whether the parameters are included in the optimizer, whether the gradients are detached, whether the effective target count is zero, whether the labels are shifted, and whether the learning rate is reasonable. Swapping in a different dataset should not be used to hide the problem.

Passing this gate does not require exact verbatim reproduction either. Sampling introduces variation, and the end marker and the context boundaries affect the result. Target probabilities, losses, and deterministic outputs under fixed conditions are easier to interpret, and generation quality is evaluated separately.

Before formal training, fix a validation set that does not update parameters. Keep the same tokenization, label, and context conventions during validation, and state the model's evaluation mode. Sum the effective losses over all batches and divide by the total effective target count, so that batches of different lengths are not weighted equally. Report Chinese prose, English material, and code separately, so the average cannot hide a regression in one category.

Three curve shapes call for different readings:

| Training behavior | Validation behavior | What to ask next |
| --- | --- | --- |
| Training loss falls | Validation loss also falls | Whether duplication or leakage explains it, whether the change exceeds run-to-run variation, and whether categories agree |
| Training improves | Validation is flat or worse | Overfitting, distribution shift, or objective mismatch |
| Neither improves | Neither improves | Whether optimization works, whether the budget suffices, whether the inputs are unusual |

The table gives diagnostic directions, not automatic causes. When the training and validation distributions differ, their absolute values need not relate to each other directly. The validation set can be used to select configurations, but repeated selection gradually adapts the research process to it. For a more trustworthy generalization result, keep a test set out of those selections, state how many settings were tried, and do not present the luckiest single run as a stable conclusion.

## Generation samples, cost accounting, and resumption

At every checkpoint, a fixed set of prompts can be used to generate text, recording the full output and the generation parameters. This surfaces termination, repetition, formatting, and local language behavior. But only the three prettiest samples should not be shown: if prompts were hand-picked, readers see a curated display rather than the model's average behavior. For a base model that has not been instruction-tuned, judging pretraining a failure because it does not answer like an assistant is also a mistake; continuation-style prompts fit the objective, and instruction following belongs to the supervised fine-tuning volume. For technical content, fluency and factual correctness are two separate measures, and without human review or executable verification a professional-sounding passage should not be recorded as factually correct.

Cost has to be measured in tokens and in time at the same time. Fixing the number of update steps is often not enough: a different effective batch means a different step, and a different tokenizer means the same token count can correspond to a different amount of raw text. A run record should therefore keep cumulative input positions, non-padding content, effective targets, the number of optimizer updates, and wall-clock time, and it should state the numerator and the timing window when efficiency is compared. For an assumed example, if a round processes 12,000 effective targets in 30 seconds, the effective-target throughput is 400 per second. If 10 of those seconds were first-time compilation or preparation, later steady-state training may behave differently, and both timing conventions should be reported rather than dropping the inconvenient part.

Peak memory also needs a named measuring tool. Allocated memory at one moment is not the peak of the whole process, and the highest resident size of a process is not the exclusive use of the current stage; on unified-memory devices, readings from several sources must not simply be added. A short run can help estimate whether continuing is worthwhile, but it cannot project long-run thermal behavior, swapping, or data-loading bottlenecks, and a small model's tokens per second should not be extrapolated by parameter count. Measure again before scaling.

Suppose training was interrupted after update 100. Reloading the model weights and continuing with a brand-new optimizer is not the same as resuming at update 101. The AdamW history, the schedule position, the mixed-precision scaling state, and the data reading position can all affect the next step, and the random number state affects dropout, sampling, and ordering. The PyTorch checkpointing tutorial states explicitly that a save intended for resumption should include the optimizer state, not just the model parameters.

A fuller checkpoint records, according to the actual pipeline: the model, optimizer, scheduler, and the gradient-scaling state where applicable; the completed updates, the consumed targets, the data permutation, and the reading position; the state of the random number generators in use; and the configuration, code and data versions, together with the save boundary.

Saving at the boundary of a completed parameter update is preferable. If the save happens halfway through gradient accumulation, without keeping the accumulated gradients and their count, the resumed run cannot be claimed to equal the uninterrupted one. Streaming data adds prefetch, buffer, and re-read problems, and "how many rows were read" may not be enough to restore exactly the next batch.

Two branches can validate resumption. From the same initial state, run A continues for $N+K$ updates, while B runs $N$ updates, saves, ends the process, then reloads and runs $K$. Compare the resumed data batches, learning rate, loss, parameters, and the necessary optimizer state, not just the final generated sentence. If the data batches already differ, fix the reading state before blaming floating-point error. In deterministic settings and the same environment, strict agreement can be required; PyTorch nonetheless notes that full reproduction is not guaranteed across versions, platforms, or devices, and that identical seeds are not a sufficient condition. For cross-device comparisons, define a tolerance and task-level acceptance in advance rather than declaring a failure on any bit difference, and rather than excusing any large difference as floating-point error.

## What the repository's evidence supports, and what a report should show

The site already contains a small-model project with a randomly initialized decoder, pretraining, supervised fine-tuning, preference training, checkpointing, and local generation. The archived validation record dated 2026-10-04 reports a short engineering run of a 138,560-parameter model: 12 pretraining steps, 8 supervised fine-tuning steps, 8 preference-training steps, and a pretraining resume to step 14. That is an existing record being cited, not a rerun of this note. Its inputs were a few synthetic fixtures, and its purpose was to verify backpropagation, target masking, save and restore, and cross-stage interfaces. It does not show that a 30M model has completed formal training, and much less that a small model trained from scratch has reliable assistant ability.

The existing skeleton also has scope limits: independent document chunking is not the same as full streaming packing, and a constant learning rate does not mean that the schedules discussed here are implemented. Check the actual source code before using it for an experiment. Such records still have value, because later work does not have to restart from "can the program run at all". The next stage, though, has to add real data, sustained resource measurement, and independent evaluation, instead of dressing up short-run numbers as a model-capability report.

A report need not be long, but it should answer five questions: what changed, what was held fixed, what actually ran, what the results were, and which explanations remain unexcluded. If a change improves the validation loss only slightly, first check the variation across repeated runs and the effective-token accounting. If a task score improves but the cost doubles, report the gain and the cost together. If the run failed, keep the failing conditions and the log locations, which can be more useful than a boundaryless recommended configuration.

The final conclusion can be narrow, for example: under this fixed data, this budget, and three initializations, change A gives a lower Chinese validation loss, the English difference is unstable, and longer contexts were not tested. That is informative enough without escalating into a claim that the model's capability improved across the board.

This volume establishes a basis for explaining the training process. Later supervised fine-tuning work can start from an open-weight model that already has capability instead of waiting for a home-grown small model to grow into a general assistant, as long as the data, objective, and evaluation conventions established here are carried forward.

## What stays in the Chinese original

The Chinese note carries the full experimental protocol, the run-record template, the resumption branch design, the cost accounting rules, the two review questions, and the reference list covering the PyTorch checkpointing tutorial, the PyTorch 2.9 reproducibility notes, and the archived project validation record with its permalink.

Main limit, restated: the protocol is a design, and the only measured artifact is the archived short engineering run of a 138,560-parameter model, which supports engineering scope and not model capability.
