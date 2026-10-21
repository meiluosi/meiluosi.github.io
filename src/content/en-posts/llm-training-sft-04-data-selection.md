---
sourceId: 2026-10-20-llm-training-sft-04-data
slug: llm-training-sft-04-data-selection
title: "Choosing and Generating Instruction Data: Coverage, Review, and Synthetic Examples"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of task coverage, answer review, synthetic data pipelines, and deduplication for supervised fine-tuning instruction data."
tags:
  - Instruction Data
  - Supervised Fine-Tuning
  - Data Curation
category: LLM Training Research
series: LLM Training Research Notes · SFT Volume
seriesOrder: 4
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on selecting and generating instruction data for supervised fine-tuning (SFT). Its Cedar model, experiment logs, and JSON payloads are constructed teaching material written by the source author, and the article states explicitly that generated examples are not treated as research facts. It proposes a coverage table, a three-layer review procedure, and a data comparison design; it reports no generated corpus, no training run, no filter pass rate, and no measured fine-tuning gain. Any count or rule described below is a proposed design, not an observed result.

## Enumerate the situations before collecting volume

Once the trainer handles answer supervision correctly, the obvious next step is to collect more question-and-answer pairs. Quantity matters, but replicating one simple sample ten thousand times teaches nothing about handling missing values, ambiguity, or new phrasing.

For this volume's task — extracting the model name, learning rate, and step count from experiment material — the real question is whether the training samples cover the situations that will actually occur, and whether the answers are reliable.

**Instruction data quality has to be judged against the task requirements.** Fluent language, detailed answers, and provenance from a larger model are each insufficient on their own as evidence of fitness for the current supervision target.

A **task coverage table** is the starting point:

| Dimension | Variation to cover | Failure it should expose |
| --- | --- | --- |
| Information completeness | All three fields, one missing, two missing | Fabrication when a value is absent, or blanking a value that is present |
| Numeric representation | Decimals, scientific notation, spelled-out numbers | Numeric conversion errors |
| Text organization | One sentence, several paragraphs, a table flattened to text | Recognition of only a fixed sentence pattern |
| Distracting content | Background proposals, an older configuration, explanatory paragraphs | Taking another experiment's values as the target |
| Information boundary | Not stated, explicitly zero, ambiguous candidates | Confusing absence with zero |
| Input length | Short material, longer material | Key fields being truncated or ignored |

This is not a requirement to fill every combination immediately. Choose the range the task must support, mark what is not supported yet, and then prepare checkable samples for each category. The first article of this volume already put records whose candidate values cannot be told apart by a rule into an ambiguity set. A task cannot claim to handle only a single unambiguous experiment while its main metric is scored on multi-experiment conflicting samples without stating a selection rule. The coverage table's value is that it makes "what is not supported" concrete instead of letting a single overall accuracy figure hide the boundary.

### One missing field needs more than one way of being missing

If every sample with a missing step count says "the record does not state the number of training steps", the model may only memorize that phrase. When real material simply does not mention the step count, it starts filling one in again.

The same fact can be expressed in several ways: omitted outright, described as unrecorded, or accompanied by other fields while never mentioning steps, and the output should stay `steps:null` in all of them. Explicit zeros should be added at the same time. If a record states "initialization only, the number of training updates is 0", and the field is defined as completed updates, then zero is a known value, not an absence.

Such paired comparisons test the rule directly: only whether the value is explicitly recorded should change, and only the corresponding field should move. All variants of one original material must still share a source group and land on the same side of a split; otherwise training has seen nine of them and the test set holds the tenth, which inflates the apparent transfer.

## Answer review has at least three layers

The first layer is **format review**: can it be parsed, are the key names and types as required, is there no extra text.

The second is **semantic review**: was scientific notation converted correctly, were fields confused, is absence distinguished from zero.

The third is **evidence review**: is every non-empty value in the answer actually supported by the current material.

For example, the answer below parses as JSON:

```json
{"model":"Cedar-30M","learning_rate":0.002,"steps":10000}
```

If the material says `2e-4` and states no step count, two fields are wrong. A format checker will not surface either error on its own.

Where a check can be mechanical, use a deterministic rule: numeric parsing can verify that `2e-4` and `0.0002` correspond. Absence, multiple candidates, and coreference may still need human review or a stronger semantic check.

One practice to avoid: letting one model generate an answer, freely explain why that answer is correct, and then treating the explanation as independent evidence. Generation and self-endorsement can share the same error.

## Synthetic data is a production process, not a provenance label

Self-Instruct studied the process of generating instructions and instances from seed tasks, then filtering them before fine-tuning. Its method includes similarity filtering and removal of invalid instances, which shows that generation has to be followed by selection; it does not mean the surviving answers have been factually verified ([Self-Instruct, Section 2](https://arxiv.org/html/2212.10560v2#S2)).

For an extraction task, the source distinguishes two flows:

| Flow | How it generates | Main risk |
| --- | --- | --- |
| Material first, then answer | Derive the extraction from existing material | The model may misread the material or add information |
| Structure first, then material | Fix known fields, then write text containing them | The text may not faithfully express the original structure |

The second flow appears to come with a ground-truth answer for free, but it still needs review. If the learning rate is fixed at 0.0002 and the generated text writes 0.002, the original structure is no longer the correct answer for that material.

A steadier starting point is to generate the explicit field records first, render a batch of materials under bounded rules, and then gradually introduce more natural rewrites. After rewriting, the field-support relation has to be re-checked rather than assuming the original structure still holds.

This flow suits building teaching data with clear boundaries; it does not cover the full complexity of real experiment logs, and real samples remain irreplaceable.

### Direct the generator at the weak spot

Suppose the current model handles complete short sentences well but often fills an unrecorded step count with a default value. The next batch should concentrate on absence and distraction rather than adding more complete three-field short sentences.

There is still a ratio to control: if a training round is almost entirely absence samples, the model may pivot to a shortcut of filling everything with `null`, so the coverage table should record sample counts, effective answer tokens, and sampling proportions.

The human review budget should be allocated by risk. New templates, numeric conversions, conflicting candidates, and long material deserve a higher sampling rate. A pass rate has to be reported together with the sampling method, because the error rate of a deliberately over-sampled hard subset is not the error rate of the whole corpus. If the goal is an overall quality estimate, use a separate representative sample; if the goal is finding the most dangerous errors, inspect a targeted subset. Both goals can coexist, but their statistics mean different things.

## Deduplication, versioning, and proving that added data helped

Training the same question and answer repeatedly increases its weight, and near-duplicates that differ only slightly in format can let one template take most of the training opportunity. That does not license deleting every similar input. Two materials may differ only by `0` versus "not recorded", and that difference is exactly what teaches the model to separate a zero from an absence.

Deduplication therefore has to combine **source grouping** with task difference: mechanical rewrites of the same source should not straddle a split, while contrastive samples with different key facts should be kept deliberately. Answer conflicts need checking too. If one input maps to two different values, decide first whether the specification allows multiple answers or whether the annotation is wrong; keeping one at random does not make the data clean.

Data versions should explain every change. A record should link back at least to its original source, the task specification version, the generation method, the generator version or identity, the review result, and its split group. Merging everything at the end into a file with only `prompt` and `answer` destroys that information. Metadata is for tracking and must not be spliced into the model prompt; fields such as "the answer has been reviewed" or "test set" must not become visible shortcuts. When a class of errors is found, the version record should say which samples were removed or corrected, whether category proportions changed, and whether the related token files have to be regenerated. Renaming a data file is not enough to reproduce an experiment.

Proving that added data helped starts from three comparisons against a fixed baseline: the original human-reviewed samples, the same set plus reviewed targeted synthetic samples, and the same set plus generic synthetic samples of the same budget. That asks not only whether more data helps but whether targeting a specific error class is more effective. The budget should be fixed in advance on a stated measure, such as effective target tokens, and the number of independent sources and the repetition count should be reported alongside it. With a fixed number of training epochs, more data also means more computation, so the result contains a budget effect; that is not an invalid experiment, and the conclusion just has to be phrased in terms of the conditions actually compared. Evaluation should use material from independent sources and include the part that already worked. If absence handling improves while complete-field accuracy drops, that is a trade-off, and only reporting the first half misrepresents it.

## The Chinese original and the limits of this note

The full Chinese article carries the six-row coverage table, the worked review example with the wrong `0.002` and `10000` values, the two-flow generation table, the review-budget and deduplication rules, and the reference list with the Self-Instruct link — material summarized but not reproduced here. The original is at [the Chinese source article](/posts/2026-10-20-llm-training-sft-04-data/); the next note covers [choosing between full fine-tuning, LoRA, and quantization](/en/posts/llm-training-sft-05-adaptation/). The main limit is unchanged: no dataset was generated, no filter was measured, and no training run shows that any of these data choices improved the task.
