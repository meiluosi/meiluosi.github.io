---
sourceId: 2026-10-20-llm-training-preference-01-feedback
slug: llm-training-preference-01-feedback
title: "Demonstrations, Preferences, Ratings, and Rules: What Each Feedback Form Commits To"
editionLabel: ENGLISH OVERVIEW · CONCEPT NOTE
published: 2026-10-20T00:00:00.000Z
description: "How demonstrations, preference pairs, ratings, and rule checks differ as training signals, and why a preferred answer is not a correct one."
tags:
  - Preference Learning
  - Human Feedback
  - Annotation Design
category: LLM Training Research
series: LLM Training Research Notes · Preference Volume
seriesOrder: 1
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on the meanings of different feedback forms. Its running example, a "Cedar" model whose successes on a 100-item validation set change from 70 to 78, is teaching material written by the source author, not a record of a training run; the counts are illustrative arithmetic inside a constructed prompt, not measurements. No preference data was collected, no annotators were recruited, and no model was trained. The article does not establish that any preference signal improves any task.

## Where one correct answer does not exist

The previous volume in this series covered extracting fixed fields from experiment records. This article turns to the more open question: given a record, explain what the experiment can support.

One answer states a conclusion outright; another explains the evidence first and then states the limits. Both can be free of factual error, and each has its own use. There is no single standard answer to write down here, and yet one of the two is often clearly the more appropriate for the task at hand.

A preference signal expresses which answer one would rather have, under a given condition and a given standard. That is a useful learning signal, but it does not automatically equal factual truth, general quality, or a view shared by everyone.

The source opens with a task whose answer cannot be reduced to a single key. The material reads:

> Cedar model on the same validation set: the original approach succeeded on 70 items, the new approach on 78, out of 100. Only one run was recorded, with no independent test set result and no timing record.

The question is: "What can be concluded from this change? Answer briefly."

Response A: "On this validation set the success rate of the new approach rose by 8 percentage points. But there has only been one run, so it is not yet possible to tell whether the gain is stable, nor to compare cost."

Response B: "The new approach is comprehensively better than the old one; this proves the method can stably improve model capability, and it is better suited to deployment."

Given the material, A fits and B does not. B adds a stability conclusion and a deployment conclusion with no evidence behind them. The difference between them is not which answer reads better, but whether the conclusion is supported by the evidence.

Response C adds a step: "This validation result went from 70% to 78%, a net gain of 8 percentage points. I recommend adding repeated runs, an independent test, and timing records before deciding whether to adopt it."

A and C can both be reasonable. If the user only wants the conclusion that is already supported, A is more direct; if the user also wants the next step, C is more useful. A preference criterion has to carry that task intent, which is why "pick the better one" is not yet a criterion.

## Four feedback forms, and what each one leaves unsaid

| Feedback form | What it states directly | What it does not state |
| --- | --- | --- |
| Demonstration answer | How the model should answer | Why the other answers are worse |
| Pairwise preference | Which of these two answers the annotator prefers | How much better it is, or whether it meets a deliverable standard |
| Multi-point rating | An evaluation of several aspects on a given scale | That different raters' scales agree |
| Rule feedback | Whether executable conditions are met | Any quality outside those conditions |

In [SFT](/posts/2026-10-20-llm-training-sft-01-demonstrations/), one answer worth imitating is supplied and the model mainly learns to raise the probability of that answer under the given condition. In pairwise preference, two answers to the same input and the relative choice are supplied; the better answer need not be perfect, and the worse answer need not be wrong in every sentence.

The forms can be combined, but they cannot substitute for one another without a design. An answer that passes a formatting rule may still be factually wrong. The better member of a pair may still be unfit as an SFT demonstration. A demonstration supplies a target and a preference supplies a relation between two candidates, which is why the article treats them as different signal types rather than as points on one scale.

The volume's ordering starts here, because the meaning of the signal has to be fixed before any data format is chosen.

## A criterion, a floor, and the reach of rule checks

For the "explain an experimental conclusion" task, the source writes the order of judgment down explicitly: first check fidelity to the material and the absence of unsupported assertions; only then compare whether the question was answered, whether the explanation is clear, and whether the brevity requirement is met.

This is a criterion for this task, not a permanent ranking for all tasks. Its value is that it gives a definite basis for comparing "short but wrong" against "slightly longer but accurate".

Scoring accuracy, brevity, and politeness five points each and summing them allows a factual error to be offset by other high scores. Whether such compensation is permitted is a design decision, and it should not be left to a total formula written on the spot. One workable starting point is to keep two separate records: a factual-support check and an expression preference. Answers that clearly fail factual support are marked first with their problems identified, and expression is compared only among answers that meet the baseline. How training consumes these two records is decided later.

Rule checks help the comparison, and their reach has to be stated. In this example, "70 to 78 is 8 percentage points" can be checked mechanically, and "there is no timing record, so speed cannot be compared" also has a clear basis in the material. Whether an answer has stated its key limitations clearly, however, takes semantic judgment. Passing a few rules does not justify labeling the whole answer as correct, and arithmetic that can be checked mechanically should not be handed over entirely to a style preference. A fluent answer with a wrong number still has to be called out.

The design goal is a division of labor: deterministic conditions get an explicit check, open trade-offs get a comparison against clear criteria, and the part no check covers keeps its boundary.

## What the label certifies, and what ties and uncertainty add

Suppose both answers turn "8 percentage points" into "improved 8 times", and one of them additionally appends a runtime conclusion. It is reasonable to prefer the one that is less bad, but it is still wrong.

If "both are unacceptable" is not recorded, then exporting `chosen` directly as SFT data later teaches the model to repeat the error. Comparison records should therefore keep an absolute-acceptability field or an error flag. Pairwise ranking and deliverability are two dimensions, and a record that keeps only the first one loses the second.

InstructGPT separates demonstrations, comparison data, and subsequent optimization into distinct stages, and its alignment objective depends on the annotator population and the specification that were used rather than standing in for everyone's values abstractly ([InstructGPT methods and discussion](https://arxiv.org/html/2203.02155v1)). For a task of this kind the same questions have to be answered explicitly: who made the comparison, under which specification, what that specification covers, and what it leaves out.

Ties and uncertainty should not be erased. A and C may both qualify, and the difference between them may be small, and an annotator may lack the background needed to judge a particular technical conclusion at all. None of these cases should be flattened at random into "the left one wins" or "the right one wins". The source suggests recording them separately: clear preference, slight preference, hard to distinguish, both unacceptable, and insufficient information.

When a later training objective accepts only strict binary labels, the records can be filtered or converted under an explicit policy, or the objective can be changed to support richer feedback. The conversion discards information, and that has to be recorded. For the same reason, "three of five people chose A" cannot be read as "A has a 60 percent probability of being factually correct". It describes the choice proportion of those people for that pair, and it is affected by sample size, by the specification, and by individual judgment.

## Disagreement, and a small annotation trial before training

Disagreement is sometimes a sign that the task definition is unclear. If one group of annotators consistently picks the shorter answer while another values stated limitations, the first thing to check is whether the prompt fixes the intended length and purpose. Disagreement can come from a repairable defect, such as vague instructions, missing material, display order, or model-identity cues, and it can also come from a real trade-off, where some users want a fast conclusion and others want a full argument.

Difference of opinion should not be called noise as a matter of course: sample the reasons, separate specification defects from insufficient capability from legitimate preference differences, and only then decide whether to revise the task or to keep different conditions. Recording the reasons per item is more explanatory than keeping a single win/lose flag. Those reasons are audit evidence, though; they do not automatically become the model's correct reasoning chain.

The procedure the source recommends before any training is a small annotation trial. Take a batch of prompts and, for each one, generate or write several candidates. Randomize the display order, hide unnecessary model identity, and give every annotator the same rules. First check whether they can consistently separate a clearly grounded conclusion from a clearly exaggerated one, then look at where reasonable answers disagree. Annotating a small sample repeatedly shows one person's stability; annotating the same sample with several people shows group disagreement. These are two different measurements and must not be merged.

The product of this step is not a training score. It is a revised specification, a comparison record with provenance, and a list of unresolved questions. If annotators do not share a rule about what evidence "stable improvement" requires, fixing the specification is worth more than running a preference objective immediately.

This article closes the definition stage of the volume. The notes that follow build these judgments into auditable data records, learn a reward model from the resulting comparisons, and then examine DPO and online optimization against the same evidence standard.

## Where the Chinese original lives

The Chinese original is [Standard Answers Are Not Unique: How to Provide a Training Signal](/posts/2026-10-20-llm-training-preference-01-feedback/). It holds the full text of responses A, B and C, the feedback-form table with its discussion, the discussion of criterion ordering and of non-compensable conditions, the treatment of ties and of the three-of-five count, the annotation-trial procedure, and the InstructGPT reference. Its main limit is unchanged: every record is a teaching case, no preference data was collected, and no model was trained. The next note in this volume turns these judgments into auditable data records.
