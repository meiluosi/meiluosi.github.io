---
sourceId: 2026-10-20-llm-training-preference-02-pairs
slug: llm-training-preference-02-pairs
title: "How a Preference Pair Is Built: Shared Context, Candidate Gaps, and Annotation Bias"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-20T00:00:00.000Z
description: "Checks for preference data: shared context at token level, candidate difficulty and provenance, length and position bias, and truncation."
tags:
  - Preference Data
  - Preference Learning
  - Data Quality
  - Annotation Bias
category: LLM Training Research
series: LLM Training Research Notes · Preference Volume
seriesOrder: 2
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on constructing preference pairs. The "Cedar" prompt, the JSON record, and the pair-counting arithmetic are self-written teaching material, not the output of a collection campaign; the JSON shown here is an audit record, not a file handed to a trainer. No preference dataset was built or cleaned, no annotation was run, and no model was trained. The article does not establish that any dataset is fit for preference training.

## The shared context, and the comparison it has to support

A preference record usually has three main fields: `prompt`, `chosen`, `rejected`. The difficult part is not the field names but whether those three contents constitute a valid comparison.

If the two answers rest on different material, if the longer answer is always the one selected, or if truncation after annotation removed the decisive error, a trainer can still read the file while the supervision it carries has changed.

The article continues the [feedback criteria](/posts/2026-10-20-llm-training-preference-01-feedback/) and follows a pair from candidate generation to training input. The shared prompt is:

> Same validation set of 100 items: the original approach succeeded on 70, the new approach on 78. One run only, with no timing and no independent test results. Briefly explain what this change supports.

A pair can then be written as an audit record:

```json
{
  "prompt_id": "cedar-example-01",
  "prompt": "Same validation set of 100 items: the original approach succeeded on 70, the new approach on 78. One run only, with no timing and no independent test results. Briefly explain what this change supports.",
  "chosen": "Success rate on this validation set rose by 8 percentage points, but it is not yet possible to tell whether the gain is stable, nor to compare runtime.",
  "rejected": "The result proves the new approach improves stably by 8 times on all tasks and runs faster.",
  "criterion_version": "evidence-first-v1",
  "reason_codes": ["unsupported_generalization", "wrong_magnitude", "unsupported_speed"]
}
```

This is an audit record for the volume, not a claim that every trainer accepts these metadata fields. What the model reads and what the pipeline manages should be handled separately.

The shared prompt carries the task, the material, the system conventions, and whatever history is required. If one side saw additional experiments and the other did not, the result cannot be read as a preference between answers to the same conditions.

For data that stores a full conversation instead of a separate prompt and completion, the shared history has to match on both sides and the difference has to begin at the answer being compared. The longer context on the `chosen` side must not be taken as the common prompt with the `rejected` answer appended to it, and natural-language sameness must not hide differences in template, role markers, or special tokens.

TRL 0.23.1's `RewardTrainer`, for example, requires the implicit prompt format: the shared prompt goes into both the `chosen` and the `rejected` side, no separate prompt column is provided, and the conversational format is supported through a chat template. The audit record above therefore has to be converted first; it cannot be used as-is as training data for that interface. A file being accepted by a loader does not mean the comparison holds ([TRL 0.23.1 reward modeling](https://huggingface.co/docs/trl/v0.23.1/en/reward_trainer)). For sampled records, keep the final sequence of each side side by side and inspect the shared input, the answer boundary, and the end markers.

## Candidate difficulty and candidate provenance

The example above contains exaggeration, a miscalculation, and an invented speed claim, so the comparison is easy. It is useful for checking the annotation criteria, but it does not cover what training needs.

If every `rejected` answer were obvious gibberish, the model could learn to separate normal text from gibberish while still failing to compare two fluent answers where one slightly crosses the evidence boundary.

Candidates can cover several difficulty levels: clearly wrong, partly wrong, both factually fine but organized differently, and cases of genuine trade-off. "Both are poor" and "cannot be distinguished" should be kept as well and handled according to the criterion.

Hard cases are not automatically better. A comparison that is too ambiguous produces unstable labels. The order is: establish that the difference is real and explainable, then decide whether the case belongs in training or in a discussion of the criterion.

Provenance limits which errors the model ever sees. Several answers sampled from one model with the same settings may be highly similar; candidates from several models may be richer but may also introduce obvious stylistic differences. If `chosen` always comes from model A and `rejected` always from model B, a trainer can exploit fixed openings or formatting instead of the evidence-support relation the criterion is about.

Record the generating model, its version, the sampling settings, and the generation time, but hide unnecessary identity during annotation. Training and evaluation should include different combinations of sources, so that it can be checked whether the model has only learned to recognize one model's style. Regenerating candidates changes the data distribution: it changes the answer space the model is asked to compare, not merely the number of samples.

## Length, format, and order are features to test, not to erase

A longer answer is sometimes genuinely more complete; it may also just repeat the same conclusion with more politeness. A short answer may fit the request better, or it may drop the decisive limitation.

Start by counting the token length difference between `chosen` and `rejected`, then group the pairs to see whether the preference leans too heavily on length. Inspect specifically the cases where the longer answer adds no new information, and the cases where the shorter one keeps the decisive evidence.

Forcing every pair to the same length, however, can destroy content and delete the part that actually decided the preference. Length matching belongs in a dedicated diagnostic set and should not unconditionally replace the original task comparison. The same holds for formatting, number of citations, and tone: diagnosing a surface feature does not mean it is unimportant in every task, but it does mean checking whether the model treats it as an unreliable shortcut.

Presentation order affects annotation, and the field order must not leak the answer. Candidates can be placed on the left or the right at random, with the raw choice saved before it is mapped to `chosen` and `rejected`; the expected-better answer must not always sit on the left.

The same applies when a model judges. Check stability after swapping the order, and keep the judging prompt and the outputs. Published model-judging research discusses position and verbosity biases, which supports treating these as diagnostic items rather than treating a judging result as a correct label ([MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)). One pair changing its outcome after a swap does not by itself prove that the whole judge is unusable, but a large number of such changes means the labels are sensitive to presentation, and the procedure needs revision or added review.

## One ranking expands into correlated pairs

If four candidates for one prompt receive a strict ranking, they expand into six pairwise comparisons:

$$
\binom{4}{2}=\frac{4\times3}{2}=6
$$

Each candidate appears in three pairs. The six pairs share one prompt and one answer set; they are not six fully independent tasks.

InstructGPT's reward modeling section discusses the correlation between comparisons within the same ranking task, and organizes comparisons by prompt to handle repeated use ([InstructGPT, Section 3.5](https://arxiv.org/html/2203.02155v1#S3.SS5)).

In one's own data, the ranking or prompt grouping has to be kept, and the training weight of each prompt made explicit. If one prompt has ten candidates and another has two, mixing all the pairs with equal weight greatly increases the former's influence. Four candidates yielding six pairs is a change in the number of labels, not six times the semantic coverage.

Splits and deduplication follow the common source. All candidates of one prompt, all of its pairwise comparisons, and any rewrites of its material belong in the same split. Otherwise training has seen a candidate compared with A while testing compares it with B, and the independence of the evaluation weakens. It is also worth checking whether different prompts come from the same original material or the same template fact: the source grouping used in the earlier volume still applies, and it does not restart because the data has become preference pairs. An identical `chosen` and `rejected` string is not a meaningful strict preference either. If cleaning turned two originally different texts into identical content, the label needs to be reprocessed rather than kept as a win.

## Truncation and a small validation set

Suppose the first part of the `rejected` answer is accurate and only the last sentence exaggerates to "comprehensively better than the old one". If the training length limit removes exactly that sentence, the text remaining on both sides may be reasonable, and the original preference label no longer has a complete basis. Checking only that the token length complies will not detect this.

Keep the material the comparison needs, and the decisive content of the answer, as intact as possible. When that is impossible, exclude the record, rewrite it to a suitable length and re-annotate, or apply another explicit policy. Do not truncate silently and still call it the original comparison. It is also worth recording the sources and the types excluded for length, so that complex answers are not systematically dropped and then mistaken for coverage the training already has.

A small validation set can be prepared with records covering: ordinary preferences, identical answers, both-poor pairs, ties, mismatched shared history, swapped order, an error at the end, and one ranking expanded into several pairs. Check which of them enter training after conversion, which are kept but not used for the current binary objective, and which need human re-review. Each disposition needs a reason and a count.

Before training, sample pairs and answer three questions: what are the shared conditions of this pair of answers? What was the basis of the choice? Does the converted model input still preserve that basis? That is closer to the real quality requirement of preference data than verifying that the fields exist.

## Where the Chinese original lives

The Chinese original is [How Preference Pairs Are Built: Shared Context, Candidate Differences, and Annotation Bias](/posts/2026-10-20-llm-training-preference-02-pairs/). It holds the full JSON audit record, the shared-prefix and token-boundary discussion, the candidate difficulty and provenance rules, the pairwise-expansion arithmetic, the split and deduplication policy, the truncation policy, and the reference list to InstructGPT, TRL 0.23.1, and MT-Bench. Its main limit is unchanged: the pair is teaching material, no dataset was built, and no model was trained. The next note in this volume turns the pairwise labels into an explicit probability model.
