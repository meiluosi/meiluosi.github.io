---
sourceId: 2026-10-23-llm-training-eval-02-evidence
slug: llm-training-eval-02-evidence
title: "What Benchmarks, Human Ratings, and Model Judges Measure"
editionLabel: ENGLISH OVERVIEW · EVALUATION EVIDENCE
published: 2026-10-23T11:01:00+08:00
description: "Separate correctness from preference, calibrate judges, and disclose ordering, ties, contamination, and uncertainty."
tags:
  - Evaluation
  - Model Judges
category: LLM Training Research
series: LLM Training Research Notes · Evaluation and Reproducibility Volume
seriesOrder: 2
lang: en
---

> **Scope note.** Votes and labels are constructed. No human-rating study or judge-model invocation was performed.

## Match evidence to the question

A fluent reservation claim can be preferred yet false when a tool only queried stock. Rules check defined conditions; people assess usability and preferences under stated criteria; model judges can scale screening but require calibration. None automatically covers everything.

Benchmark results depend on versions, prompts, decoding, parsing, and aggregation. Exact-match overlap checks cannot rule out translated or paraphrased contamination. Preserve provenance and distinguish unknown training exposure from demonstrated cleanliness. Repeated test-driven prompt tuning also weakens independence.

## Calibrate human and model judgments

Give raters task evidence, hide model identity where practical, randomize order, preserve independent judgments and disagreements, and document expertise. [The MT-Bench judge study](https://arxiv.org/html/2306.05685v4#S3) examines position, verbosity, self-enhancement, and reasoning limitations. Its reported agreement is not a universal guarantee for new domains.

Fix judge versions and prompts, review representative failures, and treat candidate instructions as content rather than authority. A delimiter alone is not proof against judge manipulation.

## Declare order and tie handling

Map positional choices back to original model identities after swapping answers. A constructed ten-pair set has four stable A choices, three stable B choices, one stable tie, and two conflicts: 80% position consistency. Consistency is not correctness.

For 40 A wins, 30 B wins, and 30 ties, A's all-pair win fraction is 40%, decisive-only win rate is about 57.1%, and tie-half score is 55%. Report counts and definitions. Unjudgeable cases and parser failures should not silently become ties or disappear.

## Inspect class imbalance and uncertainty

An always-pass judge agrees with 90 of 100 labels when 90 are passing, yet detects none of the ten failures. Report error categories and relevant slices. An intentionally enriched calibration set also differs from deployment prevalence.

Preserve task-family, paired-answer, and run structure when assessing uncertainty. Narrow intervals do not fix systematic bias or contamination. Repeated judgments of the same answer do not measure training variability.

The [Chinese article](/posts/2026-10-23-llm-training-eval-02-evidence/) proposes complementary rule, human, and calibrated-model evidence while keeping unresolved disagreements visible.
