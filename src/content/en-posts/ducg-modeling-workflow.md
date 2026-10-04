---
sourceId: 2024-07-22-ducg建模实战指南
slug: ducg-modeling-workflow
title: A DUCG Modeling Workflow for Diagnosis
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2024-07-22T00:00:00.000Z
description: An overview of a DUCG modeling workflow, illustrated with a software-defect diagnosis example and implementation sketches.
tags:
  - DUCG
  - Causal Modeling
  - Diagnosis
  - Evaluation
category: Probability
lang: en
---

> **Scope note.** This is an English overview of a Chinese modeling guide. The software-defect example, Python classes, validation routines, and deployment snippets are instructional sketches. The article does not provide a dataset, execution logs, measured accuracy, or a reproducible DUCG implementation. Its target such as “accuracy above 85%” is an example requirement, not an achieved result. A graph records causal assumptions; drawing an edge does not establish that relationship as causal.

## A seven-step modeling workflow

The guide organizes a Dynamic Uncertain Causality Graph (DUCG) workflow into seven steps: define the problem, identify variables, find candidate causal relations, construct the network, estimate parameters, validate and refine the model, then deploy an inference interface.

This structure is useful for separating decisions that are often mixed together. The modeling question determines what the target is; the available observations constrain which variables can be represented; and the domain assumptions and data determine how edges and parameters should be justified. A diagram alone does not settle any of these choices.

## Define variables and the diagnosis target

The article uses software-defect diagnosis as a running example. Candidate root causes include logic errors, memory leaks, and concurrency problems. Intermediate variables represent symptoms such as crashes or degraded performance; observable variables include test failures, code coverage, and exception counts; a target variable represents defect severity.

Its Python sketch gives variables binary, discrete, or continuous states and stores parent-child relationships, causal strengths, and conditional probabilities. Continuous variables are marked as requiring discretization. These are custom teaching classes, not a supplied library or a validated DUCG package. A real model must define state semantics, measurement procedures, missing-data behavior, and how observations relate to the target.

## Propose and encode relationships

The guide discusses combining expert knowledge, literature review, and data analysis when proposing relations. It also sketches a directed graph implementation with NetworkX, along with functions to add variables, edges, strengths, and conditional probabilities.

These sources of information have different roles. Expert judgments and published mechanisms can motivate candidate edges; observational data can reveal patterns that merit further investigation. Correlation analysis or a data-driven graph-discovery algorithm alone cannot determine causal direction without assumptions and appropriate identification conditions. The assumptions behind each edge should be documented and challenged, especially when hidden confounding, selection effects, or temporal ordering could change the interpretation.

The article describes learning causal strengths and estimating conditional probability tables, with Bayesian smoothing as an option when data are sparse. Its code is schematic: the post does not supply a complete executable package, a dataset, or a record showing that the presented APIs and formulas were run together.

## Validate the model and its reasoning

The validation section sketches k-fold cross-validation, sensitivity analysis, and comparison by metrics such as accuracy, F1, and AUC. It also proposes testing how an inference result changes when evidence is perturbed.

These are possible components of an evaluation plan, not reported findings. Cross-validation must match the data-generating structure; random folds can leak information when records are grouped or time ordered. The sketch should also be checked for state reset between folds and for metrics that suit the outcome. Its sensitivity example treats outputs as numeric, while a diagnosis may produce a category or probability distribution; a real implementation needs an explicit distance or change measure. The article reports no completed fold scores, sensitivity results, or comparison table.

For a diagnosis system, validation should include calibration and uncertainty, false-positive and false-negative costs, out-of-distribution cases, and whether the proposed intervention actually changes outcomes. A high predictive score would not, by itself, establish that the encoded causal structure is correct.

## Deployment does not close the evidence gap

The guide sketches an inference method and a web API for submitting evidence and returning a diagnosis with an explanation. A deployed interface can make a model usable, but it cannot compensate for unsupported causal assumptions or weak validation. In practice, model versioning, data permissions, latency, monitoring, and a process for reviewing incorrect recommendations are also part of the system.

## Connection to RSI and AGI

Causal models can help formulate questions about what changed and which interventions might have produced the change. For an RSI-oriented evaluation loop, a causal graph can make assumptions explicit, while controlled interventions and held-out measurements test those assumptions. Neither a modeled graph nor an inference API demonstrates that an agent can reliably improve itself; the improvement claim needs independent, repeated evidence and careful monitoring for regressions.

For the full Chinese guide—including the seven-step workflow, software-defect variables, graph-building code, parameter-estimation sketches, validation examples, and API outline—see the original article.
