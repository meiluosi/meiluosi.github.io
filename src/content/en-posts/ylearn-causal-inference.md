---
sourceId: 2024-6-30-应用ylearn框架实现因果推断
slug: ylearn-causal-inference
title: Causal Inference with YLearn
editionLabel: ENGLISH OVERVIEW · RESEARCH NOTE
description: An overview of causal effects, uplift modeling, graph discovery, and sensitivity analysis, illustrated with synthetic data.
published: 2024-06-30T00:00:00.000Z
tags:
  - Causal Inference
  - Data Science
category: Data Science
lang: en
---

> **Scope note.** This is an English overview of a longer Chinese tutorial. Its coupon, A/B-test, and healthcare examples use simulated data. The source note does not document production deployment or measured business impact. Follow the Chinese original for the full code listing.

## Why causal inference?

Machine-learning models are good at finding patterns, but an association alone does not answer what would happen if we changed something. A coupon campaign, for example, may coincide with higher spending because customers who were already likely to buy were also more likely to receive a coupon.

Causal inference frames the question in terms of interventions and counterfactuals. For an individual `i`, let `Yᵢ(1)` be the outcome under treatment and `Yᵢ(0)` the outcome without it. The individual treatment effect is `τᵢ = Yᵢ(1) − Yᵢ(0)`, but only one of those outcomes can be observed for the same individual.

Two useful population quantities are:

- **Average treatment effect (ATE):** `E[Y(1) − Y(0)]`.
- **Conditional average treatment effect (CATE):** `E[Y(1) − Y(0) | X = x]`, which describes how effects may vary across groups.

A **confounder** affects both treatment assignment and the outcome. If user activity makes a person more likely to receive a coupon and more likely to purchase, an unadjusted comparison can attribute the activity effect to the coupon. Randomized controlled trials, causal graphs, and potential-outcome models offer complementary ways to reason about such problems.

## Where YLearn fits

The source note presents YLearn as a Python toolkit for a workflow that moves from data preparation through causal-graph discovery and effect estimation to policy decisions. It discusses:

- **Effect estimation:** S-, T-, X-, and doubly robust learners for ATE or CATE analysis.
- **Uplift modeling:** ranking people by their estimated response to an intervention.
- **Graph discovery:** PC and GES algorithms, followed by identifying adjustment sets.
- **Robustness checks:** propensity scores, inverse probability weighting (IPW), and sensitivity analysis.

These methods depend on assumptions about the data and treatment assignment. A library can implement an estimator; it cannot make those assumptions true or replace a sound study design.

## The coupon example

The tutorial generates 5,000 synthetic customer records with age, income, and activity. Activity affects both the probability of receiving a coupon and spending, creating a confounding path. The outcome also includes a treatment effect that varies with age and income.

The walkthrough compares a naive treated-versus-control difference with estimates from S-, T-, X-, and DR-Learners. It then estimates CATE to inspect effect heterogeneity. The source includes code to calculate these quantities, but it does not preserve a reported run log or numerical result; this overview therefore makes no claim about which estimator performed best.

## From effects to intervention policies

Uplift modeling focuses on the incremental response to treatment. The tutorial describes four familiar response groups:

| Group | Without treatment | With treatment | Possible action |
| --- | --- | --- | --- |
| Sure Things | Convert | Convert | No coupon needed |
| Persuadables | Do not convert | Convert | Potential target group |
| Lost Causes | Do not convert | Do not convert | Avoid ineffective spend |
| Sleeping Dogs | Convert | Do not convert | Avoid a harmful intervention |

The example code ranks simulated users by predicted uplift and compares targeting the top 30% with selecting 30% at random. That comparison is an analysis template, not a verified ROI result.

## Discovering and using causal graphs

The note also demonstrates PC and GES graph-discovery examples, then uses a graph to identify an adjustment set for estimating a treatment-to-outcome effect. A discovered graph is not automatically a causal truth: results depend on the assumptions, variables, and data supplied to the algorithm.

For observational estimates, the tutorial introduces propensity scores and IPW, plus bootstrap sensitivity analysis. These are ways to inspect balance and uncertainty; they do not eliminate bias from unobserved confounders by themselves.

## Additional scenarios and limits

The source applies the same ideas to synthetic A/B-test and healthcare scenarios. It suggests using CATE to examine who may benefit more from an intervention. These examples are illustrative: the article does not report live experiments, clinical recommendations, or measured outcomes.

Common risks include selection bias, omitted confounders, and model misspecification. Possible checks include propensity-score matching, IPW, doubly robust estimators, sensitivity analysis, and comparing models. Each addresses different assumptions; none guarantees an unbiased answer in every setting.

## A practical reading of the workflow

The tutorial’s workflow can be summarized as:

1. Define the intervention and outcome precisely.
2. Identify plausible confounders and state the assumptions.
3. Choose an estimator appropriate to the data and question.
4. Estimate ATE or CATE and inspect overlap and uncertainty.
5. Run sensitivity checks before using estimates to guide a policy.

The original note includes code for these steps, including the synthetic-data generator and YLearn examples. Because the API examples were written for an earlier library context, check the [YLearn repository](https://github.com/DataCanvasIO/YLearn) before reusing them with a current installation.

## References in the source note

- Scott Cunningham, *Causal Inference: The Mixtape*.
- Judea Pearl, *The Book of Why*.
- [DataCanvasIO/YLearn](https://github.com/DataCanvasIO/YLearn).
