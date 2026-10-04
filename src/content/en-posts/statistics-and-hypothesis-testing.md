---
sourceId: 2024-08-15-统计学基础与假设检验
slug: statistics-and-hypothesis-testing
title: Statistics and Hypothesis Testing for Evaluating Changes
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2024-08-15T00:00:00.000Z
description: An overview of statistical summaries, inference, hypothesis tests, A/B testing, and common interpretation traps.
tags:
  - Statistics
  - Experiment Design
  - Evaluation
  - Data Science
category: Data Science
lang: en
---

> **Scope note.** This is an English overview of a Chinese methods tutorial. Its Python examples use small fixed lists or randomly generated data to illustrate calculations; they are not reports of completed experiments. The original article does not document a real intervention or measured product result. Statistical tests also rely on assumptions that must be checked for the data and study design at hand.

## Describe the data before testing a claim

Descriptive statistics summarize observed data. The tutorial covers measures of center—mean, median, and mode—and spread, including variance, standard deviation, interquartile range, and coefficient of variation. It also demonstrates skewness, kurtosis, histograms, box plots, Q–Q plots, and violin plots.

The choice of summary depends on the data. A mean can be pulled by outliers; a median can better represent a strongly skewed distribution; and a spread measure should be interpreted alongside the shape and scale of the data. Plots can expose patterns that one number hides, but they do not establish why a pattern occurred.

Inferential statistics use a sample to reason about a wider population. The article introduces sampling distributions and the central limit theorem, then uses confidence intervals to express uncertainty around an estimate. A confidence interval is not a guarantee that the fixed parameter lies in a particular interval with a specified probability; the interpretation depends on the sampling procedure and assumptions behind the interval.

## Frame a hypothesis test

A hypothesis test starts with a null hypothesis and an alternative, chooses a test statistic, and compares the observed data with what would be expected under the null model. The tutorial walks through one-sample, independent-sample, and paired-sample t-tests, as well as one-way ANOVA and chi-square tests for goodness of fit and independence.

The test must match the question and design. Independent samples and repeated measurements are different structures; the paired test uses within-pair differences. ANOVA can indicate that not all group means are equal, but additional analysis is needed to identify which groups differ. Chi-square procedures operate on counts and depend on conditions such as adequate expected cell counts.

A p-value describes how incompatible the observed data, or more extreme data, would be with a specified null model under the test assumptions. It is not the probability that the null hypothesis is true, the probability the result happened “by chance,” or a measure of effect size. The tutorial also introduces Type I and Type II errors, statistical power, and Cohen's *d* or eta-squared as effect-size measures.

## Design an A/B test before looking at outcomes

The A/B testing section proposes a workflow that begins with a defined metric and effect of interest, then estimates the sample size needed for a chosen significance level and target power. It sketches comparisons for continuous outcomes and conversion rates, including confidence intervals and absolute or relative changes.

For an interpretable experiment, define the eligible population, assignment unit, primary metric, stopping rule, and analysis before examining the results. Random assignment helps support a causal comparison, but implementation details such as sample-ratio mismatches, missing outcomes, interference between participants, or repeated exposure can still undermine it. Report the estimated effect and uncertainty, not only whether a threshold was crossed.

The source's examples with advertising strategies, training scores, conversion rates, and generated revenue values are instructional data. Their code output should not be read as evidence that a real intervention improved performance.

## Avoid common interpretation traps

The tutorial warns about several ways an apparently clear result can mislead:

- **Statistical versus practical significance:** a tiny difference can become statistically detectable in a very large sample. Decide what effect would matter in practice and report the estimated magnitude.
- **Low power:** a small study may miss a meaningful effect. A non-significant result does not prove that two conditions are equivalent.
- **Multiple comparisons:** testing many outcomes or variants increases the chance of false positives. Plan primary comparisons and use an appropriate correction or validation strategy for additional ones.
- **Repeated peeking and p-hacking:** repeatedly checking results and stopping only when a p-value is small changes the error properties of a fixed-sample test. Use a pre-specified stopping rule or a valid sequential method.
- **Assumption mismatch:** independence, distributional assumptions, measurement quality, and assignment mechanisms affect whether a test answers the intended question.

The article illustrates the first two points with simulated samples. Those demonstrations explain possible behavior of test statistics; they do not estimate a real-world treatment effect.

## Connection to RSI and AGI

An RSI-oriented system needs to distinguish genuine improvement from random variation, evaluation leakage, and trade-offs hidden by a single score. This tutorial's methods provide a starting vocabulary for specifying a baseline, metric, sample, uncertainty, and practical effect before comparing iterations. For adaptive systems, the evaluation design must also account for repeated changes and repeated testing; a fixed p-value rule cannot by itself certify that a system has improved broadly or safely.

For the full Chinese tutorial—including descriptive-statistics code, confidence intervals, t-tests, ANOVA, chi-square tests, A/B sample-size examples, multiple comparisons, and power analysis—see the original article.
