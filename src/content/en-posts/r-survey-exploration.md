---
sourceId: 2024-9-16-r语言学习
slug: r-survey-exploration
title: Exploring a Data Science Survey with R
editionLabel: ENGLISH OVERVIEW · ANALYSIS NOTE
published: 2024-09-16T00:00:00.000Z
description: An overview of the survey-cleaning and exploratory visualizations in the source article, with the source's stated sampling limitations.
tags:
  - R
  - Exploratory Data Analysis
  - Survey Data
  - Data Science
category: Data Science
lang: en
---

> **Scope note.** This overview reports what the Chinese article says about its analysis of a data-science survey. The notebook code and source outputs are not independently rerun here; the post itself notes that country comparisons may be affected by small or uneven samples.

## Prepare the survey data

The article uses R to explore a data-science practitioner survey. It reports a source dataset with 16,716 responses and 228 variables, then selects 16 variables for the examples. The analysis checks the data and treats ages of 0–3 and 100 as implausible responses for its age analysis.

The source demonstrates data loading, inspection, filtering, sorting, and reusable plotting functions. These steps illustrate an exploratory workflow rather than a fully documented cleaning specification for every field.

## Explore respondents and roles

One visualization compares median respondent age across selected countries. The source reports higher medians for New Zealand and lower medians for Indonesia in that comparison. It cautions that the country pattern should not be generalized: respondent counts and differences in who encounters or completes the survey can affect the result.

The article also compares the most common job categories among respondents from the United States and New Zealand. It reports Data Scientist, Software Developer / Engineer, and Other among the leading categories, while noting that the New Zealand sample is small and may not support a reliable ranking.

## Interpretation boundary

These are descriptions of the survey respondents, not representative estimates of national workforces. The source itself points to sample size, participation, and access to the survey as possible sources of bias. Stronger claims would require sample counts, weighting or a sampling model, and a clearly specified analysis protocol.

For the original R code, charts, and additional exploration of tools and learning interests, see the Chinese article.
