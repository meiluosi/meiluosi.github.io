---
sourceId: 2024-6-30-动态不确定性因果图模型理论在网络安全领域的应用
slug: ducg-cybersecurity
title: "DUCG for Cybersecurity: An Exploratory Concept"
editionLabel: ENGLISH OVERVIEW · CONCEPT NOTE
published: 2024-06-30T00:00:00.000Z
description: A conceptual sketch of linking alerts, attack stages, and possible root causes through a causal graph.
tags:
  - DUCG
  - Cybersecurity
  - Causal Modeling
  - Threat Detection
category: Causal Inference
lang: en
---

> **Scope note.** The source proposes a possible cybersecurity use for DUCG. It does not document an operational security system, a real attack dataset, or a measured improvement in detection quality.

## From alert lists to hypotheses

Security teams may receive alerts from different systems that describe symptoms at different points in a possible attack. The article's central idea is to connect those observations to candidate causes and stages, so an analyst can inspect a possible causal story rather than treating every alert as independent.

It sketches an attack-causality graph whose variables could include assets, vulnerabilities, alerts, and tactics. In principle, evidence from several sensors could update the relative support for competing hypotheses such as data theft, ransomware deployment, or cryptomining.

## Why the graph is not a detector by itself

The proposal depends on a trustworthy knowledge model: relationships need to reflect actual system architecture and threat behavior, alert reliability must be calibrated, and missing or misleading signals must be handled explicitly. The article identifies graph construction and knowledge engineering as challenges but does not report an implementation or evaluation against a baseline.

A practical study would need a defined threat model, labeled or expert-reviewed incidents, false-positive and missed-detection measures, and tests on cases not used to build the graph. Until then, the idea should be read as a conceptual application of causal reasoning.

The Chinese original contains the longer motivation, example variables, and discussion of modeling challenges.
