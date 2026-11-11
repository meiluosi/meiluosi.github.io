---
sourceId: 2026-11-11-moe-vs-dense-1-31b-token
slug: moe-vs-dense-1-31b-token
title: "After 1.31 Billion Tokens: Three Small MoE Reproductions vs a Dense Baseline, and Four Confounds I Cannot Ignore"
editionLabel: ENGLISH EDITION · LLM TRAINING
published: 2026-11-11T00:00:00.000+08:00
draft: false
description: "On a single 8 GB GPU and a 1.96B-token Chinese corpus, a dense 65M baseline and three small-scale MoE reproductions (Qwen3, DeepSeek-V3, GLM-4.5) each trained for 1.31B tokens. The MoE runs reach 0.10–0.14 lower validation NLL, but take 2.4–4.2x the dense AdamW wall clock, and optimizer, MTP head, parameter count and validation sample size all differ. This round is a direction, not an architecture ranking."
tags:
  - Pretraining
  - Mixture of Experts
  - Model Architecture
  - Experiment Log
  - Negative Results
category: LLM Training
lang: en
---

<!-- Mirrors the Chinese edition 2026-11-11-moe-vs-dense-1-31b-token. -->

> **Boundaries first: every number here comes from training logs on my own RTX 5060 Ti (8 GB), with a single seed and no repeats.** The test split has never been read. The project repository is private for now, so I only report numbers that can be checked line by line against the logs and configs; anything I could not verify is marked as an estimate. "Better" here means lower NLL on one validation split. It does not mean the model is usable, and it is not a public benchmark comparison.

## One-paragraph summary

Same corpus, same 1.31B tokens, same 10,000 steps:

- the three small MoE reproductions (scaled-down Qwen3, DeepSeek-V3 and GLM-4.5) end at validation NLL **3.2940 / 3.3379 / 3.2939**;
- the dense 65M baseline ends at **3.4347** with AdamW and **3.3799** with Muon;
- the MoE runs are genuinely lower, but cost **2.4–4.2x the dense AdamW wall clock**;
- and optimizer, MTP head, parameter count and validation sample size all differ, so **this round cannot rank architectures**.

## Why this comparison is worth running

A common claim in small-model practice is that a sparse model with more parameters but fewer active ones per token is a better deal when data is the bottleneck. My situation is exactly data-limited: 1.96B training tokens, and a single pass is far beyond what a model this size can digest.

If that claim holds on my own data and my own GPU, scaling total parameters with MoE is a good trade. If it does not, I am paying 4x the time for 0.1 nat. That has to be measured, not cited.

## Setup

- **Corpus**: one Chinese pretraining corpus (1.961B training tokens, 16K BPE vocabulary, 1024 context, document-level split, test split sealed).
- **Budget**: 10,000 steps x 131,072 tokens per step = **1.31B tokens**, about 0.67 epochs.
- **Models**: the dense `modern-66m` baseline (65.2M parameters, RoPE + RMSNorm + SwiGLU), and three faithful small-scale reproductions of published MoE architectures: 14 layers, width 512, 16 experts with top-2 routing, 217M–290M total parameters, with per-token active compute close to the dense baseline.
- **Training**: bf16, `torch.compile` plus flex attention, gradient accumulation, cosine schedule, 500 warmup steps; every model keeps checkpoints at steps 2500, 5000, 7500 and 10000.
- **Learning rate**: a three-point 500-step sweep per model first, then the peak was chosen (1.2e-3 for dense, qwen3 and deepseek; Muon at 4e-3 for glm).
- **Evaluation**: NLL on the same validation split, every 500 steps.

## Results

| Model | Total params | Optimizer | Validation NLL (step 10000) | Perplexity | Wall clock | Mean throughput |
|---|---:|---|---:|---:|---:|---:|
| modern-66m (dense baseline) | 65.2M | AdamW | 3.4347 | 31.0 | 6.05 h | 60.2k token/s |
| modern-66m-muon | 65.2M | Muon | 3.3799 | 29.4 | 7.99 h | 45.6k |
| qwen3-moe-small | 290.3M | 8-bit AdamW | **3.2940** | 26.9 | 14.5 h | 25.1k |
| deepseek-v3-small | 217.6M | 8-bit AdamW | 3.3379 | 28.2 | 16.0 h | 22.8k |
| glm4-moe-small | 230.4M | Muon | **3.2939** | 26.9 | 25.2 h | 14.4k |

All five validation curves decrease monotonically. No spikes, no out-of-memory fallbacks, and the throughput guard never fired. The test split was not touched.

## Reading 1: the MoE loss really is lower

The three MoE runs sit **0.097–0.141 nat** below dense AdamW and **0.042–0.086 nat** below dense Muon: perplexity moves from roughly 31 to 27.

qwen3 and glm differ by **0.0001**, which is indistinguishable at this validation sample size. deepseek trails the other two by about 0.044. In other words, there is no winner *among* the MoE runs, only a gap between MoE and dense.

## Reading 2: the price is 2.4–4.2x wall clock

Measured in wall clock, the dense model sees 2.4–4.2x more tokens in the same time. So the obvious question: should the comparison be per token or per unit of compute?

At this stage my answer is per token, for a concrete reason:

- the corpus has only 1.96B tokens and this round used 0.67 epochs, so **the binding constraint is unique data, not compute**;
- when data is limited, repeating the same tokens yields diminishing returns, so "learning more per token" is worth more than "seeing more tokens per second".

That reasoning has an edge: if I later allow 2–4 epochs of repetition, or move to a larger corpus, the ranking by compute could flip. **I cannot call the MoE loss advantage "more compute-efficient".**

## Reading 3: this is not a single-variable comparison

This is the main reason I am writing the post. The five models differ in more than architecture:

1. **Different optimizers**: glm uses Muon, qwen3 and deepseek use 8-bit AdamW. On the dense model, Muon alone is worth about 0.05 nat, the same order as the architecture gap.
2. **Different MTP**: deepseek and glm carry a multi-token-prediction head (extra supervision); qwen3 does not.
3. **Different total parameters**: 217M, 230M and 290M are not the same capacity.
4. **Different validation sample sizes**: 409,600 tokens for dense, 204,800 for qwen3 and deepseek, 102,400 for glm, because micro-batch size changes the evaluation batch.

So the only defensible reading is: **within this recipe, all three MoE runs are at least as good and clearly better than the dense baseline.** Which architecture is best is not answerable here.

## What this round actually establishes

Not "MoE is better", but three more basic facts:

- **The training pipeline survives five complete billion-token runs.** Checkpointing, resume, the memory guard and throughput monitoring were all exercised; none needed manual rescue.
- **An 8 GB GPU can train 200M–290M-parameter MoE models.** With an 8-bit AdamW or Muon, micro-batch 2–4 x accumulation 32–64, `grouped_mm` dispatch and compile, peak memory stays around 6.9 GB.
- **Evaluation can resolve differences at the 0.1-nat level.** Same validation split, same scoring path, no drift across reruns.

## What is not done yet

- **Downstream evaluation**: on the dense baseline C-Eval is at chance (23.0% against a 25% random baseline) and CMRC first-blank is 15.0% (against 7.9% random), the only task that separates models so far. The three MoE checkpoints have not been evaluated, so whether the NLL gain survives downstream is unknown.
- **0.5B MoE is still impractical**: at 539M parameters the model only fits at micro-batch 1, at 3.2k token/s, which means 4.7 days for 1.31B tokens. Roughly 0.9 GB more has to be saved through activation recomputation or a bf16 shadow copy before micro-batch 2 is possible.
- **Removing the confounds**: first an MTP 0/1 ablation, then reruns of the three MoE models under one optimizer; only after that a same-token comparison between a dense 125M model and a MoE candidate.

## A methodology reminder

Short learning-rate sweeps systematically favor larger rates: the schedule decays quickly and never exposes late-training instability. Every MoE learning rate here came from a 500-step sweep, and the dense baseline from a separate 500-step sweep, so **the tuning effort on the two sides is not equal**. Anyone using these numbers to argue that the MoE architecture is superior should read the four confounds above first.

Next round I will remove the confounds before claiming anything about architecture.
