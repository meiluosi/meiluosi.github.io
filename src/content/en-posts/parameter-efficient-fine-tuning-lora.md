---
sourceId: 2024-09-22-高效微调技术lora与peft
slug: parameter-efficient-fine-tuning-lora
title: Parameter-Efficient Fine-Tuning with LoRA and PEFT
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-09-22T00:00:00.000Z
description: An overview of low-rank adaptation, other PEFT methods, QLoRA, and the results reported by the source article.
tags:
  - Fine-Tuning
  - LoRA
  - PEFT
  - QLoRA
category: Deep Learning
lang: en
---

> **Scope note.** This is an English overview of a Chinese tutorial. Its code and configuration snippets are instructional examples, not a run record. The article reports a GSM8K comparison, summarized below, but provides no experiment logs, software versions, checkpoint, or detailed evaluation protocol for independent reproduction.

## Why parameter-efficient fine-tuning?

Full fine-tuning updates all model weights for each task. That can require substantial accelerator memory and creates a separate full checkpoint for every adapted model. Parameter-efficient fine-tuning (PEFT) freezes most base weights and trains a smaller set of added or selected parameters.

PEFT can reduce trainable parameter counts and adapter storage, while keeping a base model reusable across tasks. These savings depend on the method and setup; they do not by themselves guarantee the same quality as full fine-tuning or eliminate inference costs.

## LoRA: represent the update with low-rank matrices

Low-Rank Adaptation (LoRA) keeps a pretrained weight matrix $W_0$ frozen and learns an update as the product of two smaller matrices:

$$
W = W_0 + \Delta W = W_0 + BA,
$$

where the rank $r$ is much smaller than the dimensions of $W_0$. Instead of learning every entry of the full update, the adapter learns the factors $A$ and $B$. The source illustrates this with rank choices from 4 to 64 and applies LoRA to attention projections such as query and value.

The tutorial's implementation initializes one factor randomly and the other to zero, then scales the update by $\alpha/r$. This makes the adapter contribution start at zero while its factors are learned. Rank, scaling, dropout, target modules, and learning rate are choices to validate on the task rather than universal defaults.

## Other PEFT approaches

The article surveys several alternatives:

- **Adapters** insert small bottleneck layers into Transformer blocks. They add trainable modules but can also add inference steps.
- **Prefix Tuning** learns virtual key/value vectors for attention layers.
- **Prompt Tuning** learns a small set of input embeddings, leaving the original token embeddings and model weights fixed.
- **P-Tuning v2** extends learned prompts across layers and uses a parameterization network in the example.

These methods place trainable parameters in different parts of a model. Their usefulness depends on task, model scale, data, software support, and deployment constraints; the article's comparison table is qualitative rather than a benchmark.

## A LoRA and QLoRA workflow

The source demonstrates wrapping a causal language model with a PEFT LoRA configuration, choosing a rank and target module, and inspecting the trainable-parameter count. A subsequent example tokenizes a sentiment dataset, trains with a Hugging Face `Trainer`, saves adapter weights, and shows how to load or merge them for inference.

QLoRA combines a quantized base model with trainable LoRA adapters. The tutorial sketches 4-bit loading with a bitsandbytes configuration and then applies a LoRA adapter. Quantization can lower memory requirements, but actual feasibility and quality depend on the hardware, model, sequence length, batch size, optimizer state, and implementation. The article's example does not attach a verified run log for its hardware claims.

## Results reported in the source

For a stated LLaMA-7B / GSM8K setup with 7,473 training examples, the article reports these numbers:

| Method | Accuracy | Training time | Memory |
| --- | ---: | ---: | ---: |
| Full fine-tuning | 36.4% | 10 h | 140 GB |
| LoRA, rank 8 | 35.8% | 4 h | 25 GB |
| LoRA, rank 64 | 36.2% | 5 h | 30 GB |
| QLoRA, 4-bit | 35.6% | 6 h | 12 GB |

These are **figures reported by the source article**, not independently reproduced results. The post does not specify enough detail to reproduce the comparison, including the exact model and dataset revisions, hardware, evaluation settings, or run artifacts. Treat them as a lead for follow-up verification, not as a general performance guarantee or a personal result claim.

## Connection to RSI and AGI

PEFT can make it cheaper to test a targeted model update and keep task-specific adapters separate. An RSI-oriented experiment would still need a defined objective, an unchanged baseline, held-out evaluation, and records of capability, regression, latency, and cost across iterations. Efficient updates are a tool for running that loop; they are not evidence that recursive self-improvement has occurred.

For the full Chinese walkthrough—including LoRA code, Adapter/Prefix/Prompt Tuning, PEFT examples, QLoRA, reported GSM8K figures, and implementation references—see the original article.
