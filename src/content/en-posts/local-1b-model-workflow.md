---
sourceId: 2026-07-08-1b模型全链路实验计划
slug: local-1b-model-workflow
title: From Data to Deployment — A Small-Model Training Plan
editionLabel: ENGLISH OVERVIEW · PROJECT PLAN
description: A local, staged plan for data curation, tokenization, training from scratch, SFT, DPO, evaluation, and deployment, starting around 30M parameters.
published: 2026-07-08T00:00:00.000Z
tags:
  - Large Language Models
  - Pretraining
  - Post-training
  - Evaluation
  - Apple Silicon
category: Deep Learning
lang: en
---

The core question is how a language model comes into being: from raw text and random weights to instruction following, preference learning, and an inference artifact that can be evaluated and improved.

**Updated October 4, 2026.** The earlier plan started with an existing 1B-scale model and focused on adaptation. The revised project includes tokenization and pretraining from random initialization. It starts with a **tiny smoke baseline** on Apple M4 with **16 GB unified memory**. Around **30M parameters** is the first full experiment target; roughly 100M and 0.5B remain conditional on resource measurements. This article describes the implementation plan; training measurements will be added as the project progresses.

[Explore the interactive training guide →](/lab/model-pipeline/)

## Current engineering baseline

On October 4, 2026, a **138,560-parameter** tiny baseline completed real MPS updates on this Mac: 12 pretraining, 8 SFT, and 8 DPO steps, followed by a separate pretraining resume to step 14. Authored fixtures were split before fitting byte-level BPE on train only. Held-out fixture evaluation and fresh-process CLI loading were also checked.

This verifies the engineering pipeline, not language capability or sustained throughput. No 30M, 100M, or 0.5B training run is claimed. The portable `projects/small-model-lab/` directory contains code, raw summaries, and reproduction commands; no separate remote repository or weights have been published.

## Two experiments, with distinct starting points

The main path is:

```text
Data provenance → cleaning and document-level splits → tokenizer training
→ random initialization and pretraining → SFT → DPO
→ held-out evaluation → export and local inference → error analysis
```

A separate comparison uses an existing Qwen2.5-0.5B base model with LoRA or quantized LoRA on Apple Silicon. It studies adaptation costs; it does not replace the from-scratch path.

## Size follows measurement

For a transparent baseline, FP32 parameters, gradients, and two FP32 Adam states require about 16 bytes per parameter. That is approximately 0.45 GiB at 30M, 1.49 GiB at 100M, and 7.45 GiB at 500M.

These figures exclude activations, temporary tensors, runtime and operating-system memory, data buffers, and any frozen DPO reference model. They are not predictions of peak memory. Precision and optimizer choices also change the accounting.

Start with short sequences, micro batch size one, and gradient accumulation. Measure stable throughput after warm-up, peak memory, checkpoint cost, and longer-run behavior before choosing a training token budget. Estimated training time is planned tokens divided by measured training tokens per second, plus evaluation and checkpoint overhead.

The hardware inventory identifies an M4 MacBook Air with 10 CPU cores, 8 GPU cores, and 16 GB of unified memory. An early `df` snapshot on October 4, 2026 showed about 12 GiB free; a later MPS run recorded about 37.83 GiB through `shutil.disk_usage`. These snapshots differ in timing and reporting method, so neither is a lasting capacity guarantee. A 515M-parameter FP32 checkpoint with two Adam states is roughly 5.75 GiB; retaining two also requires room for data, temporary files, and exports. This tier has not been run and needs a fresh storage and memory budget before expansion.

## Data and tokenization

Keep a versioned manifest with sources, permissions, filtering decisions, document groups, hashes, and token counts. Normalize and deduplicate before splitting documents or source groups. Train an initial approximately 8K byte-level BPE tokenizer only on the training partition; freeze its vocabulary, special tokens, and chat template before encoding validation and test sets.

Hold the evaluation data out of pretraining, SFT, and preference training. Compare perplexity only with the same tokenizer, data, and scoring procedure.

## Training stages

**Pretraining:** implement a compact causal decoder and optimize next-token cross entropy. Begin with an engineering run, measure resource use, then allocate a fixed token budget. A resumable checkpoint includes optimizer, scheduler, random states, and data position as well as weights.

**SFT:** serialize examples with a fixed chat template. Score assistant answer tokens while masking prompts and padding. Keep truncation and answer-coverage statistics.

**DPO:** initialize both the policy and frozen reference from the SFT checkpoint. Use matched prompt/chosen/rejected examples and a consistent tokenizer and template. Track held-out behavior and length bias alongside the training objective.

## Evaluation and deployment

Compare random, pretrained, SFT, DPO, and exported checkpoints with fixed prompts and decoding settings. Record language-model loss, checkable instruction tasks, domain answers, preference behavior, regressions, memory, and speed.

The initial inference target is a fresh process loading the native configuration, tokenizer, template, and checkpoint. Conversion to MLX, GGUF, or another serving format depends on architecture support and needs its own numerical and output comparisons.

## Separate training project, connected writing

The independent `small-model-lab` project will hold data tools, configurations, training code, checkpoints, and evaluation reports. The website explains the stages and displays exported summaries. The interactive guide provides illustrative workflow and budget calculations; it does not run language-model training in the browser.

References include [MiniMind](https://github.com/jingyaogong/minimind), [nanochat](https://github.com/karpathy/nanochat), [LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch), [MLX LM](https://github.com/ml-explore/mlx-lm), [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html), and the [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B). Hardware-specific speed claims from these projects will be kept separate from local measurements.

The first milestone is a small, inspectable end-to-end run. Larger models follow measured resource limits and evaluation needs.
