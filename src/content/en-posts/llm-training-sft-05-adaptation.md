---
sourceId: 2026-10-20-llm-training-sft-05-adaptation
slug: llm-training-sft-05-adaptation
title: "Full Fine-Tuning or LoRA: Update Space, Memory, and Quantization"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of LoRA update spaces, parameter and memory accounting, quantization, and comparable fine-tuning designs."
tags:
  - Supervised Fine-Tuning
  - LoRA
  - Quantization
category: LLM Training Research
series: LLM Training Research Notes · SFT Volume
seriesOrder: 5
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on choosing between full fine-tuning, LoRA, and quantized adaptation. Every resource figure in it is an arithmetic estimate under stated assumptions, computed by the source author — none is a measurement on local hardware. The matrix and storage calculations can be checked by hand, but the article ran no full-parameter, LoRA, or QLoRA training, and it names no winner. The Cedar extraction task is constructed teaching material. Nothing here reports a training outcome, a memory profile, or a task score.

## Full fine-tuning and the LoRA update

Take a linear layer:

$$
h=W_0x
$$

where $W_0$ has shape $d_{\text{out}}\times d_{\text{in}}$. Full fine-tuning can update this matrix directly, along with whatever else is left trainable. That gives a large adjustment space, but training also normally has to keep gradients and optimizer state for the trainable weights. Holding one copy of the model in memory is not sufficient. "Full fine-tuning" should also be read against the actual configuration: some projects still freeze embeddings or other parts, so a report is better off giving the list and count of trainable parameters than only a label.

LoRA freezes the base matrix $W_0$ and expresses the update as

$$
W=W_0+sBA
$$

where $A\in\mathbb{R}^{r\times d_{\text{in}}}$, $B\in\mathbb{R}^{d_{\text{out}}\times r}$, $r$ is the rank bound, and $s$ is a scaling coefficient. The original LoRA uses $s=\alpha/r$ ([LoRA, Section 4](https://arxiv.org/html/2106.09685v2#S4)). The forward computation becomes the original layer plus a narrower branch:

$$
h=W_0x+sB(Ax)
$$

The input is first mapped into $r$ dimensions and then back to the output dimension. Training adjusts the selected new matrices instead of making the whole $W_0$ trainable. What this describes is a low-rank update matrix; it is not a claim that the original weights were compressed to a low rank, nor that the model's capabilities are limited to $r$ dimensions.

### A rank-one update, calculated by hand

With three input dimensions, two output dimensions, $r=1$, and

$$
A=\begin{bmatrix}1&-1&0\end{bmatrix},\quad
B=\begin{bmatrix}2\\1\end{bmatrix},
$$

the product is

$$
BA=\begin{bmatrix}2&-2&0\\1&-1&0\end{bmatrix}.
$$

The second row is half the first, so the update can only vary along one independent direction. For the input $x=(3,1,4)^\top$, first $Ax=2$, then $B(Ax)=(4,2)^\top$. With $s=1$, the original output simply gains $(4,2)$. Multiplying the full matrix $BA$ by $x$ directly gives the same result. In this small example the original matrix has six parameters while the two factors together have five, so little is saved. LoRA's parameter advantage generally appears when the matrix dimensions are far larger than $r$; a rank of one by itself does not imply a large saving.

## How the parameter share is computed

The original matrix has $d_{\text{out}}d_{\text{in}}$ parameters, and the two factors together have

$$
N_{\text{LoRA}}=r(d_{\text{in}}+d_{\text{out}})
$$

Suppose a $4096\times4096$ matrix uses $r=8$: the original matrix has 16,777,216 parameters, the LoRA factors have 65,536, and the latter is about 0.390625% of the former. This is the added trainable-parameter ratio for a **single target matrix**. Adding adapters to more layers, or also training embeddings, the output head, or biases, changes the overall count. The base matrix still exists and cannot simply be subtracted from the model's total parameter count.

Choosing target layers is a separate decision from choosing the rank: a higher rank only widens the update space of matrices that were already selected, and it cannot substitute for directly updating a module that was not selected. The two should be recorded separately.

## Initialization, gradients, and the memory that remains

One LoRA initialization makes $A$ random and $B$ zero, so the initial increment is zero and the layer starts out consistent with the base layer ([LoRA initialization](https://arxiv.org/html/2106.09685v2#S4)). If both factors are zero, the ordinary gradient of the product is also zero at that step: the gradient of $A$ depends on $B$ and the gradient of $B$ depends on $A$, so the branch cannot start learning from that initialization. Under $A$ random and $B=0$, the gradient of $A$ may be zero at the first step, but $B$ usually receives signal; once $B$ is nonzero, $A$ starts to change. Seeing one factor unchanged at the start is not by itself evidence of a broken implementation.

Real libraries support other initializations and scaling variants. The fixed-version PEFT documentation distinguishes standard scaling from a rank-stabilized option that uses a different rank scaling, so two comparisons that write the same $r$ while changing the scaling and initialization are not comparable ([PEFT 0.17.0 LoRA configuration](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora)).

Fewer trained weights does not remove the base model from the backward path. With the base weights frozen, their parameter gradients and optimizer state are usually unnecessary. Training the adapter still has to know how the output changes with the adapter, and gradients can propagate through frozen layers toward earlier trainable parts. Activations, temporary buffers, the base weights, and the adapter state therefore still occupy memory, and longer inputs and larger batches can still exhaust it.

One assumption, offered only to explain the composition, is FP32 AdamW: each trainable parameter needs four bytes of weight, four bytes of gradient, and two state tensors of four bytes each, for sixteen bytes in total. Real implementations may add extra copies, use different precisions, shard, or optimize elsewhere, so sixteen bytes is not a universal measured constant. Under that assumption LoRA removes part of the trainable state; the base weights are still counted at their storage precision, and activations are counted separately. The overall memory saving is generally not equal to the 0.390625% above.

## QLoRA, merging, and making the comparison interpretable

**QLoRA** trains low-rank adapters on a quantized frozen base model. The paper discusses NF4, double quantization, and a paged optimizer, and it distinguishes low-bit storage from higher-precision computation: the weights have to be dequantized for use according to the computation path ([QLoRA, Section 3](https://arxiv.org/html/2305.14314v1#S3)). "4-bit fine-tuning" therefore does not mean that all computation, gradients, and states are four bits wide, and it does not mean an ordinary full-parameter update is applied directly to four-bit base weights.

Suppose one billion weights, each ideally occupying four bits: the raw encoding alone needs five hundred million bytes, about 0.466 GiB. At sixteen bits each, it would be two billion bytes, about 1.863 GiB. Neither figure includes quantization metadata, unquantized modules, the adapter, activations, or runtime buffers. Practical usability also depends on device and kernel support, and the paper's hardware and results cannot guarantee that a particular laptop runs the same setup. Check the implementation and measure before declaring that something "definitely fits".

Merging an adapter still requires a regression check. For an ordinary linear LoRA, and under the same numerical semantics, $sBA$ can be added into the base weights to produce a merged matrix for inference, so the adapter branch no longer has to be executed separately. But floating-point rounding, quantization, requantization, and implementation paths can all change the numbers, and merging into quantized base weights before exporting in another format should not be treated as a full deployment-consistency proof on the strength of an algebraic identity. At minimum, compare logits or target probabilities before and after merging on fixed inputs, then compare key task outputs. If generation uses sampling, use a deterministic setting suited to numerical checking first, otherwise computation differences and sampling differences are hard to separate. An adapter also depends on the correct base model, layer names, vocabulary, and template; keeping a small file while discarding those version details does not make a complete deliverable.

For the structured extraction task in this volume, start from one baseline that the available resources allow, rather than assuming LoRA must beat full fine-tuning or that full fine-tuning must be more accurate:

| Option | What it mainly opens up | What has to be measured |
| --- | --- | --- |
| Full fine-tuning | Direct updates to the existing trainable weights | Parameter state, training time, capability change |
| LoRA | A low-rank increment on selected modules | Base weights, activations, adapter state, adaptation capacity |
| QLoRA | Adapters on a quantized base model | Quantization error, kernel support, dequantization, runtime overhead |

A fair comparison does not require blindly using the same learning rate, since different parameterizations may need different settings. It does require comparable tuning budgets, fixed data, fixed task evaluation, a fixed budget measure, and a record of the actual resource consumption. If ranks are compared, hold the target modules, the scaling definition, the data, and the evaluation constant, and state how the remaining parameters were chosen; changing the target modules and the rank at once compares two whole configurations, not the isolated claim that a larger rank is better.

Checking in the first round that the adapter is working comes before any conclusion. Print the trainable parameters and their shapes before training, to confirm that the base model was not accidentally unfrozen and that at least one adapter was actually attached. After training, check that the base parameters still satisfy the frozen convention, that the adapters changed, and that loading the same adapter restores the same output; then measure field correctness, absence handling, and format adherence on independent material. If the model only gets better at the fixed template, look at data coverage and splits first rather than blaming an insufficient rank, because a larger update space cannot repair contradictory demonstrations.

## The Chinese original and the limits of this note

The full Chinese article carries the hand-computed rank-one matrices, the parameter and storage derivations, the initialization and gradient argument, the option comparison table, and the reference list with the LoRA, QLoRA, and PEFT 0.17.0 links — material summarized but not reproduced here. The original is at [the Chinese source article](/posts/2026-10-20-llm-training-sft-05-adaptation/); the next note covers [deciding what SFT improved](/en/posts/llm-training-sft-06-evaluation/). The main limit is unchanged: the memory and storage figures are estimates under stated assumptions, and no full-parameter, LoRA, or QLoRA training was run to compare them.
