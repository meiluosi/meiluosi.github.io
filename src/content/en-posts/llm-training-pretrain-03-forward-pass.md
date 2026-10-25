---
sourceId: 2026-10-20-llm-training-pretrain-03-forward
slug: llm-training-pretrain-03-forward-pass
title: "How One Forward Pass Handles Context: Vectors, Attention, and Position"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of embedding lookup, causal attention, positional encoding, residual and feed-forward blocks, and decoder-only tensor shapes."
tags:
  - Transformers
  - Attention
  - Pretraining
category: LLM Training Research
series: "LLM Training Research Notes · Pretraining Volume"
seriesOrder: 3
lang: en
---

> **Scope note.** This is an English overview of a Chinese technical note in the pretraining volume. Its worked attention example is a hand-computed calculation on a deliberately tiny configuration, and the three checks the source proposes were not executed there. The article reports no trained model, no benchmark, and no measured performance result. The teaching structure is a decoder-only, pre-norm design; it is not presented as the original encoder-decoder, post-norm paper architecture.

## Fixed dimensions and the first representation

The note follows a single short sequence instead of an architecture diagram full of module names, and asks three questions at each step: what is in hand, where does the information come from, and what is it finally used for.

Assume two sequences of four tokens each, a vocabulary of sixteen entries, and a hidden dimension of eight. The numbers are deliberately small so the shapes can be checked by hand; they are not a recommended training configuration.

| Quantity | Meaning | In this example |
| --- | --- | ---: |
| $B$ | how many sequences in a batch | 2 |
| $T$ | how many positions each sequence occupies | 4 |
| $V$ | how many candidates the vocabulary has | 16 |
| $d$ | how many numbers represent each position | 8 |

The input ids have shape $[B,T]=[2,4]$. The integers are indices, not values the model can directly use to measure similarity.

One trap has to be recorded early: the same shape does not mean the same meaning. A $[2,4]$ tensor of input ids and a $[2,4]$ tensor of labels may point at the current position and the next position respectively, and a shape check alone cannot catch the label shift described in the previous note.

The embedding matrix $E$ has $V$ rows and $d$ columns. Encountering id 7 means taking row 7, and after looking up every input the result has shape $[B,T,d]=[2,4,8]$.

The base vector of a token is the same wherever it appears. In "the apple is delicious" and "apple released a product", assume the fragment `apple` receives the same token id in both sentences. Its initial lookup then cannot distinguish fruit from company; capitalization or different token boundaries could instead produce different ids. That distinction has to form later, when the representation is combined with context. An embedding vector is not a dictionary in which the first number is "fruitness" and the second is "companyness"; training never assigned coordinates such meanings.

If a model uses additive positional embeddings, the position vector is added at this stage. Models using RoPE usually handle position inside attention, on the query and key. The two should not be stacked in a diagram as one mandatory flow.

## Attention: choosing and weighting the positions to read

For a representation $H$, one attention head produces three sets of trainable projections:

$$
\begin{aligned}Q&=HW_Q,\\K&=HW_K,\\U&=HW_U.\end{aligned}
$$

Here $U$ denotes what is usually called the value, so it does not collide with the vocabulary size $V$. The query is the vector used for matching at the current position, the key is the vector other positions are matched against, and the value is the content actually aggregated. All three are numerical transformations, not a natural-language question, keyword, and answer written in by hand.

The current query is dotted with each key to form matching scores. After scaling and masking, a softmax over the visible positions weights the values:

$$
\begin{aligned}S&=\frac{QK^\top}{\sqrt{d_k}}+M,\\A&=\operatorname{softmax}(S),\\O&=AU.\end{aligned}
$$

The softmax runs along the direction of the positions being read. $M$ is zero where reading is allowed and negative infinity where it is forbidden, so forbidden positions receive zero weight. The original Transformer's scaled dot-product attention and the decoder's causal mask are the source of this mechanism (Attention Is All You Need, section 3).

The scale factor is not decoration. If the components of the query and key are independent with zero mean and unit variance, the sum of $d_k$ products has variance $d_k$, and dividing by $\sqrt{d_k}$ restores unit variance. Real trained representations do not satisfy those assumptions exactly, so the derivation explains the design motive and does not guarantee that scores stay inside any fixed range in every layer.

Suppose one position may read three earlier positions, and the dot products and scaling produce the scores $\log2,0,0$. The weights are:

$$
\begin{aligned}(2,1,1)/4\\=(0.5,0.25,0.25).\end{aligned}
$$

With values $(2,0)$, $(0,4)$, and $(2,2)$, the output is:

$$
\begin{aligned}&0.5(2,0)+0.25(0,4)\\&\quad+0.25(2,2)\\&=(1.5,1.5).\end{aligned}
$$

Attention therefore mixes information from several positions instead of selecting one. The highest-scoring position gets the largest coefficient, but a contribution also depends on its value vector. In this example, the first two weighted vectors, $(1,0)$ and $(0,1)$, have equal magnitude despite different weights. The result is not a copy of the highest-scoring value. These weights should not be called "the model's explanation": an output projection, a residual, a feed-forward network, and other layers still follow, and the value magnitudes differ as well. A large weight at one position does not automatically mean a large causal influence on the final answer. A controlled substitution can probe whether a position affects the output, but the substitution may itself create an unnatural input. A weight visualization is an observation; a causal claim needs additional evidence.

Multi-head attention does not mean that capabilities are assigned by hand. With two heads of four dimensions each, the query, key, and value can be arranged as $[B,h,T,d_h]=[2,2,4,4]$, and the attention weights over all heads are also $[2,2,4,4]$, the last two dimensions being the query position and the position being read. Each head's output is concatenated back to eight dimensions and passed through the output projection, keeping the shape $[2,4,8]$ so the residual addition works. Heads have different parameters and may learn different matching patterns, but nobody specifies "the first head handles grammar and the second head handles facts". That anthropomorphic reading makes the head count look like a countable set of ability modules. Some models let several query heads share fewer key and value heads, so the three counts cannot be assumed equal; the shape example here covers ordinary multi-head attention only.

## Position, residuals, and the feed-forward layer

"A helps B" and "B helps A" contain the same fragments but mean something different, so content has to be combined with order. The causal mask provides the structure of "which prefix may be read", so it would be wrong to say the model has no order information at all; yet it does not directly give every pair of positions a learnable distance representation. Positional encoding adds order or relative distance.

A core move in RoPE is to rotate the query and the key by position. In two dimensions, with $R(\theta)$ a rotation matrix, position $m$ rotated by $m\theta$ and position $n$ rotated by $n\theta$:

$$
\begin{aligned}&(R(m\theta)q)^\top R(n\theta)k\\&=q^\top R((n-m)\theta)k.\end{aligned}
$$

The relative position $n-m$ has entered the dot product. Higher-dimensional forms apply different frequencies to different two-dimensional component pairs, and an implementation must fix the pairing and the frequency configuration (RoFormer, section 3). The identity explains how relative position enters the scoring; it does not prove that a model handles unbounded context reliably. Beyond the training length, the position distribution, the difficulty of the task, and implementation limits can all change, so raising the configured maximum length is not a validated long-context capability.

Attention aggregates across positions. An ordinary position-wise feed-forward network computes each position separately with shared parameters, projecting to a wider dimension, applying a nonlinearity, and projecting back. "Position-wise" does not mean blind to context: its input has already passed through attention and carries earlier positions. It only means this layer no longer reads other positions' vectors directly.

A common pre-norm teaching form is:

$$
H'=H+\operatorname{Attention}(\operatorname{Norm}(H))
$$

$$
H''=H'+\operatorname{FFN}(\operatorname{Norm}(H'))
$$

The residual keeps a direct additive path so a sublayer modifies an existing representation, and normalization adjusts the scale of that representation. The original Transformer paper places normalization differently, so these two lines are not a literal reproduction of it.

After stacking layers, each position's representation has passed through several rounds of context aggregation and nonlinear transformation. An output matrix then maps the $d$ dimensions to the vocabulary size $V$, producing logits with shape $[B,T,V]=[2,4,16]$, which the previous note's alignment and cross-entropy take over.

## Where the computation becomes expensive

For ordinary dense attention, pairwise scoring forms a $T\times T$ relation. Going from a length of 1,024 to 2,048 makes that per-head matrix four times as large in element count. It does not follow that total training time or peak memory grows fourfold: feed-forward layers, the vocabulary projection, memory traffic, and the actual kernels all contribute, and some implementations never materialize the matrix at all.

One of FlashAttention's research targets is reducing memory traffic in attention computation, using tiling to compute exact attention rather than approximating the original objective by dropping some attention connections. When comparing efficient kernels, two questions have to be separated: is the same mathematical target being computed, and is it faster or leaner on the hardware? Even for the same target, a change in floating-point operation order can produce small numerical differences.

The source proposes three small checks: record the shapes of ids, hidden states, per-head states, and logits, and confirm the direction of the softmax; in evaluation mode with dropout disabled, change only the last input token and confirm earlier logits stay within a reasonable numerical tolerance, otherwise investigate the causal mask, position indexing, or cross-sample computation; and check the mask interface, because a boolean mask whose `True` means "allowed to participate" is not the same convention as a float mask that is added to the scores. In PyTorch 2.9, `scaled_dot_product_attention` applies the dropout probability it is given, so evaluation must pass zero. Passing these checks makes the computation boundary more trustworthy; they do not show that the model learned syntax or reasoning.

## What stays in the Chinese original

The Chinese note carries the full tensor-shape walkthrough, the worked attention arithmetic, the RoPE identity with its implementation caveats, the pre-norm block formulas, the three proposed checks, the two review questions, and the reference list covering the original Transformer paper, RoFormer, and FlashAttention.

Main limit, restated: every figure in this overview is a hand calculation on a toy configuration, and the source reports no trained model, no measured attention statistics, and no completed diagnostic run.
