---
sourceId: 2026-10-20-llm-training-pretrain-04-batching
slug: llm-training-pretrain-04-batching
title: "From Training Chunks to a Batch: Packing, Boundaries, and Effective Tokens"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of padding, packing, end-of-sequence markers, attention and loss masks, effective token counts, and gradient accumulation weighting."
tags:
  - Data Batching
  - Gradient Accumulation
  - Pretraining
category: LLM Training Research
series: "LLM Training Research Notes · Pretraining Volume"
seriesOrder: 4
lang: en
---

> **Scope note.** This is an English overview of a Chinese technical note in the pretraining volume. Its counting examples are hand calculations over two toy documents, and the packing rules are a design the source proposes rather than a configuration measured in a training run. The article reports no trained model, no throughput measurement, and no benchmark result. The document tokens used here are Latin placeholders standing in for the source's single-character Chinese tokens, with identical positions and counts.

## Two ways to build a batch: padding and packing

A model needs each step to receive inputs of a suitable shape, but source articles vary in length: one has a dozen tokens, another a few thousand. Getting them together is often treated as housekeeping before training. It is not. This step changes the context the model sees and changes which targets take part in the update. A packing program can run fast while quietly dropping the end of a document, or letting the opening of the next article depend on the ending of the previous one. Building a batch is both an efficiency problem and part of the training objective.

Take two short documents, already tokenized, each with start and end markers:

```text
document A: [BOS] x y [EOS]
document B: [BOS] z [EOS]
```

A occupies four input positions and B occupies three. Together they hold seven non-padding positions, but trained separately A offers three next-token targets and B offers two — five effective targets in total. The distinction deserves a pause: the number of input tokens and the number of effective prediction targets are not always equal. Start markers, the final position, padding, and ignored labels all change the count.

The convention here is a full-sequence input with one internal shift. The label table lists labels stored at the same position as the input, and the model aligns each previous output with the next label internally. If an implementation already shifts outside, the labels must be converted rather than shifted a second time.

Pad B to four positions and the two rows form a rectangular tensor:

| Item | A | B |
| --- | --- | --- |
| Input | BOS / x / y / EOS | BOS / z / EOS / PAD |
| Non-padding mask | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 0 |
| Labels | ignore / x / y / EOS | ignore / z / EOS / ignore |

The first label of each row is not reached by this example's internal shift; it is written as ignored to keep the boundary convention consistent across the table. The model does not need to learn to predict PAD at the EOS position.

The non-padding share of the input is $7/8=87.5\%$, while the effective target count is five. The eight cells are not eight effective training tokens.

Real tools expose a data collator that can pad dynamically to the longest sequence in a batch. The Transformers 4.57.1 documentation distinguishes dynamic padding from the label handling used for causal language modeling, where padding labels are ignored. Padding a single cell is cheap, but with one sequence of two thousand tokens beside one of twenty the waste becomes obvious, and approximate length grouping reduces it. Grouping also reorders samples, so if long documents correlate with a particular domain the training order introduces a bias that has to be checked.

Allow a block length of seven and the two documents can be concatenated:

```text
position: 0    1  2   3     4    5   6
input:    BOS  x  y  EOS   BOS  z  EOS
document: A    A  A   A     B    B   B
```

There is no padding now, but two new questions appear. May positions in B read A? And must the EOS position of A predict the BOS of B? With an ordinary lower-triangular causal mask, B can read all of A, and without extra label handling the pair from A's EOS to B's BOS becomes a prediction target. That can be a deliberately designed continuous-text objective, but it differs from the independent-document objective. Claiming that packing leaves the training content unchanged, while the context and the targets have both changed, does not hold.

An end marker is a token, not a wall. A model can learn the statistical role of EOS, such as what tends to follow the end of a passage, but ordinary attention computation does not clear earlier content when it encounters that id. To keep the two documents independent, the attention allowance should satisfy two conditions at once: the target position is not later than the query position, and both belong to the same document. With query position $i$, read position $j$, and document id $s_i$, the allowance is:

$$
\operatorname{allow}(i,j)=(j\le i)\land(s_i=s_j)
$$

Padding positions are excluded on top of that. The result is a lower triangle per document rather than one large triangle. An explicit mask or a segmented-sequence interface can implement it; the teaching formula does not require allocating a huge matrix in every scenario.

The label of B's BOS position is set to ignored at the same time, so A's EOS is never asked to predict the start of the next document. The effective targets remain three plus two, five in total. Blocking cross-document attention and blocking cross-document targets are different operations, and doing only one of them does not produce independent-document training.

## Position ids, chunk boundaries, and shared ids

If a model uses absolute position embeddings and B was trained on its own starting from zero, then a packed B should also use its own position ids, for example 0,1,2. Using the block-wide ids 4,5,6 leaves B's position embedding different from independent training even when cross-document attention is blocked. Some purely relative attention designs have a corresponding invariance under a common shift, but that does not cover every model, every position scaling, or every caching implementation. Comparing "processed separately" against "packed independently" requires fixing position ids, boundaries, randomness, and kernels, not just comparing token sequences.

This is also why independent padding is usually the easier thing to debug first: it is not necessarily the fastest, but it carries fewer boundary variables. Once the objective is right, an optimized packing can be tested for whether it preserved the intended behavior.

Short documents can be concatenated, but long ones must be cut. A document of ten tokens with a block length of four yields fragments of four, four, and two; dropping the last two means the model misses two tenths of each document on every pass. When documents follow a fixed format, the ending often carries the conclusion, the answer, or the exception handling, so the loss is not random reading. Keeping the last fragment and padding it avoids that deletion but then the cross-block prediction has to be checked: if each block is treated as an independent sequence, the last input of the previous block is no longer trained to predict the first target of the next one. An alternative design takes $T+1$ consecutive tokens each time, uses the first $T$ as input and the last $T$ as targets, and lets adjacent windows carry the boundary token over as the next window's first input. That covers adjacent prediction relations, but the later window still cannot see the earlier full prefix. Whichever design is chosen, record separately which tokens appeared as input, which were scored as targets, and how many times each target was scored: overlapping windows that score the same target twice increase its weight, and a claim of "higher coverage" does not hide that change.

Training and validation sets should not be merged into one text stream and split into blocks afterward. Related documents have to be split by source or group first and then processed independently within each set, otherwise the boundaries leak.

Some configurations set the padding token and EOS to the same id. Whether that is acceptable at the input level depends on the model and the mask convention, but label construction cannot then blanket-ignore every occurrence of that id, or the real EOS loses the supervision that teaches the model when to stop. Padding has to be identified through length, position, or an explicit non-padding marker, and the processed labels must be checked to confirm that a genuine EOS is still a target. The behavior of an actual collator is to be checked per version, not guessed from a class name. The two-dimensional non-padding mask and the full attention relation are not the same table either: a position that is not padding may still not read every other non-padding position, because the causal and document constraints still stack.

## Gradient accumulation and the effective-target accounting

Suppose one parameter update is computed from two micro-batches. The first has two effective targets with mean loss 1, the second has eight with mean loss 3. Dividing each mean by two before backpropagating produces an objective that weights the two micro-batches equally: $(1+3)/2=2$. If every token should carry equal weight, the objective is:

$$
L=\frac{2\times1+8\times3}{2+8}=2.6
$$

In general, given the effective count $n_k$ and mean loss $L_k$ of each micro-batch, the accumulation is:

$$
\sum_k\frac{n_k}{\sum_jn_j}L_k
$$

Gradients may be summed because the parameters do not change during accumulation, so the linearity of the derivative applies. Dividing by the accumulation count corresponds to token-equal weighting only when the micro-batches have equal size and equal effective counts. Distributed training adds cross-process reduction rules on top: if a framework already averages gradients and the code divides by the process count again, the gradient scale changes. The source does not offer a multiply-or-divide formula that claims to work for every trainer; derive the implementation from the global loss you want, and check it on a small example.

With four devices, two sequences of length 512 per device, and eight accumulation steps, the nominal number of input positions is:

$$
4\times2\times512\times8=32768
$$

If only three quarters of those positions are effective targets, one update carries about 24,576 effective targets, and a later supervised fine-tuning stage that scores answers only can be lower still.

| Counting convention | The question it answers |
| --- | --- |
| Nominal input positions | How many tensor positions were organized |
| Non-padding tokens | How much non-padding content was read |
| Effective targets | How many positions entered the loss directly |

Throughput follows the same split. A speedup measured in input positions may only reflect less padding; counting effective targets is closer to the rate at which training signal is processed. The learning value of a target is not decided by its count alone.

## A comparison that can expose boundary errors

Before comparing training quality, check that the packing matches the definition. Send A and B through the model separately and record the logits and losses of each effective target; repack them as independent documents with identical in-document position ids, targets, and weights; disable dropout and other randomness and compare corresponding positions within a reasonable numerical tolerance; then replace only A and check whether B's outputs stay unchanged. If the objective was meant to be a continuous text stream, that last invariance is not a requirement — the test has to serve the semantics defined in advance, and independent-packing acceptance criteria cannot be used to condemn a scheme that deliberately allows cross-document context. Once the boundary checks pass, measure throughput, peak memory, and validation behavior under a fixed corpus and objective. If throughput improves while validation worsens, go back and check whether the effective sample distribution changed as well.

## What stays in the Chinese original

The Chinese note carries the full packed-block tables, the long-document chunking designs, the per-step counting table, the diagnostic checklist, the two review questions, and the reference to the Transformers 4.57.1 data collator documentation.

Main limit, restated: the packing, masking, and position rules are a design, and the source reports no training run, no measured throughput, and no observed validation result.
