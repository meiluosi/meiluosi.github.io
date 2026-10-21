---
sourceId: 2026-10-20-llm-training-pretrain-02-objective
slug: llm-training-pretrain-02-objective
title: "What a Language Model Predicts: Label Shift, Cross-Entropy, and Loss"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of next-token prediction, label alignment, cross-entropy, effective-token averaging, and the limits of perplexity comparisons."
tags:
  - Language Modeling
  - Cross-Entropy
  - Pretraining
category: LLM Training Research
series: "LLM Training Research Notes · Pretraining Volume"
seriesOrder: 2
lang: en
---

> **Scope note.** This is an English overview of a Chinese technical note in the pretraining volume. Its numeric examples are hand-computed teaching calculations over a toy four-token sequence and three candidate logits; no model was trained, no corpus was run, and no training log in the source belongs to a real run. The alignment experiment the source proposes is a design, not a completed experiment. The source's example sentence is Chinese, so this overview relabels the same four-token structure with English words, each still assumed to be a single token.

## One sentence, several prediction problems

The source opens with a training log whose loss falls from 8 to 5 and then to 3. Read carelessly, a falling curve suggests the model is learning language. But if the model can see the answer it is supposed to predict, or if the code shifts the target by one position, the loss can still fall. A falling loss only says the model is getting better at the task the code actually defines. Whether that task is the right one has to be checked first.

The basic setup is plain autoregressive language modeling: each position predicts the next token from what came before. Label smoothing, class weights, and auxiliary losses are deliberately left out.
The teaching sequence is four tokens, with the two middle fragments each assumed to occupy one token:

```text
[BOS]  today  rains  [EOS]
```

`[BOS]` marks the beginning and `[EOS]` the end; the two middle fragments are each assumed to occupy one token. This is drawn so the positional relationship stays visible, and it does not mean that all models use these markers or this segmentation.

It corresponds to three questions:

| Visible so far | Correct next token |
| --- | --- |
| `[BOS]` | today |
| `[BOS] today` | rains |
| `[BOS] today rains` | `[EOS]` |

The second question admits several reasonable continuations — "today sunny" works as well — but this training record actually contains "rains", so "rains" is the target here. Other texts provide other opportunities to continue the same context.

A language model is not labelling whether a sentence is factually correct. It learns the conditional distribution in the training data: which tokens tend to follow a given context. Errors, bias, and repetition inside the text can therefore become easy-to-learn regularities.

## Causality, visibility, and the label shift

The entire sequence can sit in one tensor while outputs at many positions are computed together. What matters is which information each position may read. Under the usual causal attention constraint, position $t$ can read itself and earlier positions but not later ones. The original Transformer paper describes exactly this masking of future positions in the decoder (Attention Is All You Need, section 3.1).
So although the position holding "today" shares a tensor with "rains", its computation cannot use the latter:

| Computed at | BOS | today | rains | EOS |
| --- | --- | --- | --- | --- |
| BOS | visible | not visible | not visible | not visible |
| today | visible | visible | not visible | not visible |
| rains | visible | visible | visible | not visible |
| EOS | visible | visible | visible | visible |

The first three questions can therefore be scored in one pass, without generating token by token during training. In this standalone example the final EOS position has no successor, so it can carry no loss.
One point is easy to confuse here. Training supplies the true prefix; it does not feed the model's own wrong guess back in. When the target "rains" is scored, the context is the "today" that the data contains. At generation time the next step usually continues from what the model itself produced, so an earlier mistake enters the later context. Prediction loss under a true prefix is related to, but not the same measurement as, free-running generation.

With the sequence numbered as:

```text
[BOS, today, rains, EOS] = [0, 1, 2, 3]
```

The most direct hand-written arrangement is:

```text
input x:  [0, 1, 2]
labels y: [1, 2, 3]
```

Each input position's output is compared with the corresponding label. Labels are class indices, not numbers to be subtracted. Renumbering "rains" from 2 to 200 does not move it closer to "today" as long as every mapping stays consistent.

Some wrappers perform the shift inside the model. The Transformers 4.57.1 GPT-2 documentation allows `labels` to be passed identical to `input_ids`, because the package aligns each prediction position with the next label internally. That is an interface convention of this pinned version, and it does not imply that every model and trainer behaves the same way.

Both conventions are legitimate. The dangerous case is mixing them:

| Case | What the output at position "today" is actually asked to predict |
| --- | --- |
| Shifted once | rains |
| Never shifted, compared against the same-position label | today |
| Shifted outside and shifted again inside | possibly `[EOS]`, depending on the slicing implementation |

With no shift, the position already sees its own input, and the task can degenerate into copying. A double shift can skip a token. Neither case necessarily raises an error, and both can still produce a falling curve.
The most effective check is not staring at framework names but printing the pairs that actually scored: which is the last token this output can see, and what is the label? Checking three consecutive positions is usually more useful than running a few thousand steps first.

## From logits to a scored target

At each position the model emits a set of scores, usually called logits, with one score per vocabulary candidate. Scores may be negative and are not required to sum to one. Softmax turns them into probabilities. For the current context, candidate $j$ has probability:

$$
p_j=\frac{\exp(z_j)}{\sum_{k=1}^{V}\exp(z_k)}
$$

Here $z_j$ is a candidate score and $V$ is the vocabulary size. The scores have no absolute scale: adding the same constant to all of them leaves the probabilities unchanged.
Take three candidates, "rains", "sunny", and "end", with scores $\log4$, $\log2$, and $\log2$. Exponentiating gives 4, 2, and 2; dividing by the total of 8 gives 0.5, 0.25, and 0.25. This is a locally hand-computable example. A real model has far more candidates, but the scoring logic is the same.

The target here is "rains", so the loss is the negative log of the probability assigned to it:

$$
\ell=-\log p_{\text{rains}}
$$

The natural logarithm is used. The smaller the probability given to the correct target, the larger the loss:

| Probability given to the correct target | Loss, approximately |
| ---: | ---: |
| 0.8 | 0.2231 |
| 0.5 | 0.6931 |
| 0.1 | 2.3026 |
| 0.01 | 4.6052 |

The loss therefore does more than check whether the top-ranked candidate is right. Two models can both rank "rains" first, one at 0.8 and one at 0.5, and still have different losses. This is what penalizes being confidently wrong.
The objective also does not require every sample to reach probability one. When several samples share a context but differ afterwards, the model must distribute probability among those observations; capacity, data, and the optimization process decide how closely it can approach the distribution.

For a single-class target without label smoothing, the derivative of the cross-entropy with respect to the scores can be written:

$$
\frac{\partial\ell}{\partial z_j}=p_j-\mathbf{1}[j=y]
$$

where $y$ is the correct target index. For the three candidates above the values are $-0.5$, 0.25, and 0.25. Treating the scores as directly adjustable variables and moving against the gradient raises the correct candidate's score and lowers the others. A real network updates shared parameters instead of keeping a separate set of scores per sample, so one update also affects other contexts — which is why generalization and mutual interference both become possible.

At the implementation level, PyTorch's `CrossEntropyLoss` takes unnormalized logits and performs the corresponding computation internally. Converting them to probabilities first and passing those as logits would define a different objective. Its class indices, ignored positions, and averaging mode must be checked against the specific interface (PyTorch 2.14).

## Averaging over effective targets, masks, and perplexity

A sentence has several targets and a batch has several sentences, so the source states which positions take part and how they are averaged. If position $i$ has target loss $\ell_i$, and $m_i=1$ when it is scored and zero otherwise, then with no extra weights:

$$
L=\frac{\sum_i m_i\ell_i}{\sum_i m_i}
$$

The denominator is the effective target count, not the number of cells in the input tensor. Padding positions should not dilute the average merely by occupying cells. If an entire batch has no effective target, the denominator is zero; that requires a data check or an explicit skip, and must not be reported as a perfect zero loss.

Two averaging conventions have to be kept apart:

- **Per token:** all scored positions pooled together and divided by their total.
- **Per sample:** each sample averaged on its own, then samples weighted equally.

Suppose A has 2 effective positions with mean loss 1, and B has 8 with mean loss 3. The per-token average is $(2\times1+8\times3)/10=2.6$; the per-sample average is $(1+3)/2=2$. Neither is arithmetically wrong; they define different weights, and the longer sample contributes more positions under the first. A training log that prints one number without saying how it was averaged is hard to compare across experiments.

The same reasoning applies to validation batches of different lengths: accumulate the total loss over effective targets and divide by the total effective target count, rather than averaging the averages. Gradient accumulation runs into the same issue.

One distinction will recur in later supervised fine-tuning work: the attention mask decides where information may flow from, while the loss mask decides which prediction positions are scored. Marking a label as ignored means only that this target is not scored; it does not mean the model cannot see the corresponding input. In question-and-answer tuning the prompt can stay in the context while only the answer is scored. Conversely, if a future answer can still be read through attention, ignoring some labels cannot by itself remove that leak. Checking the causal constraint and checking the scoring range are two separate jobs.

Common interfaces use `-100` to mark ignored labels. It is not an ordinary vocabulary token and should not be fed into the embedding layer as an input id; whether the value is used still depends on the loss function's own convention.

Under one fixed token definition and one fixed scoring range, the exponential of the mean negative log loss is the usual perplexity:

$$
\mathrm{PPL}=\exp(L)
$$

If the model assigns the correct target a probability of 0.5 at every position, then $L=\log2$ and the perplexity is 2. At 0.25 everywhere, it is 4. It is best read as the reciprocal of the geometric mean of the target probability, not as "two candidates left"; distributions may vary widely across positions, and equal perplexity does not imply equal candidate-set size.

Perplexity's unit follows the token. Different tokenizers, corpora, truncation choices, and context policies change the prediction problems themselves, so a per-token perplexity from one model cannot be lined up against a model with a different segmentation as a capability ranking. A falling validation loss also needs a breakdown: if the validation set is mostly template sentences, predicting the template better pulls the overall loss down while rare but important technical question answering may not improve at all.

The source closes with a small alignment experiment to run before any formal training: take two or three very short sequences with a fixed tokenization, save the visible prefix, target id, target probability, scoring flag, and per-position loss, then check that manual pairing agrees, that hand-computed values match an independent negative-log computation, and that the information boundary holds by changing only future tokens in evaluation mode and confirming earlier outputs stay within a reasonable numerical tolerance with randomness disabled. That last check validates the causal boundary; it does not validate the data pipeline, since training and validation sets may still overlap. Only then comes tiny-sample overfitting, and passing it shows only that the configuration can fit a small sample, not that it generalizes.

## What stays in the Chinese original

The Chinese note carries the full worked tables for the loss and the gradient, the alignment checklist, the averaging rules for validation and accumulation, the alignment experiment design, the two review questions, and the reference list covering the Transformers 4.57.1 GPT-2 label interface, the PyTorch 2.14 `CrossEntropyLoss` definition, and the causal mask in section 3.1 of the original Transformer paper.

Main limit, restated: every number here is a hand calculation on a toy sequence, and the source reports no trained model, no measured loss curve, and no completed alignment experiment.
