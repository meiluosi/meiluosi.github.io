---
sourceId: 2026-10-20-llm-training-pretrain-05-optimization
slug: llm-training-pretrain-05-optimization-stability
title: "Why Training Becomes Unstable: Learning Rate, Accumulation, and Precision"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of learning rate, AdamW updates, gradient accumulation, gradient clipping, mixed precision, and a diagnostic order for training anomalies."
tags:
  - Optimization
  - Learning Rate
  - Training Stability
category: LLM Training Research
series: "LLM Training Research Notes · Pretraining Volume"
seriesOrder: 5
lang: en
---

> **Scope note.** This is an English overview of a Chinese technical note in the pretraining volume. Its numeric examples are hand calculations on a single parameter, a single gradient vector, and a toy schedule; no training run, sweep, or stability measurement is reported. The comparison plan the source proposes is an experiment design that has not been executed, and the source does not recommend one universal learning rate or one best numerical precision.

## From gradient descent to AdamW

The simplest update is gradient descent:

$$
\theta_{t+1}=\theta_t-\eta_t g_t
$$

where $\theta$ is the parameter, $g$ is the gradient of the current objective with respect to it, and $\eta$ is the learning rate. The gradient describes how a change in the parameters affects the objective near the current point; the learning rate determines how far the parameters move along that direction.

A single-parameter example makes the separation visible. With $L(\theta)=\theta^2/2$ the gradient is $\theta$, so the update becomes:

$$
\theta_{t+1}=(1-\eta)\theta_t
$$

Starting from $\theta_0=2$:

| Learning rate | First steps | Behavior |
| ---: | --- | --- |
| 0.5 | 2 → 1 → 0.5 → 0.25 | amplitude shrinks |
| 1.5 | 2 → −1 → 0.5 → −0.25 | crosses zero, amplitude shrinks |
| 2.5 | 2 → −3 → 4.5 → −6.75 | amplitude grows |

In this toy case only $0<\eta<2$ converges. Real neural networks have no such uniform range, and AdamW is not this bare gradient-descent rule. The point is narrow: a correct direction does not make a large step safe. In the other direction, a very small learning rate makes the curve look calm while the model learns almost nothing within a finite budget, so stability has to be read together with the speed of progress.

Adam keeps exponential moving averages of the gradient and of its square, recording roughly the recent direction and the recent scale:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

The squares and the later operations are taken per parameter component. Initializing both to zero makes the early estimates too small, so a bias correction is applied, producing $\hat m_t$ and $\hat v_t$, and the update uses a quantity of the form $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ rather than the raw gradient (the Adam paper).

One consequence follows directly: "the gradient norm doubled" does not translate into "the parameter update distance doubled". History, learning rate, and per-component scale all take part in the computation.

AdamW separates weight decay from the adaptive scaling inside the gradient. In outline:

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

where $\lambda$ is the weight decay coefficient. The original paper notes that, for this family of adaptive optimizers, adding the L2 term to the objective is not equivalent to the decoupled treatment (Decoupled Weight Decay Regularization). A concrete implementation also has to state which parameters are decayed, whether parameter groups share a learning rate, and which numerical options are in use; an experiment should record the actual version and parameters rather than only the word "AdamW" (the PyTorch 2.9 AdamW interface).

## Accumulation, batch size, and warmup

To use eight sequences in one update when memory holds only two, one can run four forward and backward passes, accumulate the gradients, and update the parameters once. Under matching target weighting and unchanged parameters during accumulation, summed gradients express the gradient of the same large batch. Actual floating-point order, random dropout, or cross-sample operations can still make the result differ, and it is not the same as four consecutive parameter updates.

Two misoperations are the most common. Either the parameters are updated after every micro-batch, which changes the update frequency, or the accumulated gradients are zeroed before each backward pass, which leaves only the last micro-batch. When the micro-batches hold different numbers of effective targets, the mean losses have to be weighted by those counts, as derived in the previous note; dividing by the accumulation count is not automatically right.

Accumulation usually lowers the activation memory held at the same time, but the parameters, the gradients, and the optimizer state still have to be stored. If a model does not fit with those states alone, adding accumulation steps does not make it fit.

A larger effective batch also changes the fluctuation of the gradient estimate and the number of updates under a fixed token budget. With a total budget of one million effective targets, using one thousand per update gives about one thousand updates, while using four thousand per update gives about two hundred fifty. If the learning-rate schedule was designed against the old step count, the rhythm of the whole run has changed. A comparison should therefore report at once the effective targets per update, the cumulative effective targets, the number of updates, and which counter the schedule follows. Rules of thumb that scale the learning rate with the batch size exist for certain settings, but they are not a theorem for any model and budget; the reasonable move is a limited comparison near an already stable baseline, read against validation results under a fixed budget.

Warmup raises the learning rate gradually to a preset level at the start of training, which softens aggressive early updates, after which the rate may be lowered by a fixed rule. In the teaching design, the first one hundred updates warm up linearly to a maximum $\eta_{\max}$, and update $t$ uses:

$$
\begin{aligned}\eta_t&=\eta_{\max}\min(t/100,1),\\&\qquad t\ge1.\end{aligned}
$$

So the first update uses one percent of the maximum and the hundredth reaches it. Whether the step counter starts at zero or at one, and whether the scheduler advances before or after the update, are details that shift the schedule by a full step. Changing the effective batch also means the same one hundred updates consume a different number of tokens, so a warmup reported only as a percentage has to state whether its denominator is planned steps or a token budget. Warmup does not repair shifted labels, a wrong mask, or a validation set full of duplicates: when a curve misbehaves, confirm the task and the numbers before touching the schedule.

## Precision, loss scaling, and clipping

Numbers in training have to be represented with a finite number of bits. FP32, FP16, and BF16 make different trade-offs between representable range and fineness. BF16 has a wider exponent range than FP16, but not every value is represented more precisely. Mixed precision is also not the same as converting every tensor to half precision: computation types are usually assigned per operator, and some states or reductions stay in higher precision. The exact behavior depends on the device, the framework, and the implementation.

In FP16 training, small gradients can be hard to retain. Loss scaling multiplies the loss so that the backward pass produces correspondingly scaled gradients, which are unscaled before the update; it does not permanently raise the effective learning rate. When gradient scaling is used, the scaling convention has to stay consistent through accumulation, and the unscaling, the checks, and the update happen only after accumulation is complete. The PyTorch AMP examples state this order and note that a non-finite gradient can cause a parameter update to be skipped.

Logs should therefore distinguish how many batches were read from how many optimizer updates actually ran. If many updates are skipped while the schedule and the budget keep advancing, the curve becomes uninterpretable. The source does not prescribe a best precision for any particular device. The reasonable order is to establish a runnable baseline with finite numbers first, and then compare precision options with the same samples and the same budget; FP32 is not the mathematically correct answer either, since it rounds as well.

Global gradient-norm clipping can be written as:

$$
g'=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right)
$$

If the norm does not exceed the threshold $c$, the gradient is unchanged; if it does, the whole group is scaled by one factor. A zero gradient stays zero, and an implementation has to avoid dividing by zero. For example, $g=(3,4)$ has norm 5, and with a threshold of 2 the clipped vector is $(1.2,1.6)$: the direction is preserved and the norm becomes 2. This is a global norm computed over all selected parameters, not a per-component truncation (the PyTorch 2.9 clipping interface).

With gradient scaling, the gradients must be unscaled before being clipped against the real threshold, otherwise the threshold applies to inflated numbers. Clipping usually belongs after a full effective batch has been accumulated; clipping each micro-batch and then adding the results is a different update rule. Frequent clipping deserves investigation: the learning rate or the data distribution may be wrong, or the threshold may simply be set low. It is not automatic proof of a gradient explosion, and a clipping formula cannot be relied on to repair a gradient that is already NaN.

## Preserve evidence that can localize a failure

Recording only an average loss is rarely enough. Each update should at least carry the data batch identifier, the effective target count, the learning rate, the loss, the gradient norm, and whether the update ran; for the norm, state whether it was recorded after unscaling, before clipping, or after clipping.

| Symptom | Check first | Conclusion to avoid |
| --- | --- | --- |
| Non-finite values on the first step | Inputs, labels, empty targets, logits, and the loss computation | The model is too large, so it is necessarily unstable |
| A sudden spike in one batch | That batch's length, source, target count, and unusual text | All the data needs re-cleaning |
| Training loss falls, validation worsens | Overfitting, distribution shift, evaluation protocol | The optimizer has failed |
| The loss is flat and barely moves | Whether gradients exist, whether parameters update, the learning rate | The data has already been learned |
| Updates are skipped frequently after a precision change | Numeric range, scaling state, the operators involved | Low precision is unsuited to this task |

When an anomaly appears, save the most recent valid checkpoint together with the batch information and reproduce over a short range. Do not change the data, the learning rate, the batch, and the precision in the same round, because if the problem disappears there is no way to tell which change did it.

A comparable study fixes the tokenizer, the data order, the model initialization, the target normalization, and the effective batch, then selects a baseline that runs stably. Around one question — for instance, whether the current learning rate is too large — compare half the baseline, the baseline, and twice the baseline under the same effective token budget, with stop conditions agreed in advance: non-finite values, sustained absence of updates, or resource limits. Keep the failed settings instead of only the surviving curve. Only then study precision or batch size separately. When reading the validation loss, also track the effective tokens and the wall-clock time it took to reach that value, because the small difference between a single random seed does not demonstrate a reliable gain.

This is an experiment design. It does not mean these configurations were run on this site, and conclusions drawn after actually running them have to carry the environment, the logs, and the repetitions.

## What stays in the Chinese original

The Chinese note carries the full update equations, the per-parameter learning-rate table, the diagnostic table with its symptom-to-check mapping, the planned comparison protocol, the two review questions, and the reference list covering the Adam paper, the decoupled weight decay regularization paper, the PyTorch 2.9 AdamW interface, the PyTorch 2.9 AMP examples, and the PyTorch 2.9 gradient-norm clipping interface.

Main limit, restated: every number here is a hand calculation on toy objects, and the source reports no training run, no learning-rate comparison, and no measured stability outcome.
