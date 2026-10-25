---
sourceId: 2026-10-20-llm-training-preference-04-dpo
slug: llm-training-preference-04-dpo
title: "Why DPO Can Use Preference Pairs Directly: Objective, Reference Policy, and Beta"
editionLabel: ENGLISH OVERVIEW · THEORY NOTE
published: 2026-10-20T00:00:00.000Z
description: "How a KL-constrained reward maximum becomes a closed-form preference loss, and what the reference policy and beta actually control."
tags:
  - Direct Preference Optimization
  - Preference Learning
  - Reference Policy
  - Likelihood Displacement
category: LLM Training Research
series: LLM Training Research Notes · Preference Volume
seriesOrder: 4
lang: en
---

> **Scope note.** The probability examples and finite-difference checks are executed teaching calculations, not model training results.

## What DPO replaces

Standard offline DPO fits preferences on existing pairs without generating new answers at every update or separately fitting a reward or value network. It needs policy and reference log-probabilities for both completions; reference values can be precomputed.

For a fixed reward, positive beta, reference support, and a finite normalizer, the unrestricted KL-regularized optimum is (conditioning on a fixed prompt x is suppressed below):

$$
\begin{aligned}
\pi_r(y)&=\frac{\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}}{Z},\\
Z&=\sum_y\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}.
\end{aligned}
$$

The reference factor is essential. Rearranging gives a log policy ratio plus a prompt-only constant. That constant cancels in a Bradley–Terry reward difference, yielding the [DPO objective](https://arxiv.org/html/2305.18290v3#S4):

$$
\begin{aligned}
\Delta_\theta(y)&=\log\frac{\pi_\theta(y)}{\pi_{\mathrm{ref}}(y)},\\
h&=\Delta_\theta(y_w)-\Delta_\theta(y_l),\\
\ell&=-\log\sigma(\beta h).
\end{aligned}
$$

Finite data, limited model capacity, optimization error, and missing response coverage still matter. The reward function is fixed in this derivation; its expectation is not independent of the policy.

## Check the two-token example

Chosen and rejected policy probabilities are 0.42 and 0.22; reference probabilities are 0.30 and 0.20. Thus h=log(14/11). At beta=1, the preference probability is 0.56 and the loss is approximately 0.5798.

The scalar derivative is −beta·sigmoid(−beta·h). It favors increasing h as an independent coordinate, but shared neural parameters and other examples prevent a guarantee that every pair improves.

For this positive h, the gradient magnitude is about 0.823 at beta=10 but only 3.36×10⁻⁹ at beta=100. It is neither globally increasing nor simply proportional to beta. Separately, with a fixed reward, increasing beta weakens the exponential tilt toward high-reward answers.

## Sequence sums do not divide responsibility equally

The chosen token contributions are log(0.6/0.5)≈0.1823 and log(0.7/0.6)≈0.1542. They sum to 0.3365 but are not equal. Parameter gradients additionally depend on each token's network derivatives.

[TRL 0.23.1](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer) documents reference precomputation and different loss variants. Changing sequence summation to length averaging changes the objective; do not treat it as an innocuous implementation detail.

## Relative improvement is not absolute improvement

Scale only the two compared policy probabilities by 0.9 to obtain 0.378 and 0.198. Move the released mass to the other two outcomes, changing each from 0.18 to 0.212. The distribution remains normalized, h is unchanged, and both compared probabilities decrease.

The reference does not prevent this behavior. [Likelihood displacement](https://arxiv.org/abs/2410.08847v4) studies decreasing preferred-response likelihood; the toy example alone does not establish how a real training run redistributes probability.

Old labels do not automatically become invalid as the policy changes. Rather, newly generated responses may lack coverage in the fixed comparisons. Track chosen and rejected absolute likelihoods, held-out preferences, and independent task performance separately.

The [Chinese article](/posts/2026-10-20-llm-training-preference-04-dpo/) includes the full conditional-probability table, beta calculations, assumptions, and a proposed controlled comparison. No DPO-versus-PPO training comparison was run.
