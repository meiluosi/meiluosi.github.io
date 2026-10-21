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

> **Scope note.** This is an English overview of a Chinese article on direct preference optimization. The two-token worked example and every loss, gradient, and KL value below are hand-computed teaching calculations written by the source author; they are not measurements of any model. No preference data was collected, no model was trained, and no comparison between DPO and PPO was run on any task. The article explains a mechanism and proposes a control design; it does not establish that direct preference optimization improves a task.

## From the KL-constrained objective to a closed-form preference loss

InstructGPT splits alignment into supervised fine-tuning, reward modeling, and reinforcement learning, with the third stage optimizing a KL-constrained reward objective ([InstructGPT, Section 3](https://arxiv.org/html/2203.02155v1#S3)):

$$
\max_{\pi_\theta}\;\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(y\mid x)}\big[r_\phi(x,y)\big]-\beta\,\mathrm{KL}\big[\pi_\theta(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)\big]
$$

Here $\pi_{\mathrm{ref}}$ is the reference policy, usually the fine-tuned model $\pi^{\mathrm{SFT}}$, and $\beta$ sets how far the policy may deviate from it. DPO takes this objective as its starting point and names two problems with it: reward-model fitting error propagates into the policy update, and actor-critic training is unstable in practice ([DPO, Sections 3 and 5](https://arxiv.org/html/2305.18290v3#S3)).

The step that makes the method possible is that this constrained problem has a closed-form optimum:

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\exp\big(r^*(x,y)/\beta\big)
$$

Taking logs and rearranging writes the reward as a ratio of two policies:

$$
r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)
$$

This is the pivot of the whole derivation. The reward model stops being a separate network that has to be fitted and becomes a function that can be written down, and it says something specific: at the optimum, the reward is the same thing as how far the policy moved relative to the reference.

Bradley-Terry then turns "preference" into a probability ([Bradley and Terry, Biometrika 39(3-4):324-345, 1952](https://doi.org/10.1093/biomet/39.3-4.324)):

$$
P(y_1\succ y_2\mid x)=\sigma\big(r^*(x,y_1)-r^*(x,y_2)\big),\qquad \sigma(z)=\frac{1}{1+e^{-z}}
$$

Substituting the reparameterized reward, the partition function $Z(x)$ cancels in the difference, and the maximum-likelihood objective becomes a binary cross-entropy:

$$
\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\big(\beta\,\Delta_\theta(y_w)-\beta\,\Delta_\theta(y_l)\big)\Big],\quad \Delta_\theta(y)=\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

That is the entire loss. There is no sampling, no reward model, and no value function; each training example needs one forward pass through the policy and one through the reference model ([DPO, Section 4 and Appendix A.2](https://arxiv.org/html/2305.18290v3#S4)). Two substitutions happened along the way: the reward became a policy ratio, and "generate then score" became "score an existing pair as a classification problem".

## The reference policy is part of the objective

$\pi_{\mathrm{ref}}$ appears in the numerator of the log ratio and in the denominator. It is not an initialization trick; it is part of what the loss measures.

Remove it and the objective degenerates into "make the probability of the chosen answer exceed that of the rejected one", which a model can satisfy by pushing down the probability of every sequence at once. Keep it, and the loss measures relative displacement: whether the policy moved probability mass onto the chosen answer from somewhere else.

In implementation the reference model is normally frozen. TRL 0.23.1's `DPOTrainer` uses a same-architecture model as the reference when no `ref_model` is passed, and offers `precompute_ref_log_probs` to compute the reference log-probabilities in advance so the reference model need not be loaded during training ([TRL 0.23.1 DPO Trainer](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)).

Which reference is chosen changes what the result means. Using the SFT model as reference makes the objective "adjust toward preference without breaking SFT behavior". The DPO paper also notes that when no SFT model is available, $\pi_{\mathrm{ref}}$ can instead be built by maximizing the likelihood of the chosen answers, which introduces a bias between the reference distribution and the true reference distribution ([DPO, Section 4](https://arxiv.org/html/2305.18290v3#S4)).

The reference is also what keeps the loss from being about absolute probability at all. The loss constrains only the difference of two log ratios, so the absolute level is free. Take the worked example below and multiply the policy probabilities by 0.9:

$$
\pi_\theta'(y_w)=0.378,\quad \pi_\theta'(y_l)=0.198
$$

The ratios become 1.26 and 0.99, the gap $h$ is still $\log(14/11)$, and the loss at $\beta=0.1$ is still 0.6812. Yet the absolute log-probability of both sides fell by $\log 0.9\approx-0.1054$. The answer the model prefers can become less likely to be generated. This is the phenomenon named likelihood displacement ([Unintentional Unalignment, arXiv:2410.08847v4, ICLR 2025](https://arxiv.org/abs/2410.08847)).

## A two-token loss, computed by hand

The worked example uses a vocabulary of two tokens and two-token replies: the chosen answer is "AB" and the rejected one is "BA". For one prompt, the policy and the reference give these conditional probabilities:

| Position | $\pi_\theta$ | $\pi_{\mathrm{ref}}$ |
| --- | --- | --- |
| First token | A 0.6, B 0.4 | A 0.5, B 0.5 |
| After A | A 0.3, B 0.7 | A 0.4, B 0.6 |
| After B | A 0.55, B 0.45 | A 0.4, B 0.6 |

The two sequence probabilities follow directly:

$$
\pi_\theta(y_w)=0.6\times0.7=0.42,\quad \pi_\theta(y_l)=0.4\times0.55=0.22
$$

$$
\pi_{\mathrm{ref}}(y_w)=0.5\times0.6=0.30,\quad \pi_{\mathrm{ref}}(y_l)=0.5\times0.4=0.20
$$

The ratios are 1.4 and 1.1, so the log ratios are $\Delta_w=\log 1.4\approx0.3365$ and $\Delta_l=\log 1.1\approx0.0953$, and their difference is

$$
h=\Delta_w-\Delta_l=\log\frac{1.4}{1.1}=\log\frac{14}{11}\approx0.2412
$$

Substituting into $\ell=-\log\sigma(\beta h)$ gives the table below. Every entry is a hand calculation, not a training measurement.

| $\beta$ | $\beta h$ | $\sigma(\beta h)$ | $\ell$ | $\partial\ell/\partial h$ |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.0241 | 0.5060 | 0.6812 | $-0.0494$ |
| 0.5 | 0.1206 | 0.5301 | 0.6347 | $-0.2349$ |
| 1.0 | 0.2412 | 0.5600 | 0.5798 | $-0.4400$ |

The $\beta=1$ row can be checked entirely by hand: $\sigma(\log(14/11))=14/25=0.56$, and the loss is $-\log(14/25)\approx0.5798$.

The gradient direction can be checked the same way. Since

$$
\frac{\partial\ell}{\partial h}=-\beta\big(1-\sigma(\beta h)\big)
$$

it is negative for every $\beta>0$, so gradient descent always pushes $h$ upward, raising the chosen answer relative to the rejected one. Evaluated across $\beta=0.05,0.1,0.25,0.5,1,2$, the sign is consistent and the magnitude grows monotonically from 0.0248 to 0.7634.

## What beta controls, and what the derivation assumes

Beta is usually waved away as "the KL coefficient". It appears in two places, and they have to be separated.

The first is the KL penalty in the original objective. Given a reward function, the optimal policy satisfies

$$
\pi^*(y\mid x)\propto\pi_{\mathrm{ref}}(y\mid x)\exp\big(r(y\mid x)/\beta\big)
$$

so a larger $\beta$ weakens the exponential tilt and keeps the optimum closer to the reference. With two replies, a reward difference of 1, and reference probabilities $(0.6,0.4)$, the arithmetic gives:

| $\beta$ | $\pi^*(y_w)$ | $\mathrm{KL}(\pi^*\|\pi_{\mathrm{ref}})$ |
| ---: | ---: | ---: |
| 0.1 | 0.99997 | 0.5105 |
| 1.0 | 0.8030 | 0.0945 |
| 10 | 0.6237 | 0.0012 |

The second is the scaling inside the loss. With the policy state fixed, $\partial\ell/\partial h$ is proportional to $\beta$; in the table above it moves from $-0.0494$ to $-0.4400$, close to a ninefold difference.

Put together, beta is not an "alignment strength" knob. It scales the implicit reward and the KL penalty at the same time, and what is actually set is the ratio between reward scale and permitted deviation. TRL 0.23.1's default for `beta` is 0.1, and its documentation describes the parameter as the degree of deviation from the reference model, where a high `beta` means less deviation ([TRL 0.23.1 DPO Trainer](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)).

The formula also acts on a whole sequence, not on individual positions:

$$
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t})
$$

A preference is a judgment about an entire reply, not about one position. Comparing token by token would require answering "which token carried this preference", and the preference data contains no answer to that. The sum spreads the responsibility evenly across positions: in the example above, each of the two tokens carries half of $\Delta$. This is a choice rather than a necessity. TRL 0.23.1 notes in its list of losses that the IPO variant averages the log-likelihood over the completion while standard DPO sums it, and the two normalizations change the gradient scale on long replies ([TRL 0.23.1 DPO Trainer](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)).

The derivation rests on assumptions that have to be stated: that preference is captured by the Bradley-Terry form, that the expected reward does not depend on the policy, that the optimum has a closed form, and that the reference policy is close enough to the distribution under which the data was collected. If any one of them fails, the conclusions need to be re-examined. A closed-form solution also means DPO optimizes a surrogate objective rather than the KL-constrained problem itself; the two agree at the optimum and need not agree far from it.

Three known failure modes follow from the same structure. Distribution shift: the preference data was sampled and annotated under the reference policy, and once training moves the policy away from that region, the labels on it are no longer reliable; the DPO paper itself notes that in practice people reuse public preference datasets rather than resampling and recollecting human preferences ([DPO, Section 4](https://arxiv.org/html/2305.18290v3#S4)). Length exploitation: dedicated work analyzes the length problem in DPO, links length exploitation to out-of-distribution bootstrapping, and notes the well-known verbosity preference of GPT-4 judges ([Disentangling Length from Quality in DPO, arXiv:2403.19159v2](https://arxiv.org/abs/2403.19159)). And preference is not correctness: Bradley-Terry describes a probability of choice, not a rate of factual accuracy, and the boundary drawn in the previous note still holds here.

The control design the source proposes follows from the displacement result. To separate "DPO raised the absolute probability of the chosen answer" from "DPO only raised the chosen answer relative to the rejected one", record $\log\pi_\theta(y_w)$ and $\log\pi_\theta(y_l)$ separately after training, not only their difference. If the difference rises while both absolute values fall, the model learned a relative ranking rather than a higher chance of generating a better answer, and reporting a single preference win-rate hides exactly that.

## Where the Chinese original lives

The Chinese original is [Why DPO Can Use Preference Pairs Directly: Objective, Reference Policy, and Beta](/posts/2026-10-20-llm-training-preference-04-dpo/). It holds the full derivation from the constrained optimum through the Bradley-Terry substitution, the complete two-token probability table with its hand-computed loss and gradient rows, the KL table over beta, the sequence-normalization discussion with the IPO variant, the failure-mode discussion and its citations, the two review questions, and the reference list to InstructGPT, DPO, Bradley and Terry, and TRL 0.23.1. Its main limit is unchanged: every number is a teaching calculation, and no preference training was run. The next note in this volume asks what the closed-form objective can reach once the answer pool is no longer fixed.
