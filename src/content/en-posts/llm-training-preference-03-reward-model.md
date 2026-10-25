---
sourceId: 2026-10-20-llm-training-preference-03-reward-model
slug: llm-training-preference-03-reward-model
title: "How a Reward Model Learns from Comparisons: Score Gaps, Probabilities, and Limits"
editionLabel: ENGLISH OVERVIEW · THEORY NOTE
published: 2026-10-20T00:00:00.000Z
description: "What a Bradley-Terry reward score means, why ranking loss is not correctness, and where score calibration and reward overoptimization break."
tags:
  - Reward Modeling
  - Preference Learning
  - Bradley-Terry Model
  - Calibration
category: LLM Training Research
series: LLM Training Research Notes · Preference Volume
seriesOrder: 3
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on reward modeling from pairwise preferences. The probability and loss values below are hand-computed teaching calculations for a toy model, and the example of a model predicting 0.9 against an observed 60 percent is an illustration of miscalibration, not an observed result. No reward model was trained, no preference dataset was collected, and no evaluation was run; the article explains a mechanism and proposes a first experiment design.

## A scalar score, and the probability it implies

The reward model takes the shared input $x$ and a candidate answer $y$ and returns

$$
r_\phi(x,y)\in\mathbb{R},
$$

where $\phi$ denotes the reward model's parameters. Unlike a generative model, which produces vocabulary scores at every position, this is one scalar for a whole answer.

One implementation attaches a scalar output head to a language model representation, taking a selected position or pooling the representation by a stated rule. Which position is used, and how end markers and padding are handled, has to be checked against the implementation; "the last tensor slot" cannot be assumed to be the end of the answer.

InstructGPT's reward modeling stage uses language-model-based scalar prediction and learns from comparisons under the same prompt ([InstructGPT, Section 3.5](https://arxiv.org/html/2203.02155v1#S3.SS5)).

A reward model can read a complete answer and then score it. That does not mean a generative model can see the future of its own text while writing it. The two carry out different computational tasks.

Now take two answers A and B with scores $r_A$ and $r_B$. The usual Bradley-Terry form is

$$
P(A\succ B\mid x)=\sigma(r_A-r_B),
$$

where $\sigma(z)=1/(1+e^{-z})$ is the sigmoid. It converts an arbitrary real score gap into a number between zero and one. The DPO paper uses this form when it sets up the preference modeling background ([DPO, Section 3](https://arxiv.org/html/2305.18290v3#S3)).

The intuition is simple: the more A's score exceeds B's, the more the model predicts that the annotators chose A, and when the two are equal the predicted probability is 0.5.

| Score gap $r_A-r_B$ | Predicted probability that A is chosen |
| ---: | ---: |
| $0$ | 0.5 |
| $\log3$ | 0.75 |
| $\log9$ | 0.9 |
| $-\log3$ | 0.25 |

These are calculations of a probability model. They are not the probability that A is factually correct. Only after a calibration check on the distribution where the model applies is there reason to read the number as an approximate prediction of observed choice frequency.

## The ranking loss, and the gradient it implies

For a record labeled `chosen` over `rejected`, let the score gap be

$$
\Delta=r_\phi(x,y_c)-r_\phi(x,y_r).
$$

The negative log-likelihood loss is

$$
\ell=-\log\sigma(\Delta)=\log(1+e^{-\Delta}).
$$

When $\Delta=0$, the loss is $\log2\approx0.6931$; when $\Delta=\log3$, it is about $0.2877$; and with the direction reversed, $\Delta=-\log3$, it is about $1.3863$.

Training therefore does more than check the ordering. It also scores the probability that the model assigns to the label. A model that places `chosen` slightly ahead has a different loss from one that places it ahead with high confidence.

The derivative of the loss with respect to the gap is

$$
\frac{\partial\ell}{\partial\Delta}=\sigma(\Delta)-1.
$$

In a local analysis that treats the two scores as independent variables, gradient descent tends to raise the `chosen` score and lower the `rejected` score. In a real network the parameters are shared, so the same update also affects other inputs, and the local direction cannot be read as a guarantee about every answer.

Numerically, implementations usually use a stable log-sigmoid or softplus, to avoid the overflow that comes from exponentiating a large number and then taking a logarithm.

## Translation invariance, scaling, and the limits of one scalar

If a constant $c(x)$ is added to every score for the same prompt, the difference is unchanged:

$$
\begin{aligned}r_A^{\prime}&=r_A+c(x),\\r_B^{\prime}&=r_B+c(x),\\r_A^{\prime}-r_B^{\prime}&=r_A-r_B.\end{aligned}
$$

So the pairs (2,1) and (102,101) give the same pairwise probability. A score of 102 cannot be said to be fifty-one times better than a score of 2 on another prompt.

More strictly, a pairwise comparison objective cannot determine the common offset for each prompt on its own. A particular parameterization and regularization may impose some numerical convention, but that is not the same as the data providing an absolute scale that is comparable across prompts.

Scaling is different. Multiplying both scores by two turns a gap of 1 into 2, and the sigmoid probability changes accordingly. If a temperature parameter is introduced, scale and temperature become related as well. Translation invariance must not be miswritten as a claim that any transformation leaves the probability unchanged.

This is one reason reward scores need care when they feed a later optimization stage: ordering, probability calibration, and optimization scale are related but distinct problems.

A scalar is also not an objective sum over all quality dimensions. The task at hand cares about factual support, brevity, and usefulness at once. If the data compresses those trade-offs into one win/lose label, the reward model learns that combined choice, and it has no way to know which conditions are non-compensable. If the training data often prefers long, confident answers, the model can learn that trade-off even when a mild unsupported assertion is present.

A scalar carries an expressiveness assumption as well. For a fixed prompt, if the score satisfies A above B and B above C, then A must be above C; but real group comparisons can be cyclic or strongly divided when different trade-offs are in play. A simple scalar model cannot express arbitrary comparison relations. That is not an argument against pairwise models; it is a reminder to inspect the label structure first rather than demand that one score reproduce everyone's preferences.

## Soft labels, ties, and calibration

If a pair receives several independent judgments and A is chosen with proportion $q$, a soft-label cross-entropy can be used:

$$
\begin{aligned}\ell_q&=-q\log p\\&\quad-(1-q)\log(1-p).\end{aligned}
$$

where $p=\sigma(r_A-r_B)$. When the probability for the pair is treated as an independently adjustable quantity and no other constraint is imposed, the optimum is $p=q$.

A $q$ obtained from a small number of votes carries uncertainty and is not a precisely known population probability. When annotators are not independent, the task instructions differ, or the sample is selected, the votes cannot simply be pooled.

"Tie" and "insufficient information" should not be encoded alike as 0.5. The first may mean the two answers are comparable; the second means no judgment is possible. Any conversion or model choice has to be consistent with the feedback definition.

Ranking accuracy can also be high while confidence is unreliable. Suppose a reward model predicts a 0.9 probability of A winning for a whole set of answers, while only about 60 percent of them are actually chosen as A. It may have some ordering ability while being clearly overconfident.

Evaluation should therefore separate at least three things: pairwise accuracy, negative log-likelihood, and the actual choice frequency within bins of predicted probability. A calibration check needs enough samples and has to preserve source grouping; a few records cannot support a stable conclusion.

Error slices matter as well: length differences, model provenance, material type, obvious factual errors, and fine expression trade-offs. Overall accuracy may be carried mostly by easy gibberish pairs, while the difficult comparisons that matter remain poor.

The pinned TRL reward training documentation provides the training interface and uses pairwise accuracy as its default evaluation metric. The grouping, the calibration check, and the error slicing still have to be implemented separately; a trainer's defaults do not cover them. These metrics support diagnosis, and they do not replace independent judgment at the task level ([TRL 0.23.1 reward modeling](https://huggingface.co/docs/trl/v0.23.1/en/reward_trainer)).

## Distribution shift and what a first experiment should check

A reward model is trained on candidates from a fixed set of models and sampling settings. Later optimization actively looks for high-scoring answers and produces content that is longer, more repetitive, or better at exploiting the scoring regularities.

Accuracy on the original preference test set therefore does not guarantee that the score remains reliable after optimization. Research on reward model overoptimization describes a proxy reward continuing to improve while another evaluation signal degrades; the extent depends on that study's setup and does not mechanically generalize to every task ([Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)).

The countermeasure is not to watch the training reward rise, but to keep independent material checks, human review, and evaluation over different candidate sources.

What a first reward-model experiment should verify is therefore concrete. Check the loss, the sign, and the gradient direction with hand-computable scores first, then check the shared prompt, the end-of-answer position, and padding on real inputs. Swapping `chosen` and `rejected` should flip the predicted preference, rather than the field names themselves entering the model and affecting the result.

Split the data by prompt and by original source, so that the same candidate cannot appear on both sides of the split. Report how many candidates each prompt generated, how many pairs it expanded into, and how training weighted them.

Then compare against a simple baseline, such as predicting preference from answer length alone. If a complex model mostly reproduces that baseline's regularity, the next step is to check whether the labels contain a length shortcut, not to announce that the model has learned to judge quality.

This article completes a mechanism explanation and an experiment design. No reward model was trained, and scaling up is only worth doing after the data definitions and these checks are settled.

## Where the Chinese original lives

The Chinese original is [How a Reward Model Learns from Comparisons: Score Gaps, Probabilities, and Failure Boundaries](/posts/2026-10-20-llm-training-preference-03-reward-model/). It holds the full derivations, the score-gap table with its worked values, the soft-label objective and its conditions, the calibration and error-slicing discussion, the first-experiment checklist, and the reference list to InstructGPT, DPO, the overoptimization study, and TRL 0.23.1. Its main limit is unchanged: every number is a teaching calculation, and no reward model was trained. The next topic in this volume is whether the preference relations can be learned directly from the generator and a reference model, without training this explicit reward model.
