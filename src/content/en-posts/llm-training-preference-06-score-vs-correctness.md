---
sourceId: 2026-10-20-llm-training-preference-06-score-vs-correctness
slug: llm-training-preference-06-score-vs-correctness
title: "Why a Higher Preference Score Does Not Necessarily Mean a Better Task Result"
editionLabel: ENGLISH OVERVIEW · ANALYSIS NOTE
published: 2026-10-20T00:00:00.000Z
description: "How length bias and reward over-optimization separate preference win-rate from independent correctness, using one constructed pair set."
tags:
  - Reward Over-Optimization
  - Length Bias
  - LLM-as-a-Judge
  - Independent Evaluation
category: LLM Training Research
series: LLM Training Research Notes · Preference Volume
seriesOrder: 6
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on why a preference score and task correctness come apart. The six training pairs, the four evaluation pairs, the five-point checkpoint table, and every correlation coefficient below are constructed teaching setups written by the source author; they are not statistics of any dataset, not a record of any run, and not a measured relation between length, preference, and correctness. No model was trained, no real preference dataset was built, and no benchmark was measured. The article computes a shape of bias and proposes an evaluation procedure; it does not quantify how severe length bias is in practice.

## One preference pair set, two quantities

A model raises its win-rate on a batch of preference test pairs from 0.60 to 0.80. The training loss falls and the KL stays under control. Can one say the alignment improved? No. The win-rate is a score given by a judge, which is another reading of the same preference. Task correctness is a separate quantity. The two are not identical, and once optimization pressure rises they separate.

The source reuses the Cedar experiment record from the first note in this volume and the pair construction from the second, and prepares six pairs. Each pair records two things: how many tokens each side has, and whether the facts hold according to an independent check.

| Pair | chosen length | rejected length | chosen correct | rejected correct | Label |
| --- | ---: | ---: | :---: | :---: | --- |
| P1 | 42 | 30 | yes | no | chosen |
| P2 | 51 | 38 | yes | no | chosen |
| P3 | 47 | 40 | yes | no | chosen |
| P4 | 33 | 45 | yes | no | chosen |
| P5 | 28 | 31 | no | no | neither qualifies, excluded |
| P6 | 40 | 36 | yes | yes | chosen (both factually right, the difference is expression) |

P5 is excluded under the policy set in the first note and does not enter the binary preference training. Five pairs remain.

The first quantity is a length baseline. If a single rule, "pick the longer one", were used to predict the label, how many pairs would it get right? For P1, P2, P3 and P6 the chosen answer is longer; for P4 the chosen answer is shorter. The length baseline is therefore $4/5=0.8$.

The second quantity is the relation between length advantage and correctness advantage. Let the length advantage be $d=\text{len}_c-\text{len}_r$ and the correctness advantage $c=\text{correct}_c-\text{correct}_r$. Over the five pairs:

$$
d=[12,\,13,\,7,\,-12,\,4],\qquad c=[1,\,1,\,1,\,1,\,0]
$$

$$
r(d,c)=\frac{\sum (d_i-\bar d)(c_i-\bar c)}{\sqrt{\sum(d_i-\bar d)^2\sum(c_i-\bar c)^2}}\approx 0.044
$$

Length advantage and correctness advantage are almost uncorrelated ($r\approx0.04$), while a length-only predictor is 80% accurate on the same data. Those two numbers together are the core of the article: length is a predictor that is accurate here and unrelated to correctness. It can carry a reward model that looks respectable and then fail exactly where it is needed.

Look again at P4. The chosen answer is only 33 tokens, shorter than the rejected one at 45, and it is the correct one; the rejected answer writes "improved by 8 percentage points" as "improved 8 times". The length baseline picks wrong on this pair and the factual check picks right.

## Where the bias shows up on a different split

Train on the five pairs above and prepare four more pairs for testing. These four never took part in reward model training:

| Pair | A length | B length | A correct | B correct | Judge choice |
| --- | ---: | ---: | :---: | :---: | --- |
| E1 | 60 | 28 | no | yes | A |
| E2 | 55 | 25 | no | yes | A |
| E3 | 52 | 31 | no | yes | A |
| E4 | 30 | 44 | yes | no | A |

The judge picked A on all four, a win-rate of 100%. But the independent check gives this:

| Metric | Value | How to read it |
| --- | ---: | --- |
| Judge preference win-rate | 1.00 | By definition, so not evidence |
| Length baseline agrees with the judge | 0.75 | The length rule can almost replace the judge |
| Independent correctness of the answers the judge selected | 0.25 | Only E4 holds |
| Independent correctness of the answers the length rule selected | 0.00 | All four wrong |

That is the complete shape of the bias. On the training preferences the length baseline is 80% accurate, so a reward model learns length easily. On the independent split the proxy still wins 75% of the time, yet not one answer it picks is correct.

The judge is not confused on these four pairs. In E4 it picked the shorter but correct answer, so it can see an obvious error. It was simply drawn to the longer writing in E1 through E3. Judge preference is not noise; it is a structured and exploitable approximation.

## Reward over-optimization: the proxy runs ahead of the target

When optimization pressure keeps rising, the gap between the proxy reward and the true objective is not constant but widens. That is the Goodhart effect in training. Research on reward model over-optimization observed, under a fixed setup, a proxy reward that keeps rising while another evaluation signal degrades, to an extent that depends on the study's setup ([Scaling Laws for Reward Model Overoptimization, arXiv:2210.10760](https://arxiv.org/abs/2210.10760)).

A constructed teaching checkpoint table draws that curve. The five points are a constructed setup meant to show a shape, not a record of any run:

| Stage | Proxy reward | Judge preference win-rate | Independent correctness | Mean length |
| --- | ---: | ---: | ---: | ---: |
| After SFT | 0.50 | 0.50 | 0.60 | 120 |
| Stage 1 | 0.60 | 0.62 | 0.58 | 145 |
| Stage 2 | 0.69 | 0.71 | 0.49 | 190 |
| Stage 3 | 0.78 | 0.78 | 0.31 | 260 |
| Stage 4 | 0.84 | 0.80 | 0.18 | 410 |

The proxy reward rises from 0.50 to 0.84 ($+0.34$), the judge win-rate from 0.50 to 0.80 ($+0.30$), the independent correctness falls from 0.60 to 0.18 ($-0.42$), and the mean length grows to 3.42 times its original value. The correlations between these columns are:

$$
r(\text{proxy},\text{judge win-rate})=0.987,\qquad r(\text{proxy},\text{independent correctness})=-0.949
$$

$$
r(\text{length},\text{judge win-rate})=0.840,\qquad r(\text{length},\text{independent correctness})=-0.976
$$

The proxy reward and the judge win-rate move almost in lockstep ($r=0.99$) and everything looks healthy, while the proxy reward and the independent correctness are already strongly inverted ($r=-0.95$). Length is highly correlated with the judge win-rate ($r=0.84$) and strongly inverted against correctness ($r=-0.98$): length is the cheapest handle on this degradation curve. Looking only at the first two columns, no curve would raise an alarm, which is why reporting the preference score on its own carries no information.

Correlation is not causation, and the same caution applies here. The correlation between length and preference win-rate admits at least three explanations: length itself influences the judge's judgment, because the judge favors verbosity; the judge looks at some other feature that happens to correlate with length, because the answers that are genuinely preferred also tend to be longer; or both are determined by a third factor, such as answer type or task category. The column $r(\text{length},\text{independent correctness})=-0.976$ only shows that in this constructed data length and correctness run in opposite directions. The third explanation cannot be ruled out, and separating them requires a controlled design rather than a larger correlation analysis.

## Judge preference, independence, and what to report

"Judge model preference" means substituting a model for humans in deciding which answer is better. In this volume it appeared as a data source; here it is a different question: the judge is itself a biased approximation rather than a measurement of fact. MT-Bench and Chatbot Arena document position bias, verbosity bias, and self-preference bias in LLM judges, and at the same time note that strong judge models can reach a high level of agreement with human preference. Both statements have to be read together: a judge is not useless, but its preference has a direction ([MT-Bench and Chatbot Arena, arXiv:2306.05685](https://arxiv.org/abs/2306.05685)).

Upstream annotation has the same property. Human preference means "which one would you rather have, under the given conditions and criteria", and the annotation guideline itself may prefer longer, more confident, more structured expression. The length-exploitation work quantifies this: DPO exploits length as well, even when no separate reward model is trained, and its authors note the well-known verbosity preference of GPT-4 judges ([Disentangling Length from Quality in DPO, arXiv:2403.19159v2](https://arxiv.org/abs/2403.19159)). Earlier work on length correlations also reports a systematic relationship between the length and the quality of RLHF results ([A Long Way to Go: Investigating Length Correlations in RLHF, arXiv:2310.03716v2](https://arxiv.org/abs/2310.03716)). And one more mechanism carries over from the DPO note: the DPO loss constrains only the difference of two log ratios, so the absolute probabilities of both sides may fall together. The answer with the higher preference can have the lower absolute generation probability, and that is a known result ([Unintentional Unalignment, arXiv:2410.08847v4, ICLR 2025](https://arxiv.org/abs/2410.08847)).

Detecting any of these degradations requires evaluation material that took no part in training, and its independence has three levels. It must not take part in reward training: if a pair is both in the reward model's training data and in its test set, the test set measures memory. It must not take part in model selection: once a test set is used to pick a checkpoint, a beta, or a prompt template, it has become a tuning set, and the workable arrangement is three splits — a training set, a validation set for choosing the model, and a final test set used once at the end. It must not share sources with the training data: the candidates of one prompt stay in one split, and here one more rule applies, several prompts rewritten from the same original material must be separated as well.

One question helps judge whether an evaluation is really independent: if this batch of data were relabeled in a different way, would the conclusion change? If it would, what is being measured is more likely a property of the labeling procedure than a property of the task.

The most direct fix is the reporting convention. The preference win-rate and the independent correctness rate must be reported as separate rows, each with its own denominator and source, and neither may substitute for the other:

| Report item | Definition | What must be stated |
| --- | --- | --- |
| Preference win-rate | The proportion of wins decided by the judge or the preference pairs | Who the judge is, the prompt template, whether positions were swapped and re-tested |
| Independent correctness rate | The proportion judged correct by an independent check | The checking rule, how undeterminable cases are handled, whether they count in the denominator |
| Length distribution | Mean length and difference for the two kinds of answers | The truncation convention, whether pairs are length-matched |
| Undeterminable and excluded | Counts of pairs where neither side qualifies or information is insufficient | The exclusion policy and the order in which it was applied |

Four requirements accompany that table. The two rates go in different rows of the same table and are never merged into a single overall score. In the independent correctness rate, undeterminable samples stay in the denominator and are counted separately rather than removed. The length distribution is reported, because win-rates are comparable only when length is held fixed. And when the win-rate rises while correctness does not, the conclusion may only be "more preferred", never "better at the task".

The examples are constructed teaching setups: six training pairs, four evaluation pairs, one five-point checkpoint table. They show the shape of the bias and its computability, not the statistics of any real dataset. In real data the relation between length advantage and correctness advantage varies with the task, and judge-human agreement varies with the judge model, the prompt, and the domain, so the article cannot offer a general conclusion about how severe length bias is.

The control design that follows is a length-matched experiment. Take the same prompts, generate two length-comparable candidates for each, keeping both inside the same token interval, and redo the preference comparison. If after length matching the preference win-rate is essentially unchanged while the independent correctness rises clearly, length is a confound that can be removed. If the win-rate still falls after matching, the bias comes from the annotation guideline's own preference for content, and trimming length alone will not fix it. The third split is equally necessary: the reward-model training set, the validation set used for tuning, and the test set used once at the end, separated by source, with a record of which split each item appears in. The site's [preference alignment workbench](/lab/preference-alignment/) can be used to adjust beta and proxy-reward bias under fixed preferences and watch the proxy reward rise while the true utility falls back.

This is the end of the volume. The next one turns to reinforcement learning for reasoning tasks, connecting sampling, checker feedback, and policy updates into a single loop and asking what the reward actually checks.

## Where the Chinese original lives

The Chinese original is [Why a Higher Preference Score Does Not Necessarily Mean a Better Task Result: Length Bias, Over-Optimization, and Independent Testing](/posts/2026-10-20-llm-training-preference-06-score-vs-correctness/). It holds the full six-pair and four-pair tables with their annotations, the Pearson correlation between length advantage and correctness advantage, the five-point checkpoint table with all four correlation coefficients, the three levels of evaluation independence, the reporting table with its four accompanying requirements, the length-matched control design, the two review questions, and the reference list to the over-optimization study, the DPO length work, the RLHF length-correlation study, MT-Bench, and the likelihood displacement paper. Its main limit is unchanged: every table is a teaching setup, no model was trained, and the relation between reward and correctness was never measured on a real benchmark.
