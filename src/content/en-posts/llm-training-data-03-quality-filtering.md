---
sourceId: 2026-10-15-llm-training-data-03-quality
slug: llm-training-data-03-quality-filtering
title: "What Counts as High-Quality Pretraining Data: From Rules to Model Scores"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-15T00:00:00.000Z
description: "Four ways to score pretraining text — reference classifiers, distilled scorers, perplexity, and resampling — and what each one actually measures."
tags:
  - Pretraining Data
  - Data Quality
  - Filtering
category: LLM Training Research
series: LLM Training Research Notes · Data Volume
seriesOrder: 3
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on quality filtering. The data pool, the retention counts, and the probability values in it are self-posed teaching examples, and the perplexity figures come from a set probability construction rather than from any web statistics. The DCLM, FineWeb-Edu, CCI3.0-HQ, DSIR, and Nemotron-CC results are reported by those sources under their own settings. No text was scored locally and no model was trained, so nothing here shows that any scorer improves a given model.

## Write the standard before choosing the scorer

The opening problem is a budget: a tutorial on gradient descent, an accurate API note, a fluent product announcement, and a set of technical forum threads are all readable, and only half can be kept. "High quality" cannot be reduced to well written; it has to be asked relative to what this training run is for.

The source splits the vague criterion into observable questions: is the content complete (no missing steps, no truncated tables or code), is it specific (facts, conditions, steps, or examples rather than generalities), is it educational (does it help a reader understand a concept or process), is it in the domain the model needs, is it credible (are key claims supported and free of obvious errors), and does it add coverage that the current data lacks. One score cannot answer all of them: an error log has no tutorial structure and is still the right material for a debugging model, while a long tutorial can be elegantly structured and explain a concept incorrectly.

A practical first version therefore filters clearly damaged text and then records educational value, domain, and genre separately, so that a later change of task can reselect rather than permanently discard everything that scored low.

## Four ways to score text, and what each one measures

### Method one: a classifier that learns what reference material looks like

The cheap approach is to take two sets of text — one from chosen reference material, one from ordinary web pages — label them, train a classifier, and score the large pool with it. The attraction is cost: the classifier never has to explain an article, it only has to learn the difference between the two groups.

DCLM compared different reference sources and reported a good experimental result for a configuration using OpenHermes 2.5 and high-scored ELI5 posts. Those instruction data were used to train the filter; they are not a conversion of the pretraining pool into conversational data.

What the classifier has actually learned is the open question, and the source states it plainly: looking more like reference material is not the same as being more true or more useful. If every positive example is a textbook, the classifier may simply prefer textbook prose; if positives share a fixed title format, it may have memorized the format.

The idealized relationship makes the dependence on the reference distribution explicit. Let $p_+(x)$ be the probability that text $x$ appears in the reference material, $p_-(x)$ the probability that it appears in the background web, and $\pi$ the proportion of positives in training. For a classifier that estimates the positive probability perfectly, Bayes' rule gives

$$
\frac{s(x)}{1-s(x)}
=\frac{\pi}{1-\pi}\frac{p_+(x)}{p_-(x)}.
$$

The left side is the odds of being called positive. The right side says those odds depend both on the positive/negative sampling ratio and on which group the text resembles. With equal positive and negative sampling and a well-calibrated model, $s(x)=0.8$ corresponds to odds $0.8/0.2=4$, meaning the text is relatively more common in the reference distribution under that idealization. It does not mean the article is 80 percent factually correct. Real classifiers add label noise, capacity limits, and calibration error, so an arbitrary score is not an exact probability ratio. The derivation's use is to show that the choice of positive examples is a statement of what the filter should favor.

Shortcuts can be probed directly: keep the body and change the title format, drop the source marker, or take the validation set from a different site than the training set. If changing only the title form moves the score substantially, the classifier is reading format, and raising the threshold then buys little content quality.

### Method two: score with a large model, then distill

The other route is to write a scoring rubric, have a large model score a batch, and train a small model to imitate those scores. The rubric captures the large model's ability to handle complex requirements without paying the same inference cost on every page. FineWeb-Edu uses this educational-value filtering idea, and CCI3.0-HQ uses Qwen2-72B-Instruct to produce educational labels, training a regression head on frozen BGE-M3 representations before filtering; the paper's model experiments carry specific conditions such as roughly 0.5B parameters and 100B training tokens, and cannot be restated as "educational data is necessarily better".

Two gates stand in the way. The first is whether the large model understood the rubric. The second is whether the small model learned the large model's judgments. A student that imitates well has only cleared the second gate: if the teacher misjudged a professional document, the student inherits that error, so a high agreement rate between student and teacher is not sufficient evidence that the method works end to end. For Chinese technical text, the rubric should reward conditions, operational steps, and necessary limitations rather than term density and length, or a short precise answer will lose to a long empty one.

### Method three: how hard is the text to predict

Perplexity-based filtering asks how surprising a text is to a reference language model. If the model assigns low probability to the true next token, the average loss rises and so does the perplexity:

$$
\begin{aligned}\ell(x)&=-\frac1T\sum_{t=1}^{T}\log q_t,\\q_t&=p_{\mathrm{ref}}(x_t\mid x_{<t}),\\\operatorname{PPL}(x)&=e^{\ell(x)}.\end{aligned}
$$

Here $T$ is the number of valid prediction positions and $p_{\mathrm{ref}}$ is the reference model's probability for the true token. The negative log turns "high probability for the correct answer" into "small loss", and averaging over positions stops long documents from dominating through sheer length. To read the scale, the source uses a constructed example: if every true token receives probability 0.5, the perplexity is 2, and if every token receives 0.25, it is 4. These are stipulated probabilities, not statistics of real pages.

The difficulty is that easy does not mean useful. A repeated template is trivially predictable, while unfamiliar technical terminology is hard to predict and may be exactly what the model should learn. Keeping only the lowest-perplexity text is therefore a poor default; both ends of the distribution should be inspected, with the reference model, tokenizer, and context handling fixed. Because tokenizers segment at different granularities, per-token perplexities from different tokenizers are not directly comparable.

### Method four: move the selected text toward the target domain

When the goal is technical Q&A, one can prepare material representative of that domain and increase the selection odds of similar content from the large pool. Importance resampling expresses the target weight as

$$
w(x)\propto\frac{p_{\mathrm{target}}(x)}{p_{\mathrm{pool}}(x)}.
$$

The intuition is that a kind of text that is common in the target but rare in the pool deserves a higher relative weight. DSIR estimates such weights in a lower-dimensional feature space so that large corpora remain tractable. It answers "closer to what content", not per-item fact verification: if the candidate pool simply does not contain what is needed, no amount of resampling creates it, and a few very high-weight samples can cause the same content to be selected repeatedly. The target reference material also cannot be the final test questions, because tuning the training material toward the answers makes a later evaluation on those same questions non-independent.

## What a higher threshold really changes

Take a self-posed pool of one thousand textbook-style documents and one thousand forum threads. An educational score threshold that keeps 80 percent of the textbooks and 20 percent of the forum leaves eight hundred and two hundred documents. The textbook share moves from one half to four fifths, so what looks like a quality adjustment also changes the genre mix. In distribution form, with original distribution $p(x)$, score $q(x)$, and threshold $\tau$, the retained distribution is

$$
p_\tau(x)=
\frac{p(x)\mathbf{1}[q(x)\ge\tau]}
{\mathbb{E}_{x\sim p}\big[\mathbf{1}[q(x)\ge\tau]\big]}.
$$

The indicator is 1 when the threshold is met and 0 otherwise, and the denominator is the total retained probability. If textbooks are longer on average, their share of tokens can exceed four fifths, which is why quality filtering and data mixture are not two independent steps.

Several scores also cannot simply be added. ChineseWebText 2.0 organizes quality, domain, and toxicity information separately, which lets a user filter by different questions; but a probability, a 0-to-5 score, and a domain label do not share a unit. Two scorers may simply reward the same style, and a productive next step is to examine the samples where they disagree: does the disagreement come from content, or from length, format, and source? Keeping the signals separate also means a later task can be served by a different selection, because a corpus scored for educational value is still a corpus that has had a particular distribution imposed on it.

## Verifying that a scorer is doing something useful

The source asks for at least three layers of checking. First, whether the standard was learned: humans judge a batch against the rubric, and precision measures how many selected documents meet the standard while recall measures how many qualifying documents were recovered. If 100 documents are selected and humans judge 80 of them compliant, and 160 compliant documents exist in the review scope, precision is $80/100=80\%$ and recall is $80/160=50\%$; these answer different questions, and if the human sample was deliberately biased toward hard cases, neither number describes the pool as a whole. Second, what the data became: distributions of domain, genre, length, source, and duplication, plus a manual look at rejected content, since a filter that removed advertising and one that removed debugging experience look identical in the retained set. Third, whether the model benefits: fix model, tokenizer, training tokens, optimization settings, and evaluation, then compare random sampling against score-based selection, holding domain ratios fixed when studying the scorer itself and reporting domain shifts when studying the whole filtering scheme.

A useful experiment begins with a prediction: educational filtering may help explanatory tasks without equally helping short API lookups, and testing those two capabilities separately is more informative than a single aggregate score.

Rewriting unsuitable text is a further option with its own errors. Nemotron-CC explores classifier combinations and synthetic augmentation. A rewrite may supply missing expression, and it may equally drop a limitation condition or introduce a false fact; generating ten versions of a source does not yield ten new pieces of knowledge. The record should therefore keep the original ID, the generation model, the prompt, and the sampling configuration, and rewrites of one source belong in the same split so that an original cannot end up in training while its rewrite sits in test. Generation and review costs belong in the comparison.

The source states its own conclusion boundaries explicitly, and they are the boundary of this overview as well:

| Conclusion | How far it is supported |
| --- | --- |
| A reference classifier leans toward the reference distribution that was chosen | The idealized Bayesian relation gives the direction; real scores also carry estimation error |
| Perplexity expresses how hard a reference model finds the text | It is not a factual-accuracy rate and not a general quality measure |
| A threshold changes the domain and genre proportions | Follows directly from the retention ratio |
| A given scorer improves a locally trained model | Has to be answered by controlled training and task-specific evaluation |

## Where the Chinese original lives

The Chinese original is [What Counts as High-Quality Pretraining Data: From Rules to Model Scores](/posts/2026-10-15-llm-training-data-03-quality/). It contains the full rubric table, the derivations with their stated assumptions, the precision and recall arithmetic, the per-source table of what a conclusion does and does not support, and the reference list. Its central limit is unchanged here: a score ranks text against a rubric and a reference distribution, and only a controlled training comparison with a fixed evaluation can say whether that ranking helps a model.
