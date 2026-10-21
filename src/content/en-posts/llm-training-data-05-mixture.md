---
sourceId: 2026-10-15-llm-training-data-05-mixture
slug: llm-training-data-05-mixture
title: "Mixing Chinese and English: From Corpus Ratios to Actual Training Distribution"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-15T00:00:00.000Z
description: "Why document ratios, token ratios, repetition counts, and the tokens a run actually reads differ, and how to state a mixture that can be checked."
tags:
  - Pretraining Data
  - Data Mixture
  - Sampling
category: LLM Training Research
series: LLM Training Research Notes · Data Volume
seriesOrder: 5
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on corpus mixture. The document lengths, the inventory ratio, and the budget in its worked examples are assumptions stated for the derivation, not measurements of a real corpus, and the formulas hold under the assumptions written next to them. DoReMi is described as it was reported in its own paper. No mixture was trained here, and the article does not establish a best Chinese-to-English ratio for any model.

## Name the unit before naming the percentage

"Chinese is half the data" can mean four different things: half the file size, half the document count, half the probability of drawing a Chinese source, or half of the tokens that actually receive a loss. These have no reason to coincide, and compressed file size additionally depends on format and compression ratio, which makes it a poor proxy for training volume at all. The source's recommended target is the share of effective training tokens, where effective tokens are the prediction positions that enter the loss, excluding padding. Inventory and usage are then separate: inventory is how much text is available, usage is how much a run reads, and inventory can be partially read or read repeatedly.

## A document ratio is not a token ratio

Simplify to drawing half Chinese and half English, assume Chinese documents average one thousand tokens and English documents two thousand, and assume every document is read in full. Drawing one hundred documents gives about fifty of each: Chinese contributes about 50,000 tokens and English about 100,000, so Chinese is one third of the tokens. In general, with document sampling probability $\alpha_i$ and average length $\mu_i$ for source $i$, the token share is

$$
\beta_i=\frac{\alpha_i\mu_i}{\sum_j\alpha_j\mu_j}.
$$

The numerator is the chance of drawing that source times the length it contributes; the denominator sums the same quantity over all sources. The relation assumes complete reading, stable average length, and no truncation or extra masking, and the realized ratio only approaches it with enough samples. The inverse gives the document sampling probabilities needed for a desired token share:

$$
\alpha_i=\frac{\beta_i/\mu_i}{\sum_j\beta_j/\mu_j}.
$$

With the lengths above, equal token shares require Chinese documents to be drawn about two thirds of the time — roughly sixty-seven Chinese and thirty-three English documents — so that the two contribute comparable token counts. This is not a recommended ratio; it explains why sampling by document and allocating by token are different operations. And once filtering rules change the average length, the earlier conversion has to be re-estimated.

### Fixed-length blocks and how the loss is averaged

Taking blocks of equal length and equal valid position counts makes each block contribute a similar amount of training, which makes ratios easier to control. Fixed length does not remove the checks: what happens to a trailing partial block, whether short documents are concatenated, whether padding contributes to the loss, and whether one source's partial blocks are discarded often enough to skew the final ratio. Averaging the loss matters as well. Averaging within each sequence and then across sequences gives every sequence a similar weight; summing token losses and dividing by the total number of effective tokens gives every prediction position a similar weight. The two implementations are not equivalent.

The mixture objective can be written compactly:

$$
p_\alpha(x)=\sum_i\alpha_i p_i(x),\qquad
\mathcal L(\theta;\alpha)=\sum_i\alpha_i\mathbb E_{x\sim p_i}[\ell_\theta(x)].
$$

It says: choose a source with the given weight, then sample from that source. Using it requires stating whether $x$ is a document or a fixed block, and how the loss is averaged; a clean formula does not by itself describe a training recipe.

The tokenizer also determines how much budget each text consumes. A token is not a fixed-length character: the same sentence can produce a different token count under a different tokenizer, and one tokenizer can segment Chinese and English at different efficiencies. "Train on 100M tokens" therefore has to be recorded together with the tokenizer version, and it is worth recording tokens per character or per byte and the length distribution of documents on a fixed sample. Fewer tokens per text shortens sequences but does not by itself show better model quality; vocabulary size and training behavior also matter. A tokenizer trained from scratch must be fitted on the training split alone and then used for validation and test, while a pre-trained one requires its version and origin to be recorded — it cannot be claimed to have seen no evaluation-related material. For the same reason, per-token perplexities from different tokenizers should not be compared directly; normalizing per character or per byte gives another view but still requires consistent text and boundary handling.

### Three allocation schemes and their costs

To keep the length question separate, assume equal-length blocks with identical valid position counts, and let $n_i$ be the inventory of source $i$.

- **Allocate by inventory.** The source with the most text is drawn most often. Large sources are read less repeatedly, but can crowd out other domains.
- **Allocate equally across sources.** Every source gets the same opportunity, and a small source can be exhausted quickly and then cycled.
- **Smooth in between.** A power-law weighting interpolates:

$$
\alpha_i=\frac{n_i^\gamma}{\sum_jn_j^\gamma},\qquad 0\le\gamma\le1.
$$

With $\gamma=1$ the weights are the inventory proportions, and with $\gamma=0$ every non-empty source is weighted equally; intermediate values raise the relative share of small sources. For an inventory of 90 to 10 and $\gamma=0.5$, the weights become $\sqrt{90}:\sqrt{10}=3:1$, that is 75:25 — the small source rises from a tenth to a quarter of the draw, without any new independent content being added to it. Some implementations express the exponent as $1/T$ and call $T$ a temperature; recording the formula is clearer than recording only "temperature 2". No form of this weighting gives a universally best Chinese-to-English ratio without a task.

## Repetition, coverage, and the design of a fair comparison

Assume a total budget of 1B effective tokens, a target of 40 percent for Chinese technical content, and an inventory of only 100M tokens in that domain after deduplication. The target allocates 400M tokens to it, which is roughly four passes over the available material. In general the equivalent number of traversals is

$$
e_i\approx\frac{B\beta_i}{n_i},
$$

where $B$ is the total budget, $\beta_i$ the target token share, and $n_i$ the independent inventory of that source. Raising the quality threshold and deleting half the inventory, with the mixture and the budget unchanged, moves the equivalent traversal from four passes to eight. A stricter filter therefore changes text quality and repetition together, and both have to be checked when results move.

Drawing one pass' worth of material does not mean every item was seen. If each draw samples one block with replacement from $N$ blocks and $m$ draws are made, the expected number of distinct blocks seen is

$$
N\left[1-\left(1-\frac1N\right)^m\right].
$$

The derivation considers one block: the probability of missing it once is $1-1/N$, the probability of missing it $m$ times in a row is $(1-1/N)^m$, and subtracting from one and multiplying by the block count gives the expectation. When $m=N$ and $N$ is large, only about 63.2 percent of distinct blocks are covered, and the rest are drawn repeatedly. Reading in a shuffled order without replacement covers all blocks in a full epoch. "One pass" cannot describe both loaders.

### Keep language and domain effects apart

If one recipe adds Chinese encyclopedia text and another adds English code, and the second yields better code ability, that does not show English to be more useful, because the content type changed as well. The record should keep language, domain, genre, quality band, natural or generated, and length interval separate; not every dimension needs a complex sampler, but the log has to carry enough to tell them apart. To study language ratios, hold domain fixed; to study technical content share, hold language fixed. Mixed Chinese and English documents need a stated convention as well: if a document is labeled Chinese, then the "Chinese token count" is the tokens contributed by Chinese-labeled documents, not a per-token language detection.

Training feedback can inform the mixture, though it is expensive to search by hand. DoReMi first searches for domain weights with a smaller model and then applies those weights to a larger model, attending to excess loss relative to a reference model rather than to raw loss magnitude. The reference matters because a high loss in one domain may only mean that the domain is noisy, not that extra budget there is valuable. That suggests smaller projects observe per-domain loss rather than only the overall average, but the full method involves specific loss handling, weight updates, and smoothing, so "feed more where the loss is high" is not a reproduction. Weights found with a small model are not guaranteed to transfer, since model capacity, training duration, tokenizer, and domain definition can all change the outcome, and the cost of the proxy run has to be counted.

### Staged training needs a fair comparison

Training broadly first and then raising the target domain share changes not only the ratios but also when content appears. To study whether order helps, give both schedules the same cumulative tokens per domain, let one hold a fixed ratio and the other change the stage order, and fix the total learning-rate schedule and other conditions, so a difference is attributable to ordering. If the new schedule also adds a large extra training segment, the gain cannot be attributed to order alone. Continued pretraining additionally has to check that general capability did not regress, not just report progress on the target domain. Each stage records the data version, token budget, ratios, learning rate, and switch point.

## Check the recipe against what was actually read

A correct configuration does not imply a correct loader. The source asks for at least four separate records: tokens read from disk, tokens fed to the model, effective tokens that entered the loss, and how often content was repeated. Concatenating short documents reduces padding, but the end-of-sequence symbol is only a boundary marker and does not by itself prevent attention across a document boundary; whether an attention mask is used, whether positions reset, and how trailing blocks are handled must match the training program.

A small interpretable experiment starts from a few Chinese-to-English ratios, fixes the candidate pool, model, tokenizer, effective training tokens, and optimization settings, and reports Chinese, English, and technical results separately rather than one average. If a task is too hard for a small model and scores sit near chance, tiny differences cannot separate the recipes. Efficiency comparisons are clearer when reported both at equal token budget and at equal compute budget, since those answer different questions. Ratios are selected on a development set, and the test set is reserved until the recipe is fixed, so the test questions do not drive the search.

The source states its own conclusion boundaries, which are the boundary of this overview as well:

| Judgment | Basis and condition |
| --- | --- |
| A ratio over documents is not a ratio over tokens | Derived under complete reading and stable average length |
| Upsampling adds repeated exposure | The inventory is unchanged while the training share rises |
| Feedback from a small model can inform domain weights | DoReMi provides a specific method; transfer stays conditional |
| A given Chinese-to-English ratio suits the current model | Requires same-condition experiments; there is no universal fixed answer |

## Where the Chinese original lives

The Chinese original is [Mixing Chinese and English: From Corpus Ratios to Actual Training Distribution](/posts/2026-10-15-llm-training-data-05-mixture/). It contains the full derivations with their assumptions, the allocation and coverage tables, the staged-schedule protocol, the reporting requirements, and the reference list. Its limit is the one the last section states: a correct configuration is not a verified distribution, so the mixture has to be checked against the tokens the loader actually read, under conditions that are held fixed.
