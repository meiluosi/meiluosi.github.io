---
sourceId: 2026-10-20-llm-training-pretrain-01-tokenization
slug: llm-training-pretrain-01-tokenization
title: "How Text Becomes Tokens: Byte-Pair Encoding, Vocabulary, and Cost"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of byte-pair encoding, vocabulary construction, normalization, special tokens, and how tokenization sets sequence length and cost."
tags:
  - Tokenization
  - Byte-Pair Encoding
  - Pretraining
category: LLM Training Research
series: "LLM Training Research Notes · Pretraining Volume"
seriesOrder: 1
lang: en
---

> **Scope note.** This is an English overview of a Chinese technical note in the pretraining volume. The byte-pair encoding example is a hand-computed teaching calculation over a three-string toy corpus with assumed frequencies; no tokenizer was trained, no vocabulary file was produced, and no model was trained. The tokenizer audit presented later is a proposed experiment design, not a completed audit. The source's three toy strings are Chinese characters, so this overview relabels the same strings with Latin symbols and preserves every count and merge operation.

## Why subword units, and a byte-pair encoding example computed by hand

Numbering every character separately is possible. For a small closed vocabulary of common characters it is easy to inspect. The difficulty sits in the word "only": technical corpora contain English identifiers, digits, spaces, code, emoji, and rare characters. A character-only table drops content, while a table of every common whole word fails on unseen words and spelling variants.

Length pushes the same way. Splitting `optimizer` into its nine letters costs the model nine positions, whereas a vocabulary entry for the whole string may cost one. But a dedicated entry for every rare long string inflates the vocabulary, and those entries receive few training updates.

Subword methods take a middle position: common fragments stay long, uncommon content is split more finely. Sennrich et al. (2016) introduced byte-pair encoding to neural machine translation to address rare words and an open vocabulary. That is the motivation and the result of that translation study; it is not evidence that byte-pair encoding is the best choice for any large model.

The source's toy corpus holds three strings, each built from single-character units, with assumed occurrence counts. Relabeling the same structure with Latin symbols:

| String | Occurrences | Initial split |
| --- | ---: | --- |
| `A B C` | 5 | `A / B / C` |
| `A B D` | 3 | `A / B / D` |
| `C E` | 2 | `C / E` |

Counting adjacent pairs, `A + B` occurs 8 times, `B + C` 5 times, `B + D` 3 times, and `C + E` 2 times. The first merge operation therefore joins `A + B` into the single unit `AB`.

After that merge the three strings read `AB C`, `AB D`, and `C E`. Recounting, `AB + C` now occurs 5 times and becomes the next merge candidate. If training stopped there, the vocabulary would hold the five initial characters plus the two learned units `AB` and `ABC` — seven entries in total.

The total length of the raw corpus is:

$$
5\times3+3\times3+2\times2=28
$$

The first merge removes 8 positions, leaving 20; the second removes 5 more, leaving 15. These counts describe the token count of this teaching corpus only, with no special markers added.

The shorter sequence is not free: it was bought with two extra vocabulary entries. Training a real tokenizer also has to fix the base alphabet, boundary rules, vocabulary size, and how to break ties when two pairs share a frequency.

For a new string, the learned rules apply in their existing priority order. The source's example yields a merged segment plus a leftover unit that need not match the segmentation a reader would choose; the tokenizer learns statistical rules, not semantic completeness. Training a tokenizer learns merge rules, while using one follows existing rules. Re-deriving a fresh set of merges for every new text would destroy the stability of the numbering system. The Hugging Face BPE tutorial is the source's reference for this distinction.

## The steps that surround the algorithm

A real pipeline usually involves normalization, pre-splitting, and special-token handling around the subword step. Saying only "it uses byte-pair encoding" does not reproduce a single segmentation. The Hugging Face Tokenizers pipeline documentation is the source's reference for these stages.

| Stage | Question it answers | Example to inspect |
| --- | --- | --- |
| Normalization | Is the input rewritten first? | Case, full-width characters, combining marks |
| Pre-splitting | Which boundaries may not be merged? | Spaces, punctuation, digits, newlines |
| Subword splitting | How is a span split within the allowed range? | Long identifiers, mixed Chinese and English |
| Special-token handling | Are extra markers added by convention? | Start, end, role boundaries |

Lowercasing every English string may be harmless in natural language, but `UserID` and `userid` are not equivalent in code. Later stages can restore the normalized text exactly, yet they cannot restore case that an earlier step erased. "Decodable" therefore needs a question attached: restored to the original input, or to the normalized input? Are spaces and newlines preserved? Are special tokens skipped on decode?

SentencePiece (Kudo and Richardson, 2018) is a tokenization tool, not a synonym for one splitting algorithm. Its paper emphasizes direct handling of raw sentences and language-independent processing, and it supports several subword models. Any experiment should therefore name the tool, the algorithm, and the training configuration together.

## Vocabulary size, sequence length, and corpus mix

A vocabulary is more than a text-to-integer table. A model usually keeps one input vector per token and computes output scores over every vocabulary candidate. With a vocabulary of $V$ entries and hidden dimension $d$, a plain embedding matrix has $Vd$ parameters. For the teaching configuration $d=512$:

| Vocabulary size | Parameters in one embedding matrix |
| ---: | ---: |
| 16,000 | 8,192,000 |
| 32,000 | 16,384,000 |

These are matrix-shape calculations, not a model's measured footprint. If input embeddings and output weights are tied, that part of the weights is shared; if not, the output matrix is counted separately. Gradients, optimizer state, activations, and implementation details continue to affect training memory.

A larger vocabulary can shorten sequences and enlarge the relevant matrices at the same time. For small models the vocabulary share of total parameters deserves attention. "The same text is now two tenths shorter in tokens" does not establish that training cost drops by two tenths.

Corpus mixing ratios are also defined through the tokenizer. Assume a fixed set of source text: under tokenizer A the Chinese part yields eight hundred thousand tokens and the English part four hundred thousand, so Chinese accounts for two thirds of the tokens. Suppose tokenizer B cuts the same Chinese into four hundred thousand tokens and leaves English at four hundred thousand; no source file changed, yet Chinese falls to one half of the tokens. These are assumed numbers that illustrate a change of unit. Under a token-budget sampling rule, every source's actual share of text must be recomputed after a tokenizer swap, and old percentage tables do not carry over. "Train each model on one hundred million tokens" and "train each on the same complete documents" answer different questions — the first fixes the number of prediction positions, the second fixes the raw content — and they generally cannot both be fixed. Reports should state which budget was used and add the other measure's actual consumption.

## Identifier compatibility is where post-training breaks

Starting supervised fine-tuning from an existing pretrained model means the tokenizer should match that model. Suppose the model learned that id 100 denotes a particular fragment; its embedding vector and its output parameters are already organized around that meaning. Swapping in a vocabulary where id 100 points elsewhere leaves the integers meaningless, even if the vocabulary size is identical.

Special tokens behave the same way. An end marker can be a target the model learns to predict, but it is not a barrier that automatically stops attention. Whether a role marker works depends on the template, the numbering, and the usage the model actually learned.

At minimum, a migration check should confirm that the tokenizer file matches the model version and that the id mapping has not changed; that the template does not add a start or end marker that the encoder then adds a second time; that the relevant matrices were extended when new tokens were added and that the new entries received training; and that training and inference share the same role and end conventions. These are interface-consistency problems, and "the program reported no out-of-vocabulary id" is far from sufficient.

The source then proposes a first experiment: a small tokenizer audit over validation text split by category — Chinese prose, English prose, code, mixed Chinese and English, rare characters — recording, for each candidate, the original text, the token ids, the readable pieces, the total length, and the full decoded text. If a tokenizer is trained locally, the vocabulary must be learned on the training portion only, with validation text reserved for checking. The audit answers narrow questions, and each one comes with a limit on what may be read out of it. Whether content is corrupted is checked by comparing original and decoded text under the allowed normalization; a representation that round-trips does not imply the model understands the text. Which categories become long is reported per category rather than on average; a shorter average does not mean every category got shorter. Whether truncation is likely is measured under a fixed context window; not truncating does not mean the model learns better. Whether boundaries stay consistent is checked across whitespace, end markers, and role templates; consistent formatting does not mean the training objective is right. This audit is a proposal in the source, not a result.

## What stays in the Chinese original

The Chinese note carries the full hand-computed merge tables, the byte-coverage discussion, the per-category audit record format, the two review questions, and the reference list. This overview keeps the arithmetic and the decision rules; it adds no tokenizer ranking, no measured segmentation length, and no training result the source does not contain.

Main limit, restated: the byte-pair encoding example is a hand calculation over assumed frequencies, and nothing in the source was trained, benchmarked, or measured.
