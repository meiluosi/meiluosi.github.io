---
sourceId: 2026-10-15-llm-training-data-04-dedup
slug: llm-training-data-04-dedup-and-contamination
title: "Deduplication and Evaluation Contamination: What the Model Actually Learned"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-15T00:00:00.000Z
description: "How fingerprinting, shingles, MinHash, and LSH find near-duplicates, group them, and keep evaluation questions out of the training corpus."
tags:
  - Pretraining Data
  - Deduplication
  - Contamination
category: LLM Training Research
series: LLM Training Research Notes · Data Volume
seriesOrder: 4
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on deduplication and contamination. Its shingle example, its LSH probability table, and its grouping sketches are teaching computations on made-up inputs, not measurements on a real corpus. The deduplication findings attributed to FineWeb and to Lee and colleagues were reported by those sources under their own experimental settings. Nothing was deduplicated locally and no benchmark was run, so the article does not establish that a particular dedup strength is right for a given model.

## Decide which kind of repetition is being removed

Two documents that look similar are not automatically redundant. Two authors explaining gradient descent separately may offer different examples and angles. The source separates cases before choosing a method:

| Case | Example | Where to start |
| --- | --- | --- |
| File duplication | The same shard downloaded twice | File checksums |
| Identical documents | Two URLs with byte-identical bodies | Body fingerprints |
| Small edits | Same body, different date and footer | Text-fragment similarity |
| Local copying | A whole tutorial section copied into a long article | Paragraph or containment checks |
| Similar topic | Independent explanations of one concept | Judge the use case, do not delete |

This article concentrates on copying and near-duplication: text that has been edited while large stretches stay identical.

## Exact fingerprints, and the rules that must not be over-eager

Hashing the matching text with a stable algorithm produces a fixed-length digest, so identical texts can be grouped quickly without comparing every body character by character. The matching text is the dedicated copy introduced for this purpose, with line endings unified and clearly meaningless differences settled first. A long enough cryptographic digest has a low collision probability, and for strict confirmation the source still compares the original content of documents that share a digest.

The normalization rules cannot be aggressive. Two configurations — "set the learning rate to 0.001 and evaluate every 100 steps" against "set the learning rate to 0.01 and evaluate every 1000 steps" — become identical if digits are removed to ignore small differences, and dropping case or indentation can break the meaning of code. Exact deduplication therefore has a clear boundary: it finds text that is identical under the stated rules quickly, and it does not find small rewrites. After removing copies, the mapping from each copy to its representative should be kept, so other sources are not forgotten along with the duplicate.

### Shingles, Jaccard, and the case Jaccard misses

A long document that differs by one line has a different whole-document fingerprint. Splitting the text into many consecutive small fragments and counting shared fragments identifies such cases. A fragment of this kind is a **shingle**. The source illustrates it with two short strings and two-character shingles, using Chinese text; written here in Latin script so the example survives translation, A = `abcdef` yields {ab, bc, cd, de, ef} and B = `abcdxy` yields {ab, bc, cd, dx, xy}. Three shingles are shared and seven distinct shingles appear in total, so the Jaccard similarity — shared kinds over all kinds — is

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{3}{7}\approx0.429.
$$

That short string exists to explain the algorithm; it is not a recommendation to use 0.429 or character bigrams as a threshold on real corpora. The background for this method is the near-duplicates and shingling chapter of an information retrieval textbook. The shingle unit can be characters, words, or model tokens. Shorter shingles make ordinary phrases collide more often; longer ones let a small local edit break shingle agreement. Text shorter than the shingle length needs separate handling rather than being folded into one group of empty sets.

The important miss is containment. If a short document appears in full inside a much longer one, all of its shingles are copied, but the longer document contributes many additional shingles, the union grows, and whole-document Jaccard stays low. If the question is whether the short document is contained, the right measure is the fraction of its shingles covered, or a direct containment check; whole-document Jaccard answers overall overlap and is not a substitute for containment relations.

### MinHash compresses the comparison, LSH decides whom to compare

Even with shingle sets available, comparing every pair of a million documents is expensive: with $N$ documents there are about $N(N-1)/2$ pairs, so doubling the corpus roughly quadruples the work. MinHash replaces full sets with a short signature that estimates Jaccard. Imagine all possible shingles placed in a random order, where each document remembers only the first shingle that belongs to it. Two documents collide exactly when the earliest shingle among their combined set lies in the shared part, and the shared part's share of all shingles is the Jaccard similarity, so under ideal random ordering

$$
P[h_{\min}(A)=h_{\min}(B)]=J(A,B).
$$

One ordering is too unstable, so many independent permutations are used and the first shingle under each is recorded. With $m$ signature positions, the estimate is the fraction of positions that agree:

$$
\widehat J(A,B)=\frac{1}{m}\sum_{j=1}^{m}
\mathbf{1}[h_j(A)=h_j(B)].
$$

With 100 positions agreeing in 80 cases, the estimate is 0.8 — an estimate, not a conclusion. Positions must be compared position by position; pooling the signature values and counting equal ones is a different procedure. Under ideal independence the estimate has variance $J(1-J)/m$, so longer signatures fluctuate less at the cost of computation, storage, and transfer. Real implementations approximate the random permutation with a hash family, which brings hash design and collisions into the picture, and every document must use the same shingle definition, normalization rules, and seed.

MinHash compresses each document but does not reduce the number of pairs, so LSH — locality-sensitive hashing — decides which pairs are worth comparing. The signature is split into $b$ bands of $r$ rows, and any single band that matches exactly puts two documents on a candidate list. With $b$ bands, $r$ rows each, and true similarity $s$, under ideal per-row independence a row matches with probability $s$, a whole band matches with probability $s^r$, and all bands differ with probability $(1-s^r)^b$, so the probability of becoming a candidate is

$$
P(\text{candidate}\mid s)=1-(1-s^r)^b.
$$

For $b=14$ and $r=8$, the resulting teaching values are:

| True similarity | Probability of becoming a candidate, about |
| --- | ---: |
| 0.50 | 5.3% |
| 0.75 | 77.2% |
| 0.90 | 99.96% |

Candidate generation is a probability curve. Similarity of 0.75 does not guarantee discovery, and lower similarity does not forbid candidacy. Candidates can then be scored with a more precise similarity before a duplicate is declared, and some pipelines instead connect documents on signature collisions directly, trading accuracy for speed. Recording "became a candidate" separately from "was removed" makes debugging much easier. Template pages can also produce very large candidate buckets, and truncating them for resource reasons must be recorded, because truncation creates misses.

## Groups, representatives, and how much to remove

Consider A similar to B and B similar to C, while A and C are no longer that similar. Grouping by connected components still places all three in one group, because they can be linked through a chain of similarity relations. This says they are connected, not that every pair exceeds a threshold, and a document that is updated regularly can build a large group through successive versions.

Keeping only one document per group can therefore discard meaningful differences between versions. Keeping the longest may keep the one with the most advertising; keeping the newest may keep the least complete. The representative rule has to be explicit, large groups and long chains have to be inspected, and a stable secondary ordering prevents execution order from changing the outcome. The source also separates two decisions that are easily merged: grouping documents together prevents them from straddling the training and test boundaries, while deciding how many of them to train on is a separate choice.

### Removing more is not automatically better

Deduplication reduces repetition and changes the proportions in which different content appears. If technical tutorials are widely reposted and ordinary short posts are not, deduplication shifts the balance between them. In FineWeb's deduplication comparison, global processing across crawl batches and processing within a batch did not stand in a simple "the more thorough the better" relation, and the retention order also influenced the outcome. Separately, Lee and colleagues observed in their own setting that deduplication reduced memorization of outputs and improved related training and evaluation problems. The two observations do not conflict. The useful questions are which duplicates were removed, which copy was kept, and what filled the freed training budget: if a good tutorial was duplicated and an advertisement page took its place, a quality drop cannot be attributed to deduplication alone.

## Splitting, contamination, and how to read the result

Training and test sets should be separated before chunking. Cutting long documents into small pieces and then assigning them at random can place adjacent paragraphs on both sides, so the model never saw the same sentence but did see most of the same article's context. The clearer order is to find originals, derived versions, and duplicate groups first, split by group, and only then chunk within each set. Rewrites of one source inherit that relationship. The grouping unit depends on the question: document groups for new articles of the same kind, sources for unseen websites, and time for future content — with the last two also requiring checks for cross-boundary reposting. Group sizes differ, so the realized document proportions need not be exactly 90/5/5, and splitting a group to hit a target ratio breaks the independence the grouping was meant to protect.

Contamination is a separate search with a separate policy. Test questions are short and may sit inside a long page together with answers, explanations, translations, or rewrites, so whole-document similarity misses them. The source suggests starting from exact matches between question text and long spans, then checking local coverage, answer explanations, and suspicious derivatives, and finally having a human separate ordinary knowledge discussion from substantive question leakage. For a frozen external benchmark, the evaluation is protected and overlapping candidates are excluded from training; the decision to place a question in training or test cannot be made at random. The rule also must not expand into "training may not contain knowledge the test touches": the model should know the relevant material, and what has to be avoided is having already practiced the exact question or a direct answer to it. A report should state the matching rules, the share of questions hit, the overlap types, and the number of items handled; "nothing detected" means this method found nothing, not that contamination is absent, and a clean subset may simply be an easier benchmark, so a score difference cannot be read as the benefit of contamination removal.

A comparable experiment compares three treatments on the same candidate pool with a fixed effective token budget: no extra deduplication, exact deduplication only, and exact plus near-duplicate deduplication. The prediction is made before training — which copies stronger near-duplicate handling should remove, and which updated versions it is most likely to damage — and inspecting those changes is more informative than reading the dedup rate. The training report records the representative rule, source proportions, replacement content, and group sizes, and when the outcome worsens the check runs in order: wrong representative, reduced coverage, or worse replacement data.

The source closes with a table of what these mechanisms do and do not establish, which is also the boundary of this overview:

| Claim | Basis and condition |
| --- | --- |
| MinHash can estimate Jaccard | A mathematical property under ideal random ordering |
| LSH cannot guarantee finding every document above a similarity level | The candidate probability formula; real implementations add further error |
| Two documents in one duplicate group need not be similar to each other | It follows from the definition of connected grouping |
| A given dedup strength suits a local model best | Requires a controlled training budget and a check of the accompanying distribution changes |

## Where the Chinese original lives

The Chinese original is [Deduplication and Evaluation Contamination: What the Model Actually Learned](/posts/2026-10-15-llm-training-data-04-dedup/). It contains the full derivations with their assumptions, the LSH probability table, the grouping sketches, the reporting requirements for contamination, and the reference list. Its boundary holds: MinHash estimates Jaccard and LSH produces candidates with a stated probability, and whether a given dedup strength helps a given model is a question that requires a controlled training budget and a check for the distribution changes that come with it.
