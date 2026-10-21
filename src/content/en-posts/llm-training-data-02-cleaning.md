---
sourceId: 2026-10-15-llm-training-data-02-cleaning
slug: llm-training-data-02-cleaning
title: "From Web Page to Training Text: What Corpus Cleaning Actually Does"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-15T00:00:00.000Z
description: "How pretraining corpus cleaning separates extraction, normalization, filtering, and fragment handling, and where each step deletes useful text."
tags:
  - Pretraining Data
  - Data Cleaning
  - Web Extraction
category: LLM Training Research
series: LLM Training Research Notes · Data Volume
seriesOrder: 2
lang: en
---

> **Scope note.** This is an English overview of a Chinese cleaning walkthrough. The example page, the retention arithmetic, and the duplicate-ratio figures are self-written teaching examples constructed to expose failure modes; they are not measurements on a real corpus. The conclusions it attributes to FineWeb, FineWeb2, and C4 come from those projects' own published settings. No crawl was cleaned, no pipeline was executed, and the article does not establish that any particular extractor or rule set is correct for a given corpus.

## Four operations that are usually collapsed into one

Cleaning a page is not one decision but four. The source keeps them apart because each one fails in a different direction.

| Operation | In plain terms | What it must get right on a technical page |
| --- | --- | --- |
| Body extraction | Locate the main content | Separate the article from navigation and advertising |
| Text normalization | Unify formatting that does not change meaning | Unify line endings and encodings while keeping code indentation |
| Filtering | Decide whether to keep it | Judge whether what remains is complete and useful |
| Fragment handling | Deal with local special content | Apply an explicit policy to contact details instead of dropping the page |

The worked example is a self-written page containing a navigation bar, an article with an English technical sentence, a code block, an aside with a coupon prompt, and a footer with an address. Stripping the tags leaves the navigation and the advertisement in place. Collapsing all whitespace to single spaces destroys the code's nesting. Requiring most characters on each line to be Han may delete the second technical paragraph. None of these are "insufficient cleaning"; they are different operations interfering with each other.

The practical consequence is bookkeeping: if every step records its input and its output, the point where the body text broke is identifiable. A single monolithic `clean(text)` that returns an empty string leaves no way to attribute the loss to a rule.

### Input choice decides what can still be repaired

Common Crawl ships WARC (raw fetch), WET (extracted text), and WAT (metadata). Starting from WET removes the HTML-parsing step; starting from WARC allows the extractor to be replaced. The difference is how far downstream the damage can be undone: if upstream has already flattened a table into "12 24 36", the column names and the row-column relation are gone, and no later filter can recover what those numbers meant.

Working from cleaned JSONL or Parquet is a legitimate starting point, but the research question has to be restated accordingly: you are comparing how existing bodies are filtered, not how pages are extracted. The source's recommendation for a first version is to begin from processed text to reduce engineering cost, and to return to original pages only if sampling shows that code blocks are missing in bulk. Not every project needs to redo the crawl.

### Extraction has to preserve structure, and the checks are type-specific

Extractors guess the main content from HTML node structure, text density, and link ratios. Those cues suit news pages and can be wrong for forums, API documentation, and table-heavy pages. FineWeb compared WET text against text extracted from WARC with trafilatura and reported better downstream performance for the latter in its own training setup; that supports treating extraction as a separately comparable step, not that one tool suits every page.

The source recommends fixing a small set of pages and comparing source and output by hand, and it maps page types to what to check first:

| Page type | Check first |
| --- | --- |
| Technical blog | Whether title, body, and code survive intact |
| API documentation | Whether parameter names still map to their explanations |
| Table explanation | Whether headers and values keep their relation |
| Forum thread | Whether question, answers, and quotes did not merge into one paragraph |
| News or general article | Whether navigation and recommendations leaked into the body |

Fluency is not a sufficient check. An article that lost a whole "does not apply when..." caveat can still read smoothly. Extracting more is not automatically better either: keeping all comments adds text and can splice different people's positions into a single argument, so comment retention is an explicit decision rather than a tool default.

## One text for training, another for matching

Deduplication needs to recognize articles that differ superficially, which pushes toward lowercasing, stripping digits, or converting between traditional and simplified characters. Those same transforms can destroy meaning for training data: `API` and `api` may be different variables, `0.001` and `0.01` become indistinguishable without digits, and traditional-to-simplified conversion can change regional expression. Text that is convenient to match is not automatically suitable for learning.

The source's answer is to keep two fields with different rules: `training_text`, the body actually used for training, receiving only reviewed transformations; and `matching_text`, a dedicated copy used solely for duplicate matching under its own stated rules. A record of the original position, or a controlled raw cache, allows lookups later. Unicode normalization must be stated precisely as well — NFC and NFKC are not equivalent, since NFKC folds more compatibility characters, and calling both "normalizing the encoding" hides the difference. Each rule set needs a version, otherwise a changed fingerprint in the next run cannot be attributed to a changed document rather than a changed cleaning step.

### Mixed Chinese and English is not defective Chinese

The target is Chinese technical content, which by nature contains CUDA, batch size, memory bandwidth, code, and formulas. The proportion of Han characters can be a signal, but it cannot replace a content judgment: a normal tutorial and a page of product part numbers can each be half Han while serving entirely different purposes. Language identification models also err, and a score of 0.9 does not mean "90 percent probability of being correct" unless that score was validated for the relevant language and domain; short texts and specialized terminology are harder to classify. FineWeb2 studies language identification and thresholds and configures handling by language and script, which suggests checking for language differences without supplying a universal threshold for all Chinese corpora. Traditional characters should be handled according to the coverage goal rather than downgraded for glyph differences alone.

## Every rule needs a unit, and every threshold needs an audit

"How much repetition is too much" becomes decidable only once the unit is fixed. In the source's example, a document has ten lines, five of which repeat the same navigation sentence: counting extra repeated lines gives

$$
r_{\mathrm{line}}=\frac{L-U}{L}=\frac{4}{10}=40\%.
$$

Counting characters instead gives a different answer for the same document. With five navigation lines of two characters each and five body lines of one hundred characters each, the repeated characters are $4\times2=8$ out of $5\times2+5\times100=510$, or about 1.6 percent. Neither computation is wrong; they measure different things, which is why a threshold is meaningless without its unit. Empty documents need their own handling rather than a denominator patched by adding one.

The same care applies to other signals, each of which can also destroy content that matters:

| Signal | What it may indicate | What it can wrongly remove |
| --- | --- | --- |
| Very short | An empty page or a failed fetch | A concise API note |
| Many links | Navigation or an aggregation page | Documentation rich in references |
| Many repeated fragments | Keyword stuffing or looping text | Code whose format repeats but whose parameters differ |
| Unusual punctuation | Corrupted text | Code, formulas, poetry |
| High non-Han ratio | Not the target language | A Chinese technical tutorial |

Word-count heuristics for English rely on spaces; splitting a Chinese paragraph on whitespace treats the whole paragraph as one word, so whether a statistic counts characters, tokens, or model tokens must be stated.

The second half of the audit asks what was removed, not only how much. Starting from one million records, extraction keeps 800,000, language filtering keeps 600,000, and rule filtering keeps 480,000. The last step retains 80 percent of the previous stage, and the pipeline retains 48 percent of the original. Those answer different questions, and a report that states only "80 percent retained" invites the reading that most of the original data survived. The breakdown that matters is by source, language, domain, and length: an overall 80 percent can hide a case where ordinary articles survive and technical documents are nearly all deleted, which for a Chinese technical project is a rule that needs re-examination. Document counts are also not a substitute for token counts, because removing many short advertisement pages barely changes training length.

Order matters for the same reason. An advertisement page may be both too short and too repetitive, and whichever rule runs first is the one that removes it. Counting only the first rejection reason cannot be used to attribute contributions to individual rules; recording all signals that fired, then the final decision, is the record that supports that analysis.

## Content filtering needs context, and comparisons need a prediction

Detection of email addresses, names, or account information can use patterns, named-entity recognition, and context rules. A pattern finds only part of the cases, and can mistake dates, version numbers, or product codes for sensitive content. After detection, the policy still has to be chosen: redact the fragment, quarantine the whole document, or exclude the whole source. A local detection result is not a document-level quality verdict, and raw text kept for traceability need not be published in full; controlled storage is enough. Keyword filtering has the same shape — an article discussing a social issue may contain a filtered term, and discussing is not endorsing. The C4 analysis by Dodge and colleagues, published at EMNLP 2021, found that filtering affected group-related text unevenly, which is why coverage of the corpus has to be checked after cleaning and not just its tidiness.

Comparing one cleaning change against another starts with a concrete prediction, such as "the new method should leave less navigation while not increasing missing code blocks", which is far easier to falsify than "the new method should be cleaner". The same pages run through both methods produce four categories of samples: kept by both, rejected by both, near the threshold, and decided differently — the last being the one that explains the change. Manual review is best done blind to which method produced the output, recording body completeness, code retention, and template residue separately. Only when the data layer is clear is it worth fixing the model, tokenizer, training tokens, and evaluation for a training comparison; text that reads tidier is not automatically better for a model.

The source's summary table states what each kind of comparison result can support:

| Comparison result | Conclusion it supports |
| --- | --- |
| Less navigation, code still intact | On the reviewed pages, the extraction change fits the goal better |
| A higher rejection rate | The new rule removes more; it does not show that it removes the right things |
| Better target-task results under the same training conditions | The change helps under those training and evaluation conditions |
| Only the technical domain improves while others decline | A trade-off has to be discussed, not collapsed into one total |

A pipeline of this kind also needs a defined policy for reads that fail. DataTrove and its FineWeb example show how read, filter, and write tasks are separated, and the source's instruction for a small project is to write down the result of every step, the reason for each rejection, and the error records before scaling up. A read failure belongs in the error record; letting it pass as "low quality, removed" destroys the evidence that the pipeline is broken rather than selective.

## Where the Chinese original lives

The Chinese original is [From Web Page to Training Text: What Corpus Cleaning Actually Does](/posts/2026-10-15-llm-training-data-02-cleaning/). It contains the full teaching page, the per-page-type check table, the rule-signal table, the retention arithmetic, the comparison protocol, and the reference list, including the DataTrove and FineWeb examples for splitting read, filter, and write tasks. Its examples explain why rules misfire; whether a new rule helps on a given corpus still has to be decided by the comparison procedure it describes.
