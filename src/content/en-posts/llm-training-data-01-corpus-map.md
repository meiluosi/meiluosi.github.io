---
sourceId: 2026-10-15-llm-training-data-01-map
slug: llm-training-data-01-corpus-map
title: "Choosing a Pretraining Corpus: A Map of Chinese and English Open Datasets"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-15T00:00:00.000Z
description: "A selection framework for Chinese and English pretraining corpora, separating file size, token count, release history, and shared upstreams."
tags:
  - Pretraining Data
  - Corpus Selection
  - Dataset Survey
category: LLM Training Research
series: LLM Training Research Notes · Data Volume
seriesOrder: 1
lang: en
---

> **Scope note.** This is an English overview of a Chinese corpus-selection guide. The dataset sizes it reports come from publisher data cards and papers as checked on 2026-10-15; they are a dated snapshot rather than current totals, and the projects use different units. Its worked scenario — a Chinese-first technical model — is a teaching premise, not a project that was run. No corpus was downloaded, built, or trained, and the article does not establish that any named dataset is the best choice.

## Four layers between a download and the tokens a model reads

The source separates four things that are easy to conflate:

- **Source** — where the text originates: web pages, encyclopedias, code repositories, or documents the project is licensed to use.
- **Corpus** — what survives extraction: the readable text pool with markup removed and obvious junk dropped.
- **Recipe** — how much of each kind of text is used: language shares, domain shares, and whether a small source is repeated.
- **Training input** — what the run actually read, after tokenization, shuffling, chunking, and packing. Tail blocks may be discarded and some content may be read more than once, so realized proportions need not match the configuration.

The chain runs from source to extraction, cleaning, and deduplication; then to an optional text pool; then to per-source training shares; then to tokenization and packing; and finally to what the model actually reads. The last step is the one that is usually invisible. A source may hold 100 GB of Chinese text while one run consumes only 20 GB of it, three times over. The questions "what is available" and "how will it be used" therefore have to be answered together, not in sequence.

The six articles of this volume each close one of those decisions, and the choice made in one changes the inputs of the next:

| Article | Decision it supports |
| --- | --- |
| 1, this one | Which sources to look at, and which small piece to try first |
| 2 | Which parts of a web page are kept and which are removed |
| 3 | Which text to train on first when the budget is limited |
| 4 | How to reduce repetition and to avoid meeting the test answers early |
| 5 | How much Chinese, English, and each domain to train on |
| 6 | How to turn those decisions into a data version that can be rebuilt |

## Six construction routes and what each one fixes

Most open corpora fall into a small number of construction routes. The route tells you what problem the project set out to solve, not how well it solved it, and a single project may span several routes.

| Construction route | Problem it targets | Representative projects |
| --- | --- | --- |
| Clean a large web crawl | Web text is plentiful but noisy | C4, RefinedWeb, FineWeb |
| Mix several sources | One crawl does not cover enough | The Pile, Dolma, SlimPajama |
| Retain scores and re-filter | Even readable text has a more useful subset | RedPajama-V2, DCLM, FineWeb-Edu |
| Handle Chinese specifically | English filtering habits do not transfer | SkyPile, ChineseWebText, CCI3.0-HQ |
| Cover many languages | Different languages need different parameters | mC4, OSCAR, FineWeb2 |
| Add generated text | Attempt to supplement or rewrite existing content | Nemotron-CC |

The English releases quoted in the source can be read against those routes. Common Crawl is the crawl itself and ships WARC (raw fetch), WAT (metadata), and WET (extracted text); taking WET saves a parsing step but is not the same as a cleaned, deduplicated training set. C4 remains the classic case study for rule-based cleaning, and its English, unfiltered, and multilingual mC4 configurations must not be treated as one dataset. RefinedWeb's public extraction reports roughly 968M web pages and 500B to 650B tokens, with the token figure varying by tokenizer; the phrase "public extraction" matters, because it is not the full pool described in the paper. FineWeb launched at about 15T tokens, and the data card checked by the source records more than 18.5T; for anyone comparing against an older paper's experiment, the scale of the version that experiment used is the one to use, not the later number.

Multi-source releases are described the same way. The Pile used roughly 800 GB across 22 components, which is a good study of source selection and weighting, though the historical component list does not guarantee that every part is still downloadable from its original address. Dolma describes about 3T tokens of English data spanning web pages, scientific literature, code, public-domain books, social media, and encyclopedias, published together with the processing tools — which is what makes it useful for the question "could someone else rebuild this processing step". RedPajama-V2 exposes quality signals rather than only final text, and its official table estimates 30.4T tokens for the annotated and deduplicated `head_middle` slice across five European languages; that is one slice, not a project total. SlimPajama is cleaned and cross-source deduplicated from roughly 1.21T tokens of first-generation RedPajama material, published at 627B tokens; the similar name does not make it a shrunken RedPajama-V2.

DCLM is not only a dataset name but an experimental framework for comparing data-processing methods, so it has to be recorded carefully: raw pool, baseline corpus, or evaluation framework. The DCLM repository recorded a correction in September 2025 to the baseline values behind the CORE and EXTENDED aggregate scores, so older and newer aggregate numbers cannot be subtracted directly. FineWeb-Edu filters FineWeb for educational value and reported about 1.3T tokens at release; it is derived from FineWeb, so adding the two sizes does not add two independent corpora. Nemotron-CC reported about 6.3T tokens, of which roughly 1.9T was synthetic, and its v2 recipe is a separate release — generated expression is not automatically new knowledge. FinePDFs points at PDF data, where reading order, tables, and OCR errors make it a different input type from web text.

## Chinese corpora: units, overlap, and near-identical names

Chinese technical articles routinely mix English terms and code, so a rule demanding that most characters be Han will delete exactly the CUDA tutorials worth keeping, and Chinese has no word spaces, so English "too few words" and "unusual average word length" heuristics do not transfer unchanged. Beyond language adaptation, overlap between sources has to be checked.

| Dataset | Release scope as checked | What to inspect before choosing |
| --- | --- | --- |
| SkyPile-150B | Chinese web, about 150B tokens, 620 GB of plain text | Whether the source fields you need are preserved |
| WanJuan 1.0 | Multilingual release containing Chinese and English text from web, encyclopedias, books | Whether the download is the text subset or another modality |
| ChineseWebText 2.0 | Project reports about 3.8 TB, with quality, domain, and toxicity signals | Whether it can be re-filtered, and which upstreams it inherits |
| CCI3.0-HQ | 500 GB subset selected from CCI3.0 | How educational value is labeled and what the threshold removed |
| OpenCSG chinese-fineweb-edu-v2 | Data card reports about 188M records and 420B tokens | Scoring criteria, tokenization accounting, version differences |
| IndustryCorpus2 | A series of industry-organized corpora | Whether the industry labels match the continued-pretraining task |
| Matrix | Data card reports about 4.69T tokens, Chinese and English mixed | What the full pool contains, including Q&A or exam material |

These figures cannot be ranked against each other directly. GB and TB describe storage on disk, tokens describe how many pieces a particular tokenizer produced, and record counts describe how many entries exist. A thousand records can be a thousand short sentences or a thousand long books.

Two source-level traps are worth stating directly. First, the ChineseWebText 2.0 source description includes upstream projects such as SkyPile, WanJuan, and WuDao, so concatenating all of them does not give several independent corpora; "may overlap" has to be turned into a measured overlap on the versions actually used. Second, similar names do not mean similar training use: OpenCSG's V2 and Hugging Face's FineWeb2 are different projects, and OpenCSG's Fineweb-Edu-Chinese-V3 is positioned as educational SFT data rather than a drop-in replacement for V2 web text. For historical projects such as WuDao, "was published at some point" and "which files were actually obtained this time" are separate facts, and only the second one determines reproducibility.

Multilingual releases such as mC4, OSCAR, and FineWeb2 are valuable for comparing processing across languages: FineWeb2 offers configurations by language and script, which is useful, but a 0.8 score for Chinese text and a 0.8 score for English text do not mean the same reliability unless the scores are calibrated across languages. Download a specific language configuration and version, and do not report a multilingual pool's total as a Chinese figure.

## A selection procedure that yields a record rather than a ranking

The source answers selection with three questions rather than a leaderboard.

1. **Which step do you intend to change?** If the research question is body-text extraction, you need original pages or locatable crawl records. A file with only a `text` field is fine for studying filtering, but it cannot recover tables that upstream already deleted.
2. **What information do you need to keep?** If the question is about quality thresholds, prefer material that carries scores, domain labels, or a scorer that can be re-run. Taking only the top-scoring subset hides what a threshold deleted.
3. **How much can you actually process?** A personal project can start with one Chinese slice and one English slice, learn the fields and provenance, and only then extend. The criterion is whether the processing can be observed and rebuilt, not a dataset ranking.

Each candidate gets a short record covering which copy was used, what it contains, what processing it has already received, what is still missing, and why it is being considered at all:

| Item | What to record |
| --- | --- |
| Which copy is used | Repository, immutable version, language configuration, crawl batch, file list |
| What is inside | Web body or Q&A, domain, language, natural or generated text |
| What was processed | Extraction, filtering, deduplication, and scoring methods |
| What is missing | Source fields, scorers, original pages, or build notes |
| Whether it can be used and re-fetched | Usage conditions in the data statement, upstream notes, actual acquisition route |
| Why it is included | Which question of this project it helps answer |

Code licenses and data usage terms can attach to different objects, so the specific conditions and their versions get recorded rather than summarized as "open source".

Verification starts small: take a sample from candidates A and B spanning different sources and lengths, check body completeness, Chinese technical content, mixed Chinese and English layout, and duplication with one common rubric, and state a prediction in advance — for instance, which candidate should retain more complete code blocks. That round supports "this material suits further processing". Claiming the data is better for a model requires a further comparison with the model, tokenizer, effective training tokens, optimization settings, and evaluation all fixed, and two models on two leaderboards cannot decide it.

The source closes with a table of what may be claimed on this evidence and what may not, and the distinction is worth keeping:

| Claim available now | Basis or boundary |
| --- | --- |
| These projects follow different construction routes | Publisher data cards and papers, not a market ranking |
| Some corpora share upstreams | Supported by the source descriptions; the actual overlap still has to be measured |
| More files do not necessarily mean more independent content | Duplication and derivation cause this directly |
| A candidate suits this Chinese technical model | Requires local sample review and same-condition training; no conclusion is available yet |

## Where the Chinese original lives

The Chinese original is [Where Pretraining Data Comes From: A Map of Chinese and English Open Corpora](/posts/2026-10-15-llm-training-data-01-map/). It holds the per-dataset tables with every linked data card, the selection record template, the six-articled plan for the rest of the volume, and the reference list. Its sizes are the ones checked on 2026-10-15, and the selection question it leaves open is the one it answers next: which body text to keep from a web page.
