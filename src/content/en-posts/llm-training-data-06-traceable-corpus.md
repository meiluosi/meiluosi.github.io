---
sourceId: 2026-10-15-llm-training-data-06-build
slug: llm-training-data-06-traceable-corpus
title: "Building a Traceable Small Pretraining Corpus"
editionLabel: ENGLISH OVERVIEW · METHODS NOTE
published: 2026-10-15T00:00:00.000Z
description: "How manifests, record lineage, stable group splits, token provenance, and acceptance checks make a small corpus explainable and rebuildable."
tags:
  - Pretraining Data
  - Data Engineering
  - Reproducibility
category: LLM Training Research
series: LLM Training Research Notes · Data Volume
seriesOrder: 6
lang: en
---

> **Scope note.** This is an English overview of a Chinese article that designs a small, traceable pretraining corpus. Its record format, Python helper, directory layout, bucket sizes, and per-stage counts are a design and teaching examples, not a report of a corpus that was built. No dataset was downloaded, no corpus was produced, and no training run was performed. The article therefore describes how a corpus could be made inspectable; it does not show that any filtering method improves a model.

## What the first version has to deliver

The motivating failure is a training run whose results changed between weeks while the data directory held a single `train.jsonl` with one `text` field per line. Without a record of the upstream files, the filter thresholds, which copy of a duplicated article was kept, and how the splits were drawn, the change cannot be attributed to the data rather than to the pipeline.

The first version does not start with terabytes. It targets a Chinese-majority corpus with some English, usable for a small model, and able to explain its difference from the previous version. Scale increases in three steps: a few dozen self-written samples to check fields, rejection records, and grouping logic; a fixed real slice to observe language, domain, duplication, and wrongful deletion; and only then a size chosen from the model and resource budget, followed by a training comparison. Passing the first step shows the pipeline runs as designed, not that the data is good; passing the second does not show the model will benefit. Keeping those layers apart is what makes the engineering progress honest.

The deliverable is therefore more than a text file. It includes the input manifest, the processing configuration, the split assignment, the statistics report, and the instructions for how training reads the data.

## Fixing the inputs, and identifying every record

"Download the latest from a repository" is convenient and cannot pin a research material: when a repository adds shards or edits content, the same command may read different data. Two things have to be recorded together — which version is logically intended, and which files were actually obtained. The source collects them in a manifest:

| Information | Question it answers later |
| --- | --- |
| Repository and immutable revision | Did the upstream content change |
| Language, subset, crawl batch | Which part of it was used this time |
| File name, byte size, and SHA-256 | Are the local files intact and identical to the record |
| Acquisition time | When the content was obtained |
| Version of the source and usage terms | Which statement of the terms guided the selection |
| Sampling method and selected record IDs | Why exactly these samples were chosen |

Hugging Face Datasets can pin a repository with a revision, and the resolved file list should be saved alongside it. Streaming avoids downloading everything in advance, but the data still crosses the network: it saves storage, not transfer, and a released version still needs pinned inputs or recorded record IDs rather than a note that streaming was used.

Each record then needs an identity, and the source distinguishes several identifiers that are easy to conflate: the project's own record `id`; the upstream `source.record_id` used to find the original; `parent_ids`, which record what the item was generated or split from; and `group_id`, the related group that must move together when splitting into training, validation, and test. If one source document produced two rewrites, the three records have different `id` values but share one group, and a second source that copied the original has to be linked into that group as well. This record of origin and derivation is what the source calls data lineage.

Two conventions matter for later analysis. An unrated item carries `null`, not `0`, because "not scored yet" and "scored lowest" are different facts. A missing language, source, or required field is marked as missing rather than filled with a plausible default such as English.

The processing order is fixed as well, and a few of its orderings are load-bearing. Relations are established before splits: if source A duplicates copy B and A also generated rewrite C, the three are considered together, and deleting A must not erase the link between A and C. Splits are fixed before any component that learns from data is fitted: a tokenizer trained from scratch uses the training split only, an external fixed scorer is recorded by version and applied by its own rules, and a self-trained scorer's labeled training and validation data are isolated as well. Each split is then tokenized and packed separately, so that a training tail is never concatenated with the head of a validation set. Fingerprints over all candidates can be used to detect cross-split duplication; that protects independence, whereas repeatedly consulting test scores to set thresholds would let the test set participate in the selection.

## Reconciliation and a stable split

Stages should be auditable. If a cleaning stage takes ten thousand records and keeps eight thousand, rejects fifteen hundred, and quarantines five hundred, the counts balance. Quarantine is the third bucket: a JSON parse failure or a missing required field cannot yet be judged, and it belongs in an explicit error record rather than being quietly discarded as low quality. For stages that only keep or reject, the check is

$$
N_{\mathrm{input}}=N_{\mathrm{kept}}+N_{\mathrm{rejected}}+N_{\mathrm{quarantined}}.
$$

When one input expands into several segments the equation no longer applies, and the parent records, child records, and their correspondence have to be checked instead of forcing the row counts to match. Each stage should leave three things: the kept data, the specific processing records, and the summary statistics. Deduplication additionally states which document represents the removed copies, scoring records the model and prompt version, and filtering records the threshold and the reason for each hit. Changing a threshold then only requires rerunning the dependent stages, while replacing the extractor changes the body text and normally requires rescoring, refingerprinting, and retokenizing; whether a cache can be reused depends on the inputs, configuration, code, and tool versions together.

Splitting is done on groups rather than on documents, so that related material never straddles the boundary. The source's helper takes an already-merged group identifier, normalizes the text for matching, computes a fingerprint, and hashes the group ID into a bucket:

```python
import hashlib
import unicodedata


def normalize_for_match(text: str) -> str:
    """Keep case, digits, and inline indentation; unify representation and trailing whitespace only."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def text_fingerprint(text: str) -> str:
    matching = normalize_for_match(text)
    if not matching.strip():
        raise ValueError("empty document")
    return hashlib.sha256(matching.encode("utf-8")).hexdigest()


def split_group(group_id: str, salt: str = "example-split-v1") -> str:
    if not group_id:
        raise ValueError("missing group_id")
    payload = f"{salt}\0{group_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    bucket = int.from_bytes(digest[:8], "big") % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "validation"
    return "test"
```

Python's built-in `hash()` is avoided because its result is affected by process-level randomization for some types and is therefore not a stable basis for splitting across processes. The hash result is mapped onto 0 to 999, with the first 900 buckets for training, the next 50 for validation, and the last 50 for test, which targets roughly 90/5/5 by group — not necessarily by document count or by token count. With only a few groups there may be no test group at all, so coverage has to be checked and a stratified design considered, rather than trying different salts until one produces a good test score. The normalizer exists for matching: keeping case, digits, and inline indentation reduces false merges, and it still needs checking per text type rather than being applied to all training text.

The snippet is retained as teaching material: it shows how normalization, fingerprinting, and group splitting fit together. It was not run as part of a corpus build, and it is not a corpus, not a run, and not a result about model quality.

## Sampling, token provenance, storage, and acceptance

Reading the first ten thousand records is good for checking fields and poor for estimating quality overall, because a file may be ordered by site, time, or quality and its opening may hold one narrow kind of content. To learn about the whole set, sample stratified by source, language, and length and save the selected IDs. Inspecting samples near a threshold is a different purpose: it is good at finding filter errors and does not represent the distribution. Shuffling a streaming buffer is not a global shuffle of all records, and the smaller the buffer the more the early ordering matters. Reservoir sampling draws a sample from a fully scanned stream without holding every record, but it still reads the whole candidate stream, and with limited resources the alternative is to fix a few shards and restrict the conclusion to them rather than letting a small sample speak for unchecked data.

A training file may end up as a stream of integers, so a separate index should record which document, position, and source group each token segment came from. That is what makes a later contamination finding, anomaly, or withdrawal locatable instead of a guess about a whole batch of integers. The tokenization and packing configuration states the tokenizer file version, special tokens, maximum length, truncation, trailing-block handling, and mask rules. Concatenating short documents reduces waste, but an end-of-sequence token only adds a marker and does not isolate attention; whether cross-document attention is allowed and whether positions reset are decided together with the training code. Discarding a trailing block can lose a larger share of a small shard, and carrying tails across files reduces waste but must not cross splits, so the number of discarded tokens is recorded to allow the effective ratio to be recomputed. The integer type used for token IDs has to hold every ID, including special tokens, and should not be chosen from the maximum seen in the current sample.

Storage is organized by role — configurations, manifests, raw inputs, intermediate results, rejected records, processed splits, tokenized output, reports, and release notes. The exact directory names are not a standard; what matters is that each file's purpose is clear. Git suits code, configurations, small manifests, and publishable reports, while large corpora and caches live elsewhere. The source text needed for traceability may contain sensitive information and need not be published in full. Not every intermediate copy has to be kept forever: the inputs required for a rebuild, important intermediate products, and the reports are worth retaining, regenerable caches are not — but if an upstream source later becomes unavailable and no necessary copy was kept, a rebuild cannot be promised. Disk estimates have to include deduplication signatures, sort temporaries, intermediate results, and retries, not only the final text.

Handover to training is checked in three layers rather than by the exit code of a script: files and records (checksums correct, IDs unique, bodies non-empty, counts reconciled), sources and splits (sources locatable, groups not crossing splits, duplication and contamination checks recorded), and training reads (token IDs valid, masks consistent, realized ratios plausible, restore positions correct). Not all deviations are equally serious. A group crossing a split is a structural error to fix; a small batch of Chinese text falling slightly below its target share may be ordinary sampling variation, and the two are handled differently. The report records the build version, the per-stage counts, distribution shifts, duplicate relations, token statistics, resource use, and known problems. Training results that were never run stay blank; expected values do not stand in for measurements.

Resuming after an interruption needs more than parameters and optimizer state. If the loader restarts from the beginning, or the shuffle order changes, the data seen afterwards is different. The record covers shard order, read position, sampler and random state, epoch, and parallelism settings. Some loaders can only preserve the distribution approximately once the worker or device count changes, and that has to be stated. A small check compares one path that reads a sequence of IDs continuously against one that saves state, restarts, and continues, and compares the second halves; that verifies data-read recovery, not that every other part of training is deterministic.

The last step of the first version is a single controlled change, such as one cleaning threshold, chosen so the records can answer which items changed status and why, which downstream products were rebuilt, and how the token totals per source moved. If the answers come from the records, the engineering has begun to support research; if all that is visible is that the new file is smaller, the records are still insufficient. The design can pin files, configurations, and versions to support a rebuild; it cannot guarantee that an upstream source stays available. It can split stably by merged groups; it does not guarantee exact ratios or automatic coverage. It can record every stage's changes; it does not show that a filtering method improves model capability. It can verify that a resumed run reads in the same order; it does not guarantee that the whole training run is deterministic.

## Where the Chinese original lives

The Chinese original is [Building a Traceable Small Pretraining Corpus](/posts/2026-10-15-llm-training-data-06-build/). It contains the full record schema, the Python helper with its original comments, the stage-by-stage acceptance table, the storage layout, the reporting requirements, and the reference list. Its limit is the one stated in its own closing table: the design makes a corpus inspectable and, where the inputs survive, rebuildable, and none of that is evidence that the resulting data makes a model better.
