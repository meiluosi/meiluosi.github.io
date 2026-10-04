---
sourceId: 2024-10-06-rag检索增强生成实战
slug: rag-retrieval-augmented-generation
title: Retrieval-Augmented Generation in Practice
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-10-06T00:00:00.000Z
description: An overview of a RAG pipeline, retrieval strategies, evaluation, and production considerations from the Chinese tutorial.
tags:
  - RAG
  - Retrieval
  - AI Applications
  - Evaluation
category: NLP
lang: en
---

> **Scope note.** This is an English overview of a Chinese implementation guide. The source includes code examples and illustrative evaluation output, but does not provide run logs, data, model versions, or a reproducible benchmark. The examples below describe design options; they are not a report of a verified production system or measured results.

## What retrieval-augmented generation does

Retrieval-augmented generation (RAG) adds an external retrieval step to a language-model application. Given a question, the system finds relevant material from a document collection and supplies it to the model as context for an answer. A typical flow is:

```text
documents → clean and chunk → embed and index
question → retrieve candidate chunks → rerank and select context
context + question → generate an answer, ideally with source references
```

This can make a system use a collection that changes independently of the model's training data and can expose which documents supported an answer. Retrieval does not guarantee that the selected context is correct, complete, current, or followed faithfully by the model. Those properties need to be evaluated.

## Prepare and index documents

The tutorial sketches loaders for formats such as PDF, Markdown, text, and HTML, followed by text cleaning and chunking. Chunk size and overlap affect both retrieval precision and whether a passage contains enough context. Fixed-size chunks are simple; sentence, paragraph, recursive, and semantic boundaries make different trade-offs. There is no single setting that fits every corpus.

An embedding model maps chunks and queries into vectors. A vector index can then retrieve semantically similar passages. The source demonstrates a Chroma-based workflow and discusses FAISS and Milvus as other storage options. These examples use framework APIs that may have changed since the article was written, so versions and current documentation should be checked before reuse.

## Retrieve useful context

The guide covers several ways to improve or vary retrieval:

- **Top-K similarity search** returns a fixed number of candidates. The value of K affects how much context is available and how much irrelevant material may be included.
- **Hybrid search** combines lexical matching, such as BM25, with vector similarity. This can help when exact terms, identifiers, or names matter alongside semantic matches.
- **Reranking** applies a second scoring step to a larger candidate set before selecting a smaller context set.
- **Multi-query retrieval** reformulates a question into several searches and combines their results.
- **Parent-document retrieval** searches smaller chunks while returning a larger surrounding section.
- **HyDE and self-query retrieval** generate a hypothetical document or structured query to guide search; these add model calls and can introduce their own errors.
- **Context compression** filters or condenses retrieved material before generation.

These are options to compare against a simple baseline, not guaranteed improvements. A useful evaluation should include representative questions, relevant-document judgments, and cases where the answer is absent from the corpus.

## Generate answers with provenance

The source's generation prompt asks the model to answer from supplied context and acknowledge when it lacks enough information. It also discusses citations and multi-turn question handling. A production design should keep source identifiers alongside each chunk, make citations traceable to the original document, and distinguish retrieved evidence from generated explanation. Prompt instructions can encourage grounded answers but cannot enforce grounding on their own.

For follow-up questions, conversation history may clarify references, but it should not silently replace the current retrieval step. Rewriting a follow-up into a standalone search query and retrieving fresh evidence helps make the retrieval decision inspectable.

## Evaluate the pipeline

The tutorial proposes measuring retrieval and answer quality separately. Retrieval measures can include precision, recall, and ranking metrics such as MRR or NDCG, based on relevance judgments. Answer evaluation can inspect factual support, relevance, and whether the response addresses the question. The article also introduces RAGAS and an A/B testing outline.

The RAGAS scores shown in the Chinese article are example output, not a reported experiment: no evaluation dataset, judge configuration, or run artifact is supplied. Treat any automated score as a diagnostic signal and validate it against human-reviewed examples. Track failures by stage—for example, missing evidence, poor ranking, unsupported generation, or incorrect abstention—so a change to one component can be assessed without obscuring regressions elsewhere.

## Production considerations

The source outlines a FastAPI service and Docker packaging. A deployed system also needs to account for document refreshes, access permissions, latency, model and embedding costs, prompt-injection content in retrieved documents, and observability. Logging should preserve enough information to debug retrieval and generation while respecting data-handling requirements. These operational concerns are part of answer quality: stale or unauthorized context can make an otherwise plausible answer wrong or unsafe.

## Connection to RSI and AGI

RAG provides an external information loop: retrieve evidence, generate a response, and evaluate where the result failed. That can support more informed model-assisted work, but it does not update model capability by itself and is not evidence of recursive self-improvement. An RSI-oriented system would need controlled changes to the retrieval or reasoning process, a stable evaluation set, and measurements showing that successive changes improve performance without hiding regressions.

For the full Chinese guide—including document loaders, chunking examples, Chroma indexing, retrieval variants, RAGAS examples, FastAPI, and deployment notes—see the original article.
