---
sourceId: 2024-10-20-多模态大模型技术
slug: multimodal-models-overview
title: A Practical Overview of Multimodal Models
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-10-20T00:00:00.000Z
description: An overview of image-text alignment, vision-language architectures, training techniques, and evaluation from the Chinese tutorial.
tags:
  - Multimodal AI
  - Vision-Language Models
  - Deep Learning
  - Evaluation
category: Deep Learning
lang: en
---

> **Scope note.** This is an English overview of a Chinese technical tutorial published in 2024. It summarizes the architectures, code sketches, datasets, and metrics discussed there. The snippets are illustrative rather than a record of a completed training run, and the article does not report a reproducible benchmark or measured result. Model APIs, dataset versions, and the state of the field may have changed since publication.

## Align images and text with contrastive learning

The tutorial introduces CLIP as a dual-encoder approach: an image encoder and a text encoder map their inputs into a shared embedding space. Training uses paired image-text examples to make matching pairs more similar than non-matching pairs. The article gives a symmetric InfoNCE-style loss over the batch and projects both modalities into a common embedding dimension.

Once trained, the shared space can support zero-shot classification. Encode an image and candidate text descriptions, compute their similarities, and rank the candidates. This avoids fitting a separate classifier for every label set, but results depend on the model, prompt wording, domain, and candidate labels. Similarity scores are not calibrated probabilities unless a separate calibration process establishes that interpretation.

## From alignment to image-conditioned generation

The article presents BLIP as combining three objectives:

- **Image-text contrastive (ITC):** align image and text representations.
- **Image-grounded text generation (ITG):** generate text conditioned on an image, such as a caption.
- **Image-text matching (ITM):** classify whether an image and text correspond.

Its captioning example encodes an image, lets text representations attend to image features, and generates tokens until an end marker or length limit. This conveys a cross-attention pattern, not a full reproduction of a specific BLIP checkpoint or implementation.

## Connect visual features to a language model

The guide describes two ways to pass visual information to a language model. A projection layer can map vision-encoder features into the language model's embedding dimension, as in the simplified LLaVA-style sketch. A Q-Former-style module can instead use learned query vectors to select or compress information from image features, as in the BLIP-2 example.

The article then sketches a multimodal prompt by combining visual and text representations before generation. Real systems have additional choices around image resolution and crops, token ordering, instruction data, alignment training, and model-specific input formats. The example should be read as a conceptual interface, not as a direct implementation of GPT-4V internals.

## Training and evaluation choices

The source highlights hard-negative mining: use highly similar but mismatched image-text pairs to give a contrastive objective more difficult comparisons. It also discusses temperature scaling, which changes the sharpness of similarity-based logits. Both choices can affect optimization and should be assessed in the context of the training setup.

For evaluation, the article lists image-captioning, retrieval, and visual-question-answering datasets, then gives image-to-text recall at K as a retrieval metric. The code is a compact illustration and assumes that each row's matching image and text share an index; real datasets can have multiple valid captions, and evaluation must account for their annotation structure. Dataset names and sizes in the original post reflect its 2024 publication context and should be checked against the specific dataset version used.

The final “SimpleCLIP” example combines a ResNet vision encoder, a text encoder, projection layers, and a bidirectional contrastive loss. It is a tutorial sketch: the article does not include a dataset configuration, training log, checkpoint, or measured outcome for this code. Some library interfaces shown may also be outdated.

## Connection to RSI and AGI

Multimodal representations let a system connect language with observations such as images, which can broaden the evidence available to downstream reasoning. A multimodal model still needs task-specific evaluation for perception errors, grounding, uncertainty, and robustness. The architectures described here are building blocks; the article provides no evidence of autonomous learning or recursive self-improvement.

For the full Chinese tutorial—including CLIP loss and zero-shot examples, BLIP objectives, visual-token and Q-Former sketches, training techniques, benchmark references, and a SimpleCLIP example—see the original article.
