---
sourceId: 2026-10-20-llm-training-sft-02-messages
slug: llm-training-sft-02-messages-and-templates
title: "What One Instruction Sample Must Contain: Roles, Chat Templates, and Boundaries"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2026-10-20T00:00:00.000Z
description: "An overview of message roles, chat templates, multi-turn history, and truncation boundaries for a single supervised fine-tuning sample."
tags:
  - Supervised Fine-Tuning
  - Chat Templates
  - Instruction Data
category: LLM Training Research
series: LLM Training Research Notes · SFT Volume
seriesOrder: 2
lang: en
---

> **Scope note.** This is an English overview of a Chinese article on the anatomy of one supervised fine-tuning (SFT) example. Its Cedar model, experiment material, and JSON payloads are constructed teaching material written by the source author, not observations from a trained model. The article describes a data transformation, proposes an acceptance set, and cites fixed-version library documentation; it contains no fine-tuning run, no trained checkpoint, and no measured output. A sample that passes the checks described here has only been shown to be well-formed, not to describe any capability the model actually has.

## Three kinds of content in one sample

Task rules, the material to process, and the demonstration answer each do a different job. A minimal version of the extraction task looks like this:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Extract model, learning_rate, and steps from one experiment record. Return only a JSON object. model is a string or null, learning_rate is a number or null, steps is an integer or null. Use null for any field the material does not state explicitly, and add no speculation."
    },
    {
      "role": "user",
      "content": "Experiment record:\nThis run used Cedar-30M with a learning rate of 2e-4. The record does not state the number of training steps."
    },
    {
      "role": "assistant",
      "content": "{\"model\":\"Cedar-30M\",\"learning_rate\":0.0002,\"steps\":null}"
    }
  ]
}
```

The assistant `content` is a string that contains JSON text, so the outer data file has to escape the inner double quotes. What the model should finally produce is that JSON text, not the data file echoed back. The original article carries the same sample written in Chinese, with the task rules and the material stated in Chinese.

The distinction matters during processing. A sample file that parses as JSON only proves the outer record is legal; the assistant content still has to be parsed on its own, and its fields and types checked separately.

## Roles and the chat template

**system**, **user**, and **assistant** are role labels in the message structure. How they reach the model is decided by the chat template, and whether the model uses them as intended depends on prior training.

Do not assume from a field name that any base model understands a system instruction, and do not force a message into a template that does not support that role. Templates may merge roles, may require alternating turns, and may restrict the role set. For a model already trained for chat, start from its own template and conventions. For a pure base model, defining a new dialogue format is part of the training design: it means deciding the control tokens, the termination convention, and the coverage — not a cost-free substitution.

Role conventions also do not mean text inside the material is reliably isolated. If the material contains "ignore the previous instructions", the intent is that it stays extractable text rather than becoming a new task instruction. That requires a task definition, data coverage, and a dedicated test; adding a label does not solve it.

The model ultimately sees tokens, not a Python dictionary. The **chat template** converts a message list into text or tokens carrying roles and boundaries. A purely pedagogical form of the shape is:

```text
<SYS>task rules<END>
<USER>experiment material<END>
<ASSISTANT>{"model":"Cedar-30M",...}<END>
```

Those angle-bracket markers are illustrative only and must not be pasted in place of a model's real control tokens. Two models built on the same base can use different chat formats ([Transformers 4.57.1 chat templating](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)).

A reproducible sample therefore has three layers, and the source keeps them distinct:

| Layer | What it stores | What to check |
| --- | --- | --- |
| Task record | Raw material, rules, answer, and source | Is the task sensible, is the answer supported |
| Message list | Role, content, and order of each message | Are turns and roles correct |
| Model sequence | Template, token IDs, positions, labels | Are control markers, truncation, and supervision scope correct |

Reading only the first layer can miss template errors; reading only the last makes it hard to trace why an answer was labeled that way. The three layers have to correspond to each other.

## Training ends, generation begins, and markers are added once

During training there is a complete demonstration answer. At inference there is usually only the task and the material, and the model starts producing the assistant content. The two ends should be handled separately according to the template: a training record that already contains a complete answer does not need an extra empty assistant turn opened on top of it, while a generation input may need a prefix expressing "the assistant starts here". Transformers' `add_generation_prompt` supplies that kind of generation-start information, but whether to add it and what it contains depends on the template ([generation prompt conventions](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating#add_generation_prompt)).

A practical check is to print the complete training sample next to the real inference prompt. Does the inference prefix correspond to the sequence just before the answer actually began in training? If not, explain why before training starts. At the tokenization level the same discipline applies: text that looks identical on screen does not guarantee identical token IDs.

A template may also already contain start, end, or turn-boundary markers. Rendering to a string first and then letting the tokenizer add special tokens under its default behavior can duplicate them. The fixed-version documentation warns against re-adding special markers when re-encoding the result of `apply_chat_template(tokenize=False)`; letting the template do the tokenization removes that step's ambiguity ([special tokens in chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)). That is an interface convention, not a general rule to switch off all special tokens. What is needed is to compare the actual output against the model's template and check how many times the start marker appears and where, whether the answer's end marker is preserved, which boundary separates turns, and whether the inference stop condition recognizes the corresponding marker.

Ending a message and ending the whole generation need not use the same marker. Displayed similarity is not a reason to mix up their IDs and the stopping logic.

## Multi-turn history, truncation, and metadata

Suppose the first turn's material contains only the model name and the learning rate, and the model correctly outputs `steps:null`. The user then adds "the number of training steps is 800". A new answer can now update `steps` while keeping the earlier two values.

The training record must contain the history that was actually visible when the second answer was produced. Keeping only "the number of training steps is 800" while requiring the model to also output Cedar-30M and the learning rate teaches unsupported filling. The reverse error is equally damaging: putting information that only appears in the second turn into the first turn's input and then evaluating the first turn as "accurate extraction" changes the visible task information.

**Multi-turn history** also brings a supervision choice: train on every assistant turn, or only the last one? If an earlier assistant turn contains a known error, should it be kept as correction context or should the model imitate it? That is a decision about training intent, not something a JSON format settles, and the next note in this volume takes it up.

Truncation can turn a correct sample into a contradictory one. A message file being free of problems does not mean the truncated training sequence is still sound. Assume the material opens with the model name and closes with the learning rate, and that the program cuts the tail of the material to satisfy a length limit while keeping the complete demonstration answer: the model is now asked to output information that no longer exists in its input. In another case the answer is cut in half and only a fragment of JSON remains; some labels may still be valid, but the sample is no longer a demonstration of the complete task. So the final length and the truncation result should be checked after template rendering and tokenization. Several policies are possible — keep short samples whole, segment long material in a justified way, exclude incomplete records, or handle long context as a separate task — and which one applies depends on the task. A default truncation direction is not a substitute for a decision, and the count of excluded samples and their material types should be recorded, because length handling can quietly shift the task distribution.

Metadata is for tracking and must not leak into the prompt. For auditing one keeps `source_id`, annotation version, split, review status, and error category; these are useful, and not all of them belong in front of the model. If a source identifier in the training data happens to correlate with the answer or the category, the model can take the identifier as a shortcut and then lose performance when the field is absent at test time. A review conclusion or an answer hint must not be spliced into the input merely because it sits in the same object as the messages. One clear organization keeps input messages separate from administrative information and states explicitly which fields take part in rendering; before training, randomly sample and print the final prompts to check that no answer label, split name, or review note has leaked in. Rewrites, multi-turn splits, and synthetic variants that share a source should share grouping information so a split places them on the same side; deduplicating by final text hash alone cannot recognize a shared source.

## A small acceptance set, and what the Chinese original holds

Starting from a small set of hand-checked samples is more effective at finding problems than processing a hundred thousand records at once. The source lists the situations worth covering:

| Sample situation | What should be checked |
| --- | --- |
| A normal single-turn dialogue | The complete answer and the end boundary are both preserved |
| A missing step count | The assistant value is `null`, not a guess |
| Material containing quotes, newlines, and code | Outer escaping does not corrupt the actual content |
| Material containing role-style strings | It is not mistaken for a new message boundary |
| A second turn supplying material | The later turn can see the necessary history |
| Exceeding the length limit | The stated policy is applied and counted |
| A template that does not support a role | It errors clearly or converts by a defined rule |

For each, store the messages, the rendered text, the token IDs, and the expected boundaries, then hand them to the label-mask check that follows. Keeping only the final token file makes problems very hard to trace back. This acceptance set exists to check the data conversion; it is not evidence that a model has these capabilities.

The full Chinese article carries the complete JSON sample in Chinese, the three-layer table, the seven-situation acceptance list, the multi-turn and truncation walkthroughs, and the reference list with the Transformers 4.57.1 links — material summarized but not reproduced here. The original is at [the Chinese source article](/posts/2026-10-20-llm-training-sft-02-messages/); the next note covers [which tokens should carry the loss](/en/posts/llm-training-sft-03-loss-mask/). The main limit is unchanged: this is a description of how to construct and check one sample, and no fine-tuning was run to see whether the construction leads to better behavior.
