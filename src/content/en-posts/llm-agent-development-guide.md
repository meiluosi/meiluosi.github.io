---
sourceId: 2024-10-13-llm-agent开发指南
slug: llm-agent-development-guide
title: A Practical Guide to LLM Agents
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-10-13T00:00:00.000Z
description: An overview of agent loops, tools, memory, planning, multi-agent workflows, and evaluation.
tags:
  - AI Agents
  - Tool Use
  - Planning
  - Evaluation
category: AI Applications
lang: en
---

> **Scope note.** This is an English overview of a longer Chinese technical guide. It summarizes the architectures and illustrative code patterns in the source; it is not a report of a personal implementation or benchmark. The article's framework APIs and examples reflect its writing period and may need updating before use. The Chinese original contains the full code listings.

## What makes a language model an agent?

A language model can produce a response from a prompt. An agent adds a loop around that model: it receives an objective, chooses an action, uses a tool or environment, observes the result, and decides what to do next. The source guide groups the pieces into reasoning, tools, planning, and memory.

One compact view is:

```text
input → plan or decide → act with a tool → observe → continue or finish
```

The loop gives a model a way to interact with external systems, but it does not make every action correct. Tool quality, context, stopping rules, and evaluation all affect what the agent can accomplish.

## ReAct: alternate reasoning and action

The guide introduces ReAct as an alternation between **Thought**, **Action**, and **Observation**. An agent might search for a fact, read the result, then call a calculator. It repeats the cycle until it has an answer or reaches a configured step limit.

This structure makes intermediate tool use visible and gives the system a place to incorporate observations. A practical implementation needs to parse the model's action, check that the requested tool exists, pass valid inputs, and handle tool failures. The source's from-scratch example uses a maximum-step setting to bound the loop.

Function calling packages the action as structured arguments rather than asking a model to emit a free-form string. The guide describes a tool registry in which each tool has a name, description, input schema, and callable function. Clear schemas make it easier for a model to select and call a tool, while the application remains responsible for executing it and returning the result.

## Memory and task planning

The article distinguishes several kinds of memory:

- **Conversation history** keeps recent messages in context. This is straightforward, but the amount of text grows with the conversation.
- **Long-term retrieval** stores information in a searchable representation and brings back items relevant to a new query. The source demonstrates this pattern with a vector store.
- **Entity memory** records structured facts about named people or objects, such as properties associated with a person.

Memory changes what an agent can use in later steps; it does not verify whether a stored fact is accurate or still current. Retrieval and update behavior therefore matter alongside the storage choice.

For tasks with multiple steps, **Plan-and-Execute** first asks a planner to decompose an objective, then sends each step to an executor and combines the results. A hierarchical version assigns subtasks to agents that can handle them. The source examples illustrate these patterns with code sketches rather than a measured comparison of planning strategies.

## Autonomous and multi-agent workflows

An autonomous loop repeatedly builds context from a goal, memory, and available tools; asks a model for a next action; executes the action; stores the result; and stops on completion or an iteration limit. The guide also sketches a self-critique pattern in which an agent reviews an answer and tries to revise it.

These patterns add opportunities to recover from an incomplete first attempt, but repeated model calls alone do not establish improvement. A critique needs a useful standard, and the system needs a way to check whether a revision actually performs better.

The multi-agent section describes two arrangements:

1. A role-based workflow, inspired by software development, passes work between product, architecture, engineering, and testing agents.
2. A debate workflow lets several agents propose or refine answers before a judge selects one.

The examples show how responsibilities can be separated. They do not establish that adding agents improves quality; that question requires comparison against a simpler baseline on the same tasks.

## Evaluate the whole loop

The source proposes measuring more than the final answer. Its example evaluator tracks:

- success against expected outputs;
- steps or model interactions;
- API cost; and
- execution time.

Those measures describe different trade-offs. A system may succeed more often while also taking longer or costing more. Prompt changes should be compared on a fixed set of test cases, with the same correctness checks and operating conditions, rather than judged from a few appealing demonstrations.

## Connection to recursive self-improvement

An agent loop has a useful structural resemblance to a learning loop: act, observe feedback, assess the outcome, then choose another action. For RSI, the important question is whether successive changes produce reliable, measurable gains under an evaluation process that remains informative as the system changes.

Planning, memory, tools, or self-critique can support such a process, but none is evidence of RSI by itself. This guide is a conceptual and implementation-oriented survey; it reports no agent benchmark or verified self-improvement result.

For the complete Chinese walkthrough—including ReAct, function-calling, memory, planning, AutoGPT-style loops, multi-agent examples, and evaluation code—see the original article.
