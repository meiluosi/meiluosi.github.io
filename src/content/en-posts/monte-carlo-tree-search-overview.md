---
sourceId: 2024-08-20-mcts算法深度解析
slug: monte-carlo-tree-search-overview
title: "Monte Carlo Tree Search: Core Ideas and Variants"
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-08-20T00:00:00.000Z
description: An overview of MCTS, UCT and PUCT, engineering variants, a tic-tac-toe example, and evaluation considerations.
tags:
  - Monte Carlo Tree Search
  - Game AI
  - Reinforcement Learning
  - Search
category: Game AI
lang: en
---

> **Scope note.** This is an English overview of a Chinese technical tutorial. The article contains pseudocode and implementation examples for MCTS, UCT, PUCT, and tic-tac-toe, plus example evaluation procedures. It does not include run logs or measured playing-strength results. Treat code and parameter values as instructional starting points; the original post itself notes that some game-environment details are abbreviated.

## The MCTS loop

Monte Carlo Tree Search (MCTS) allocates computation to promising parts of a game tree instead of exhaustively searching every continuation. The tutorial introduces the four steps:

1. **Selection:** follow a tree policy from the root toward a leaf.
2. **Expansion:** add a child for an action that has not yet been tried.
3. **Simulation:** estimate the outcome from the new node, in the basic version by playing out the game.
4. **Backpropagation:** update visit and outcome statistics along the path.

The cycle repeats for a chosen computation budget, after which a move is selected using the search statistics. The tutorial contrasts this selective search with full game-tree enumeration and presents a compact node implementation. Its sample code is a teaching sketch: a real solver must define legal actions, terminal states, player perspectives, reward sign changes, and draw handling consistently.

## UCT balances exploration and exploitation

Upper Confidence bounds applied to Trees (UCT) augments an action's observed value with a term that favors less-visited actions. In the tutorial's notation, a common form is:

$$
\operatorname{UCT}(s_i) = \frac{w_i}{n_i} + c\sqrt{\frac{\ln N}{n_i}},
$$

where the first term represents an empirical outcome value, the second encourages exploration, and $c$ controls their trade-off. This formula is only meaningful with a clearly defined reward perspective and visit-count convention. In alternating-player games, implementations often express values from the perspective of the player choosing at the current node; careless backpropagation can make the selection logic inconsistent.

The source includes a UCT node example and a sketch for comparing exploration constants. Such a comparison needs a fixed opponent or baseline, enough games to account for randomness, and the same game rules and computation budget. The article does not report results from running that tuning procedure.

## PUCT adds a prior from a policy

AlphaGo- and AlphaZero-style search can use a policy network's prior probability for each action to guide exploration. The tutorial gives a PUCT score of the form:

$$
Q(s,a) + c_{\text{puct}} P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)},
$$

where $P(s,a)$ is the prior and $Q(s,a)$ is the backed-up action value. This lets search combine learned preferences with outcomes found during tree traversal. The article's AlphaZero-style example also sketches root noise during self-play and a temperature-controlled choice from visit counts. These are components of a training setup, not evidence that a particular model improved or reached a given playing strength.

## Engineering variants in the tutorial

The guide surveys several techniques for changing search cost or information flow:

- **RAVE** shares information about moves seen later in a rollout with estimates for earlier tree decisions, which can help in some domains but introduces an additional blending rule.
- **Virtual loss** discourages parallel workers from duplicating the same path while simulations are in flight; synchronization and cleanup after failed workers matter in a production implementation.
- **Progressive widening** limits how quickly children are added when a state has many possible actions, using an expansion rule to decide when to consider more.
- **Root parallelization** runs independent searches and combines their statistics, trading communication simplicity against shared-tree information.
- **Value-network evaluation** can replace or shorten random rollouts, connecting MCTS to learned evaluation as in AlphaZero-style systems.

These alternatives address different bottlenecks. The right choice depends on branching factor, rollout quality, model cost, hardware, and the evaluation objective; a more elaborate search is not automatically stronger under a fixed time budget.

## A small implementation and how to evaluate it

The source walks through a tic-tac-toe implementation with a game state, terminal checks, legal moves, UCT selection, expansion, random rollout, and result propagation. It proposes comparing search budgets against a random opponent and separately varying the UCT exploration constant.

Those are useful experiment shapes, but the example evaluator is not a reported result. To interpret playing strength, specify the opponent, color or seat assignment, number of games, random seeds, tie handling, and compute budget. A comparison should include uncertainty and preferably a baseline stronger than random once the basic implementation works. The post gives no completed tournament table or measured win rate.

## Connection to RSI and AGI

MCTS provides a concrete decision loop: search candidate actions, use simulated or learned feedback, update estimates, and choose a move. In AlphaZero-like systems, self-play can also produce data for updating a policy and value model. This makes game search a useful small-scale setting for studying feedback-driven improvement, but a working search loop alone does not demonstrate recursive self-improvement. That would require evidence that successive system changes yield reliable gains under evaluations that remain informative across iterations.

For the full Chinese tutorial—including node pseudocode, UCT and PUCT examples, RAVE, virtual loss, progressive widening, tic-tac-toe code, and suggested evaluation procedures—see the original article.

## Follow the search further

Try the [MCTS playground](/lab/mcts/) with different search budgets and exploration weights. Its single-agent tree uses fixed terminal rewards so that you can follow each phase and watch the root recommendation change.

Then continue to [AlphaZero-style Gomoku](/en/posts/alpha-zero-gomoku/): how do search statistics become training signals, and how does an updated policy-value network affect the next round of self-play?
