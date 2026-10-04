---
sourceId: 2024-6-30-alpha-zero算法实现五子棋
slug: alpha-zero-gomoku
title: Implementing AlphaZero for Gomoku
description: A minimal policy-value network, PUCT search, self-play loop, and engineering notes for Gomoku.
published: 2024-06-30T00:00:00.000Z
tags:
  - Game AI and Search
  - Reinforcement Learning
category: AI Applications
lang: en
---

## Introduction

AlphaZero combines self-play, Monte Carlo tree search (MCTS), and a deep neural network in a closed training loop. It reached superhuman performance in Go, chess, and shogi. This note uses Gomoku as a smaller setting and presents a minimal implementation together with engineering suggestions.

## 1. Algorithm overview

- **Policy-value network** `fθ(s) → (p, v)`
  - `p`: prior probabilities for the legal actions.
  - `v`: the estimated outcome for the current player in state `s`, in `[-1, 1]`.
- **MCTS:** combines the network's priors and value estimate to search the game tree and produce a stronger policy `π`.
- **Self-play:** uses MCTS as the behavior policy, with temperature-controlled sampling, to collect `(s, π, z)` examples.
- **Training:** minimizes `L = (z - v)^2 - π·log p + c||θ||^2`.

## 2. Environment and board representation

- **Board:** an `N × N` grid (9×9 or 11×11 by default); each move passes the turn to the other player.
- **Terminal conditions:** five consecutive stones horizontally, vertically, or diagonally, or a full-board draw.
- **State encoding:**
  - Two planes: stones belonging to the current player and stones belonging to the opponent.
  - Optional planes: the most recent move and an indicator for whose turn it is.

## 3. Policy-value network (PyTorch)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=9, channels=64):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*board_size*board_size, board_size*board_size)
        )
        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size*board_size, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )
    def forward(self, x):
        h = self.conv(x)
        p = self.policy_head(h)
        v = self.value_head(h)
        return p, v
```

## 4. MCTS implementation (PUCT)

```python
import math, numpy as np

class Node:
    def __init__(self, prior):
        self.P = prior   # prior probability
        self.N = 0       # visit count
        self.W = 0.0     # accumulated value
        self.Q = 0.0     # mean value
        self.children = {}  # action -> Node

class MCTS:
    def __init__(self, net, board_size, c_puct=1.5):
        self.net = net
        self.board_size = board_size
        self.c_puct = c_puct
        self.root = Node(1.0)

    def search(self, state, legal_moves, to_tensor):
        path = []
        node = self.root
        # 1) Selection
        while node.children:
            # Select the action with the highest UCB score
            best_a, best_child, best_score = None, None, -1e9
            for a, ch in node.children.items():
                u = self.c_puct * ch.P * math.sqrt(node.N + 1) / (1 + ch.N)
                score = ch.Q + u
                if score > best_score:
                    best_a, best_child, best_score = a, ch, score
            path.append((node, best_a))
            node = best_child
            # The external environment advances the state. This example
            # simplifies the flow and expands only the first leaf.
            if not node.children:
                break
        # 2) Expansion & evaluation
        x = to_tensor(state)  # [1, 2, N, N]
        with torch.no_grad():
            logits, v = self.net(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            v = float(v.item())
        # Keep priors for legal actions only
        priors = {a: p[a] for a in legal_moves}
        norm = sum(priors.values()) + 1e-8
        for a in priors:
            priors[a] /= norm
        leaf = Node(1.0)
        leaf.children = {a: Node(priors[a]) for a in legal_moves}
        node.children = leaf.children
        # 3) Backup
        for n, _ in path:
            n.N += 1
            n.W += v
            n.Q = n.W / n.N
        return v

    def get_policy(self, temperature=1.0):
        # Derive a visit-count distribution from the root
        visits = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for a, ch in self.root.children.items():
            visits[a] = ch.N
        if temperature == 0:
            pi = np.zeros_like(visits)
            pi[np.argmax(visits)] = 1.0
            return pi
        visits = visits ** (1.0 / max(1e-6, temperature))
        if visits.sum() == 0:
            visits += 1.0
        return visits / visits.sum()
```

The example leaves out the mechanics for advancing the environment state during search. In a full implementation, the environment must be copied and stepped through; this sketch focuses on the PUCT selection rule and how it combines values with priors.

## 5. Minimal self-play and training loop

```python
from collections import deque

def encode_state(board, current_player):
    # board: shape (N, N), values in {0, 1, -1}; 1 moves first, -1 second
    cur = (board == current_player).astype(np.float32)
    opp = (board == -current_player).astype(np.float32)
    x = np.stack([cur, opp], axis=0)  # [2, N, N]
    return torch.from_numpy(x).unsqueeze(0)  # [1, 2, N, N]

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, pi, z):
        self.buf.append((s, pi, z))
    def sample(self, batch):
        idx = np.random.choice(len(self.buf), batch, replace=False)
        s, pi, z = zip(*[self.buf[i] for i in idx])
        return torch.cat(s), torch.tensor(np.array(pi), dtype=torch.float32), torch.tensor(z, dtype=torch.float32).unsqueeze(1)

def loss_fn(outputs, targets_pi, targets_v, l2=1e-4):
    logits, v = outputs
    loss_p = F.cross_entropy(logits, targets_pi.argmax(dim=1))
    loss_v = F.mse_loss(v, targets_v)
    l2_penalty = sum((p**2).sum() for p in net.parameters()) * l2
    return loss_p + loss_v + l2_penalty
```

Training-loop outline:

- Run MCTS for a number of simulations in each game (for example, 100–800) and use the resulting `π` as the training target.
- At the end of a game, assign `z ∈ {+1, -1, 0}` for a win, loss, or draw.
- Train the network for several steps after collecting a batch of games. A learning-rate schedule and data augmentation (rotations and reflections) can help.

## 6. Suggested parameters and engineering notes

- **Board size:** 9×9 converges faster; 11×11 is more challenging.
- **MCTS:** 200–800 simulations; `c_puct ≈ 1.0–2.5`; root Dirichlet noise `α ≈ 0.3` encourages exploration.
- **Temperature:** use `T = 1` for the first 10 moves, then gradually move toward greedy selection with `T → 0`.
- **Data augmentation:** the eight board symmetries (rotations and reflections) can improve sample efficiency.
- **Mixed precision and gradient accumulation:** can speed up training and reduce memory pressure.

## 7. Playing and visualization

- Use a web Canvas/HTML page to display the board and call a backend for move inference.
- Add a Gomoku widget to an “algorithm visualization lab” and display MCTS visit heatmaps in real time.

## 8. Summary

For a closer look at the search loop, return to the [MCTS overview](/en/posts/monte-carlo-tree-search-overview/) or step through the [MCTS playground](/lab/mcts/). Compare its random rollouts with the policy priors and value estimates used here: where does each method get its feedback?

This note outlines a minimal AlphaZero-style Gomoku system: a policy-value network, PUCT MCTS, self-play, and training. Start by getting the loop running on a 9×9 board, then try a more capable network and more simulations to improve play strength.

---

References:

- Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero).
- *Mastering the Game of Go without Human Knowledge* (AlphaGo Zero), Nature, 2017.
