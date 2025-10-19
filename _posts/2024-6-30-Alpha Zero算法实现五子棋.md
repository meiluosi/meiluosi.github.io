---
layout:     post
title:      Alpha Zero算法实现五子棋
subtitle:   深度强化学习在游戏AI中的创新应用
date:       2024-06-30
author:     冯宇
header-img: img/post-bg-recitewords.jpg
catalog: true
permalink: /2024/06/30/Alpha-Zero算法实现五子棋/
tags:
    - AlphaZero
    - 强化学习
    - 游戏AI
    - MCTS
    - 深度学习
    - 神经网络
categories: 
    - AI Applications
    - Game AI
    - Deep Learning
---
## 引言

AlphaZero 通过“自对弈 + 蒙特卡洛树搜索（MCTS）+ 深度神经网络”的闭环训练，在围棋、国际象棋、将棋等博弈中达到超人水平。本文以五子棋（Gomoku）为载体，给出可运行的最小实现与工程化建议。

## 1. 算法框架概览

- 策略-价值网络 fθ(s) → (p, v)
  - p：对每个合法动作的先验概率
  - v：状态 s 对当前玩家的胜率评估 ∈ [-1, 1]
- MCTS：用 fθ 提供的先验与价值引导树搜索，得到更强的策略 π
- 自对弈：使用 MCTS 作为“行为策略”与温度控制采样，收集 (s, π, z) 数据
- 训练：最小化 L = (z - v)^2 - π·log p + c||θ||^2

## 2. 环境与棋盘表示

- 棋盘：N×N（默认 9×9 或 11×11），落子即切换玩家
- 终止条件：任一方连续五子连线（横/竖/斜）或棋满和棋
- 状态编码：
  - 2 通道平面：当前玩家落子平面、对手落子平面
  - 可选：最近一步落子位置平面、轮到谁走的指示平面

## 3. 策略-价值网络（PyTorch）

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

## 4. MCTS 实现（PUCT）

```python
import math, numpy as np

class Node:
    def __init__(self, prior):
        self.P = prior   # 先验概率
        self.N = 0       # 访问次数
        self.W = 0.0     # 累计价值
        self.Q = 0.0     # 平均价值
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
            # 选择 UCB 最大的动作
            best_a, best_child, best_score = None, None, -1e9
            for a, ch in node.children.items():
                u = self.c_puct * ch.P * math.sqrt(node.N + 1) / (1 + ch.N)
                score = ch.Q + u
                if score > best_score:
                    best_a, best_child, best_score = a, ch, score
            path.append((node, best_a))
            node = best_child
            # 同步到下一状态由外部环境完成，这里简化只做第一次扩展
            if not node.children:
                break
        # 2) Expansion & Evaluation
        x = to_tensor(state)  # [1, 2, N, N]
        with torch.no_grad():
            logits, v = self.net(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]
            v = float(v.item())
        # 只保留合法动作的先验
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
        # 从根节点导出访问计数分布
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

说明：上面将“同步到下一状态”的细节略写，工程中需要复制环境并逐步推进；这里强调 PUCT 与先验融合的主线逻辑。

## 5. 自对弈与训练循环（最小雏形）

```python
from collections import deque

def encode_state(board, current_player):
    # board: shape (N, N), 值∈{0,1,-1}，1为先手，-1为后手
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

训练流程要点：
- 每回合用 MCTS 进行若干次搜索（如 100~800 次），得到 π 作为训练目标
- 终局得到 z ∈ {+1, -1, 0}（和棋）
- 在一批对局后训练网络若干步；可使用学习率调度与数据增强（旋转/翻转）

## 6. 参数建议与工程技巧

- 棋盘大小：9×9 更快收敛；11×11 难度更高
- MCTS：模拟次数 200~800；c_puct≈1.0~2.5；根节点 Dirichlet 噪声 α≈0.3（促进探索）
- 温度：开局 10 手内 T=1，之后 T→0 变贪婪
- 数据增强：八向对称（旋转/翻转）显著提升样本效率
- 混合精度与梯度累积：加速训练、稳定显存

## 7. 对弈与可视化

- 使用 web Canvas/HTML 在交互页展示棋盘，调用后端推理得到落子
- 在“算法可视化实验室”中嵌入五子棋对弈小部件，实时标注 MCTS 访问热度

## 8. 总结

本文给出 AlphaZero 在五子棋的最小可运行框架：策略-价值网络、PUCT MCTS、自对弈与训练。建议先在 9×9 上跑通闭环，再扩展更复杂的网络与更高模拟量，逐步达到稳定强力水平。

---

参考：
- Silver et al., Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)
- AlphaGo Zero Nature 2017