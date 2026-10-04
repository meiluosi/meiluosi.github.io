---
sourceId: 2024-6-30-应用cfr实现德州扑克对战
slug: kuhn-poker-cfr
title: Implementing CFR for Kuhn Poker
description: A compact CFR implementation for Kuhn Poker, with practical steps for scaling the approach toward Texas Hold’em.
published: 2024-06-30T00:00:00.000Z
tags:
  - Game AI and Search
  - Reinforcement Learning
category: Game AI
lang: en
---

## Introduction

Counterfactual Regret Minimization (CFR) is a classic algorithm for finding Nash equilibria in imperfect-information games. This note presents a complete CFR implementation for the small game Kuhn Poker, then outlines practical engineering steps for applying the method to Texas Hold’em through abstraction and opponent modeling.

## 1. The core idea behind CFR

- **Information set:** an equivalence class of states that a player cannot distinguish from their observations in an imperfect-information game.
- **Strategy:** a probability distribution over available actions at each information set, `σ(I)`.
- **Regret:** the difference in value between choosing an action and following the current strategy.
- **CFR iterations:** accumulate positive counterfactual regret `R⁺(I, a)`, then normalize it to form a new strategy with regret matching.

## 2. Kuhn Poker

- **Cards:** three cards—J, Q, and K—with one card dealt to each player.
- **Actions:** pass or bet, followed by a call or fold when facing a bet.
- **Settlement:** compare cards at showdown, or award the pot to the player whose opponent folds.

## 3. Python implementation: CFR for Kuhn Poker

```python
import random
from collections import defaultdict

CARDS = ['J', 'Q', 'K']

class KuhnCFR:
    def __init__(self):
        self.regret_sum = defaultdict(lambda: [0.0, 0.0])   # two actions per information set: [pass, bet]
        self.strategy_sum = defaultdict(lambda: [0.0, 0.0])

    def get_strategy(self, info_set):
        regrets = self.regret_sum[info_set]
        # Regret matching
        positive = [max(r, 0.0) for r in regrets]
        normalizer = sum(positive)
        if normalizer > 0:
            strategy = [r/normalizer for r in positive]
        else:
            strategy = [0.5, 0.5]
        # Accumulate the average strategy
        for i in range(2):
            self.strategy_sum[info_set][i] += strategy[i]
        return strategy

    def get_average_strategy(self, info_set):
        s = self.strategy_sum[info_set]
        total = sum(s)
        if total > 0:
            return [x/total for x in s]
        return [0.5, 0.5]

    def cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player
        # Terminal-state checks
        if len(history) >= 2:
            terminal_pass = history[-1] == 'p'
            double_bet = history[-2:] == 'bb'
            if terminal_pass:
                # p or bp
                if history == 'pp':
                    return 1 if cards[player] > cards[opponent] else -1
                else:
                    return 1
            elif double_bet:
                return 2 if cards[player] > cards[opponent] else -2
        # Information set: private card + action history
        info_set = cards[player] + history
        strategy = self.get_strategy(info_set)
        # Recursively evaluate the opponent's value
        if player == 0:
            v_pass = -self.cfr(cards, history + 'p', p0*strategy[0], p1)
            v_bet  = -self.cfr(cards, history + 'b', p0*strategy[1], p1)
        else:
            v_pass = -self.cfr(cards, history + 'p', p0, p1*strategy[0])
            v_bet  = -self.cfr(cards, history + 'b', p0, p1*strategy[1])
        v = strategy[0]*v_pass + strategy[1]*v_bet
        # Counterfactual regret
        regrets = [v_pass - v, v_bet - v]
        if player == 0:
            reach = p1
        else:
            reach = p0
        self.regret_sum[info_set][0] += reach * regrets[0]
        self.regret_sum[info_set][1] += reach * regrets[1]
        return v

    def train(self, iterations=100000):
        util = 0.0
        deck = CARDS[:]
        for _ in range(iterations):
            random.shuffle(deck)
            util += self.cfr(deck[:2], '', 1.0, 1.0)
        print(f"Average game value for Player 0: {util/iterations:.4f}")
        # Print the average strategy
        strat = {I: self.get_average_strategy(I) for I in self.strategy_sum}
        return strat

if __name__ == '__main__':
    solver = KuhnCFR()
    avg_strategy = solver.train(50000)
    for I in sorted(avg_strategy.keys()):
        s = avg_strategy[I]
        print(I, { 'pass': round(s[0],3), 'bet': round(s[1],3) })
```

The resulting average strategy should approach the kind of approximate Nash strategy reported in the literature. For example, a player holding K tends to bet more often, while a player holding J tends to pass more often.

## 4. Toward a practical Texas Hold’em implementation

Applying CFR to full Texas Hold’em requires abstraction to reduce its enormous state space:

- **Information-set abstraction:** bucket states by public-card texture, hand strength or board texture, and action history.
- **Action abstraction:** discretize the betting space into fixed sizes, such as `0.33 pot`, `0.66 pot`, and `pot`.
- **Public-card sampling:** use Public Chance Sampling or External Sampling CFR.
- **Strategy representation:** use tabular policies or function approximation, including combinations such as NFSP, CFR+, or Deep CFR with neural networks.
- **Acceleration:** train through parallel self-play, save the average strategy incrementally, and use CFR+ regularization and pruning to speed up convergence.

Suggested progression:

1. Validate the abstraction and CFR+ implementation on Leduc Poker, a small public-card game.
2. Add the rounds of Texas Hold’em—pre-flop, flop, turn, and river—along with an action abstraction.
3. Evaluate with exploitability or win rate against fixed opponents.

## 5. Conclusion

This note provides a basic CFR implementation and an engineering path toward Texas Hold’em. A next step would be to add CFR+, external sampling, and a function-approximation policy network to support larger-scale training and play.
