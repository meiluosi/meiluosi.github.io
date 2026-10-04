---
sourceId: 2024-07-12-actor-critic算法家族
slug: actor-critic-algorithm-family
title: The Actor–Critic Algorithm Family
editionLabel: ENGLISH OVERVIEW · TECHNICAL NOTE
published: 2024-07-12T00:00:00.000Z
description: An overview of Actor–Critic, A2C/A3C, PPO, SAC, advantage estimation, and practical training diagnostics.
tags:
  - Reinforcement Learning
  - Policy Gradient
  - PPO
  - SAC
category: Reinforcement Learning
lang: en
---

> **Scope note.** This is an English overview of a longer Chinese technical article. It summarizes the concepts, equations, code examples, and recommendations covered by the source; it does not report a benchmark or claim that the listed implementations were run successfully. The Chinese original contains the full A2C and PPO listings and additional implementation details.

## Why Actor–Critic?

REINFORCE estimates policy gradients from complete episode returns. That makes the method conceptually direct, but its gradient estimates can have high variance. Actor–Critic methods add a learned value function to evaluate outcomes and provide a lower-variance learning signal.

- The **Actor** is a policy, usually written as $\pi_\theta(a\mid s)$, that chooses actions.
- The **Critic** estimates state or action values, such as $V_\phi(s)$ or $Q_\phi(s,a)$.

The advantage measures how much better an action is than the policy's baseline expectation:

$$
A(s,a)=Q(s,a)-V(s)
$$

The policy update can then use the estimated advantage:

$$
\nabla_\theta J(\theta) \approx \mathbb{E}\left[\nabla_\theta\log\pi_\theta(a\mid s)\,A_\phi(s,a)\right].
$$

In the basic interaction loop, the Actor selects an action, the environment returns a reward and next state, and the Critic estimates a temporal-difference error or advantage. Those estimates update the Critic and Actor respectively. Bootstrapping from a learned value estimate allows updates before an entire episode has finished, though it also makes learning depend on the quality of that estimate.

## Estimating the advantage

The source article compares three estimators, with different trade-offs between bootstrapping and longer reward traces:

- **TD(0):** $A_t \approx r_t+\gamma V(s_{t+1})-V(s_t)$. It uses one-step bootstrapping.
- **n-step TD:** sums the next $n$ discounted rewards, then bootstraps from $V(s_{t+n})$.
- **Generalized Advantage Estimation (GAE):** combines temporal-difference residuals with an exponentially decaying weight:

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},\quad
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

The discount $\gamma$ and trace parameter $\lambda$ set how far future residuals contribute. In practice, the estimator and its parameters are part of the algorithm configuration, not a guarantee of performance.

## A2C and A3C: synchronous and asynchronous updates

**A3C** (Asynchronous Advantage Actor–Critic) runs multiple workers that collect experience in parallel. Each worker periodically computes n-step returns and gradients, then asynchronously updates shared global Actor and Critic parameters. The original article describes it as a way to parallelize sampling without an experience replay buffer.

**A2C** uses synchronized workers: they gather experience in parallel, then combine it for a batch update. This can make implementation and GPU batching more straightforward. The source contrasts A3C's asynchronous CPU-oriented updates with A2C's synchronous batching, and includes a PyTorch CartPole example with a shared feature network and separate policy and value heads.

These are architectural choices about collecting and applying updates. Which one is appropriate depends on the environment, implementation, hardware, and the behavior one needs to measure.

## PPO: constrain policy updates

Policy-gradient updates can be too small to learn efficiently or so large that the policy changes sharply. Proximal Policy Optimization (PPO) addresses this with a clipped objective. It reuses samples collected by an older policy, correcting their contribution with an importance ratio:

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

PPO-Clip optimizes:

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\left[\min\left(\rho_t(\theta)A_t,\;\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right].
$$

Clipping limits the incentive to move the new action probability too far from the old one in the direction favored by the advantage. The article uses $\epsilon=0.2$ in its example implementation and also describes GAE, minibatches, multiple optimization epochs, an entropy term, and gradient clipping. These are example configuration choices, not universal defaults.

## SAC: reward plus policy entropy

Soft Actor–Critic (SAC) extends the objective to reward both task performance and policy entropy. The entropy term encourages stochastic behavior and exploration:

$$
J(\pi)=\mathbb{E}_\pi\left[\sum_t\gamma^t\big(r_t+\alpha H(\pi(\cdot\mid s_t))\big)\right].
$$

The source describes three main components:

1. A stochastic Gaussian Actor.
2. Two Q Critics, whose minimum is used to reduce overestimation.
3. A temperature parameter $\alpha$ that controls the reward–entropy balance and can be adjusted automatically.

The Critic learns a soft target that includes the next action's log probability. The Actor is optimized to favor high-value actions while retaining entropy, and the temperature can be tuned toward a target entropy. The article's code discussion highlights reparameterized sampling, a tanh action transform with a log-probability correction, and twin Q networks; it does not provide a complete SAC training loop.

## Choosing an algorithm

The comparison below reflects the qualitative guidance in the Chinese source, rather than a controlled head-to-head evaluation:

| Algorithm | Source's sample-efficiency assessment | Stability assessment | Action spaces noted | Typical use suggested by the source |
| --- | --- | --- | --- | --- |
| A2C | Medium | Medium | Discrete and continuous | Simple environments and quick prototypes |
| A3C | Medium | Medium | Discrete and continuous | Parallel sampling with asynchronous updates |
| PPO | Medium–high | High | Discrete and continuous | General-purpose, stable policy optimization |
| SAC | High | High | Continuous | Continuous control and sample-efficient exploration |

These labels are broad heuristics. They do not replace evaluation on a particular task. The original article includes A2C and PPO code for CartPole, an SAC component sketch, and a longer discussion of hyperparameters and debugging.

## Practical diagnostics

When training behaves unexpectedly, the source recommends tracking more than episode reward. Useful signals include Actor and Critic losses, policy entropy, value estimates, mean advantage, and gradient norm. Plotting these together can help distinguish, for example, a changing reward curve from unstable value estimates or vanishing exploration.

The article also lists example PPO and SAC settings, including learning rates, discount factors, batch sizes, PPO clipping and epochs, and SAC replay-buffer and target-network settings. Treat these as starting points to inspect and validate for a chosen environment. Record the environment, implementation, configuration, and evaluation method when comparing runs.

## Further study

The original note points readers to the foundational A3C, PPO, and SAC papers, along with Stable-Baselines3, CleanRL, RLlib, OpenAI Spinning Up, and UC Berkeley's CS 285. It also suggests exploring PPO variants, TRPO, TD3, DreamerV3, and larger environments such as MuJoCo and Atari.

For the complete Chinese walkthrough—including the A2C CartPole implementation, full PPO listing, SAC network components, parameter examples, and plotting code—see the original article.
