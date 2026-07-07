---
layout: post
title: "Actor-Critic 算法家族"
subtitle: "A2C/A3C、PPO、SAC 全面梳理"
date: 2024-07-12
author: "冯宇"
header-img: "img/post-bg-recitewords.jpg"
catalog: true
permalink: /2024/07/12/Actor-Critic算法家族/
tags:
  - 强化学习
  - Actor-Critic
  - PPO
  - SAC
categories:
  - Reinforcement Learning
---

## 引言

在上一篇文章中，我们详细介绍了策略梯度方法，特别是 REINFORCE 算法。虽然 REINFORCE 理论优雅，但其高方差问题严重制约了学习效率。**Actor-Critic** 方法应运而生，成为现代强化学习的主流范式。

**Actor-Critic 的核心思想**：
- **Actor（演员）**：策略网络 $\pi_\theta(a|s)$，负责选择动作
- **Critic（评论家）**：值函数网络 $V_\phi(s)$ 或 $Q_\phi(s,a)$，负责评估动作质量

**相比 REINFORCE 的优势**：
- ✅ **低方差**：使用学到的值函数替代 Monte Carlo 回报
- ✅ **在线学习**：可以在每步更新，不必等到回合结束
- ✅ **Bootstrap**：利用时序差分（TD）学习加速收敛

本文将系统介绍 Actor-Critic 家族的核心算法：A2C/A3C、PPO、SAC，并提供完整的代码实现。

## 1. Actor-Critic 基础框架

### 1.1 核心原理

**策略梯度回顾**：

$$
\nabla_\theta J(\theta) = E_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a) \right]
$$

**关键改进**：用学到的 Critic $Q_\phi(s,a)$ 或 $V_\phi(s)$ 替代真实 $Q^\pi(s,a)$。

**优势函数（Advantage）**：

$$
A(s,a) = Q(s,a) - V(s)
$$

**Actor-Critic 梯度**：

$$
\nabla_\theta J(\theta) \approx E \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A_\phi(s,a) \right]
$$

### 1.2 算法架构

```
环境交互循环：
1. Actor 根据当前策略 π_θ(a|s) 选择动作 a
2. 执行动作，观察奖励 r 和下一状态 s'
3. Critic 计算 TD 误差或优势函数
4. 使用 TD 误差更新 Critic 参数 φ
5. 使用优势函数更新 Actor 参数 θ
```

### 1.3 优势函数的估计方法

#### 方法 1：TD(0) 估计

$$
A(s,a) = r + \gamma V(s') - V(s) = \delta_t
$$

#### 方法 2：n-step TD

$$
A(s,a) = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)
$$

#### 方法 3：GAE（广义优势估计）

$$
A^{\text{GAE}(\gamma,\lambda)}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

## 2. A2C/A3C：同步与异步优势 Actor-Critic

### 2.1 A3C（Asynchronous Advantage Actor-Critic）

**核心创新**（Mnih et al., 2016）：
- 多个 worker 并行采样，异步更新全局网络
- 无需经验回放（Experience Replay）
- CPU 多线程，成本低

**算法流程**：

```
全局网络：θ_global（Actor）、φ_global（Critic）

每个 Worker i：
  1. 从全局网络复制参数：θ_i ← θ_global, φ_i ← φ_global
  2. 采样 n 步轨迹 {s_t, a_t, r_t}
  3. 计算 n-step 回报和优势
  4. 计算梯度 ∇θ 和 ∇φ
  5. 异步更新全局网络（加锁）
  6. 重复
```

### 2.2 A2C（Advantage Actor-Critic）

**改进**：
- **同步更新**：所有 worker 同时采样，集中更新
- **批量梯度**：减少梯度噪声
- **GPU 友好**：批量操作更适合 GPU 加速

**A2C vs A3C**：

| 维度 | A3C | A2C |
|------|-----|-----|
| 更新方式 | 异步 | 同步 |
| 硬件 | CPU 多线程 | GPU 批处理 |
| 梯度噪声 | 较大 | 较小 |
| 实现难度 | 需要线程同步 | 简单 |
| 流行度 | 历史经典 | 当前主流 |

### 2.3 A2C 完整实现（CartPole）

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class ActorCriticNetwork(nn.Module):
    """共享特征提取的 Actor-Critic 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor 头（策略）
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic 头（值函数）
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = torch.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
    def select_action(self, state):
        """选择动作并返回概率、值"""
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, state_value = self.network(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), state_value.squeeze()
    
    def compute_returns(self, rewards, values, dones, last_value):
        """计算 n-step returns"""
        returns = []
        R = last_value
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        
        return torch.tensor(returns)
    
    def update(self, states, actions, log_probs, returns, values):
        """更新 Actor-Critic 网络"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.stack(log_probs)
        returns = returns.detach()
        old_values = torch.stack(values)
        
        # 前向传播
        action_probs, state_values = self.network(states)
        dist = Categorical(action_probs)
        
        # 计算优势函数
        advantages = returns - old_values.detach()
        
        # Actor loss（策略梯度）
        new_log_probs = dist.log_prob(actions)
        actor_loss = -(new_log_probs * advantages).mean()
        
        # Critic loss（值函数）
        critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
        
        # 熵正则化（鼓励探索）
        entropy = dist.entropy().mean()
        
        # 总损失
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }

def train_a2c(env_name='CartPole-v1', n_episodes=1000, n_steps=5):
    """训练 A2C 智能体"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2CAgent(state_dim, action_dim)
    
    episode_rewards = []
    running_reward = 0
    
    state, _ = env.reset()
    
    for episode in range(n_episodes):
        # 收集 n-step 轨迹
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        for step in range(n_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            
            state = next_state
            running_reward += reward
            
            if done:
                state, _ = env.reset()
                episode_rewards.append(running_reward)
                running_reward = 0
        
        # 计算最后状态的值（bootstrap）
        with torch.no_grad():
            _, last_value = agent.network(torch.FloatTensor(state).unsqueeze(0))
            last_value = last_value.squeeze()
        
        # 计算 returns
        returns = agent.compute_returns(rewards, values, dones, last_value)
        
        # 更新网络
        metrics = agent.update(states, actions, log_probs, returns, values)
        
        if (episode + 1) % 100 == 0 and episode_rewards:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, "
                  f"Entropy: {metrics['entropy']:.4f}")
    
    env.close()
    return episode_rewards, agent

# 训练
rewards_a2c, agent_a2c = train_a2c()

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(rewards_a2c, alpha=0.3, label='Raw')
plt.plot(np.convolve(rewards_a2c, np.ones(50)/50, mode='valid'), label='Moving Avg')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('A2C Training on CartPole-v1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 3. PPO：近端策略优化

### 3.1 PPO 的动机

**问题**：策略梯度更新步长难以控制
- 步长太小 → 学习慢
- 步长太大 → 策略崩溃（performance collapse）

**解决方案**：限制策略更新的幅度，确保新策略不会偏离旧策略太远。

### 3.2 重要性采样（Importance Sampling）

**目标**：使用旧策略 $\pi_{\theta_{\text{old}}}$ 的数据优化新策略 $\pi_\theta$。

**重要性权重**：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

**策略梯度（重要性采样形式）**：

$$
\nabla_\theta J(\theta) = E_{\theta_{\text{old}}} \left[ \rho_t(\theta) \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\theta_{\text{old}}}(s_t, a_t) \right]
$$

### 3.3 PPO-Clip 算法

**目标函数**（Schulman et al., 2017）：

$$
L^{\text{CLIP}}(\theta) = E_t \left[ \min \left( \rho_t(\theta) A_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

**关键参数**：$\epsilon$（通常取 0.1 或 0.2）

**直观解释**：
- 当 $A_t > 0$（好动作）：
  - 如果 $\rho_t > 1+\epsilon$（新策略概率过大），裁剪为 $1+\epsilon$
- 当 $A_t < 0$（坏动作）：
  - 如果 $\rho_t < 1-\epsilon$（新策略概率过小），裁剪为 $1-\epsilon$

**防止过度更新**：保守地增加好动作概率、减少坏动作概率。

### 3.4 PPO 完整实现

```python
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, value_coef=0.5, entropy_coef=0.01, 
                 n_epochs=10, batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, state_value = self.network(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, log_prob, reward, done, value):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def compute_gae(self, next_value):
        """计算 GAE"""
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - self.dones[step]) * gae  # λ = 0.95
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages)
        returns = advantages + torch.tensor(self.values)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """PPO 更新"""
        # 获取下一状态的值（bootstrap）
        with torch.no_grad():
            if self.dones[-1]:
                next_value = 0
            else:
                _, next_value = self.network(torch.FloatTensor(self.states[-1]).unsqueeze(0))
                next_value = next_value.item()
        
        # 计算 GAE 和 returns
        advantages, returns = self.compute_gae(next_value)
        
        # 转换为张量
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # PPO epochs
        for epoch in range(self.n_epochs):
            # 随机打乱
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                action_probs, state_values = self.network(batch_states)
                dist = Categorical(action_probs)
                
                # 新的 log 概率
                new_log_probs = dist.log_prob(batch_actions)
                
                # 重要性权重
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO-Clip 目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                # 熵
                entropy = dist.entropy().mean()
                
                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
def train_ppo(env_name='CartPole-v1', n_episodes=500, update_every=2048):
    """训练 PPO 智能体"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim)
    
    episode_rewards = []
    episode_reward = 0
    state, _ = env.reset()
    
    for step in range(1, n_episodes * 500 + 1):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_transition(state, action, log_prob, reward, done, value)
        episode_reward += reward
        
        state = next_state
        
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            state, _ = env.reset()
        
        # 定期更新
        if step % update_every == 0:
            agent.update()
            
            if episode_rewards:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Step {step}, Episodes: {len(episode_rewards)}, Avg Reward: {avg_reward:.2f}")
    
    env.close()
    return episode_rewards, agent

# 训练
rewards_ppo, agent_ppo = train_ppo()

# 可视化对比
plt.figure(figsize=(12, 5))
plt.plot(rewards_a2c, alpha=0.3, label='A2C')
plt.plot(rewards_ppo, alpha=0.3, label='PPO')
plt.plot(np.convolve(rewards_a2c, np.ones(50)/50, mode='valid'), label='A2C (smoothed)', linewidth=2)
plt.plot(np.convolve(rewards_ppo, np.ones(50)/50, mode='valid'), label='PPO (smoothed)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('A2C vs PPO on CartPole-v1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 4. SAC：软演员-评论家

### 4.1 最大熵强化学习

**传统目标**：最大化累积奖励

$$
J(\pi) = E_\pi \left[ \sum_t \gamma^t r_t \right]
$$

**最大熵目标**（Maximum Entropy RL）：同时最大化奖励和策略熵

$$
J(\pi) = E_\pi \left[ \sum_t \gamma^t (r_t + \alpha H(\pi(\cdot|s_t))) \right]
$$

其中 $H(\pi) = -E[\log \pi(a|s)]$ 是策略熵。

**优势**：
- 鼓励探索
- 提升鲁棒性
- 自动平衡探索-利用

### 4.2 SAC 算法（Soft Actor-Critic）

**核心组件**（Haarnoja et al., 2018）：
1. **Actor**：高斯策略 $\pi_\theta(a|s)$
2. **双 Critic**：两个 Q 网络 $Q_{\phi_1}, Q_{\phi_2}$（减少过估计）
3. **温度参数** $\alpha$：自动调节探索强度

**目标函数**：

**Critic 更新**：
$$
L_Q(\phi) = E_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q_\phi(s,a) - y)^2 \right]
$$

其中：
$$
y = r + \gamma (Q_{\text{target}}(s', a') - \alpha \log \pi_\theta(a'|s')), \quad a' \sim \pi_\theta(\cdot|s')
$$

**Actor 更新**：
$$
L_\pi(\theta) = E_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ \alpha \log \pi_\theta(a|s) - Q_\phi(s,a) \right]
$$

**温度参数自动调节**：
$$
L_\alpha = E_{a \sim \pi_\theta} \left[ -\alpha (\log \pi_\theta(a|s) + \bar{H}) \right]
$$

其中 $\bar{H}$ 是目标熵（通常设为 $-\dim(\mathcal{A})$）。

### 4.3 SAC 实现要点

```python
class SACActor(nn.Module):
    """SAC Actor（高斯策略）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std
    
    def sample(self, state):
        """采样动作（使用重参数化技巧）"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 重参数化
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # rsample() 支持梯度回传
        
        # Tanh 压缩到 [-1, 1]
        action = torch.tanh(x_t)
        
        # 计算 log 概率（修正 Tanh 变换）
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class SACCritic(nn.Module):
    """SAC Critic（Q 网络）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACCritic, self).__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

# SAC 完整实现略（代码较长，核心思想如上）
```

## 5. 算法对比与选择指南

### 5.1 性能对比

| 算法 | 样本效率 | 稳定性 | 离散动作 | 连续动作 | 实现难度 |
|------|---------|--------|---------|---------|---------|
| **A2C** | 中 | 中 | ✅ | ✅ | 简单 |
| **A3C** | 中 | 中 | ✅ | ✅ | 中等 |
| **PPO** | 中-高 | 高 | ✅ | ✅ | 中等 |
| **SAC** | 高 | 高 | ❌ | ✅ | 复杂 |

### 5.2 应用场景建议

**选择 A2C/A3C**：
- 简单环境（如 Atari、CartPole）
- 需要快速原型验证
- 计算资源有限

**选择 PPO**：
- 最推荐的通用算法
- 需要稳定训练
- 连续或离散动作空间
- OpenAI/DeepMind 工业标准

**选择 SAC**：
- 连续控制任务（机器人、自动驾驶）
- 需要最大样本效率
- 对探索要求高
- 有充足计算资源

## 6. 实战技巧

### 6.1 超参数调优

```python
# PPO 推荐参数
{
    'lr': 3e-4,
    'gamma': 0.99,
    'epsilon': 0.2,       # clip 范围
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'n_epochs': 10,       # 每次更新的 epoch 数
    'batch_size': 64,
    'gae_lambda': 0.95
}

# SAC 推荐参数
{
    'lr_actor': 3e-4,
    'lr_critic': 3e-4,
    'lr_alpha': 3e-4,
    'gamma': 0.99,
    'tau': 0.005,         # 软更新系数
    'buffer_size': 1e6,
    'batch_size': 256,
    'auto_entropy': True  # 自动调节温度
}
```

### 6.2 调试建议

```python
# 监控关键指标
metrics = {
    'episode_reward': [],
    'actor_loss': [],
    'critic_loss': [],
    'entropy': [],          # 策略熵（探索程度）
    'value_estimate': [],   # 值函数估计
    'advantage_mean': [],   # 优势函数均值
    'grad_norm': []         # 梯度范数
}

# 可视化学习曲线
def plot_training_curves(metrics):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(metrics['episode_reward'])
    axes[0, 0].set_title('Episode Reward')
    
    axes[0, 1].plot(metrics['actor_loss'])
    axes[0, 1].set_title('Actor Loss')
    
    axes[0, 2].plot(metrics['critic_loss'])
    axes[0, 2].set_title('Critic Loss')
    
    axes[1, 0].plot(metrics['entropy'])
    axes[1, 0].set_title('Policy Entropy')
    
    axes[1, 1].plot(metrics['value_estimate'])
    axes[1, 1].set_title('Value Estimate')
    
    axes[1, 2].plot(metrics['grad_norm'])
    axes[1, 2].set_title('Gradient Norm')
    
    plt.tight_layout()
    plt.show()
```

## 7. 总结

### 7.1 核心要点回顾

1. **Actor-Critic 框架**：
   - Actor 优化策略
   - Critic 评估动作
   - 低方差、快速学习

2. **A2C/A3C**：
   - 基础 Actor-Critic
   - 并行采样提升效率

3. **PPO**：
   - Clip 限制策略更新
   - 工业界首选
   - 稳定且易调

4. **SAC**：
   - 最大熵框架
   - 连续控制最佳
   - 样本效率高

### 7.2 进阶学习路径

1. **深入 PPO 变体**：
   - PPO-Penalty
   - PPO with LSTM
   - Multi-agent PPO

2. **探索其他算法**：
   - TRPO（Trust Region Policy Optimization）
   - TD3（Twin Delayed DDPG）
   - DreamerV3（Model-based RL）

3. **实战项目**：
   - MuJoCo 机器人控制
   - Atari 游戏
   - 自定义环境

## 参考资源

1. **经典论文**：
   - Mnih et al. (2016): *Asynchronous Methods for Deep RL* (A3C)
   - Schulman et al. (2017): *Proximal Policy Optimization Algorithms* (PPO)
   - Haarnoja et al. (2018): *Soft Actor-Critic* (SAC)

2. **开源实现**：
   - [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
   - [CleanRL](https://github.com/vwxyzjn/cleanrl)
   - [RLlib](https://docs.ray.io/en/latest/rllib/)

3. **推荐课程**：
   - OpenAI Spinning Up in Deep RL
   - CS 285: Deep Reinforcement Learning (UC Berkeley)

---

Actor-Critic 方法将策略优化与值函数学习完美结合，成为现代强化学习的基石。掌握 PPO 和 SAC，你将能够解决大部分实际强化学习问题。记住：调试强化学习需要耐心，多实验、多可视化、多对比！
