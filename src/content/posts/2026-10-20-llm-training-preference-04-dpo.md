---
title: "DPO 为什么能直接使用偏好对：目标、参考策略与 beta 的作用"
published: 2026-10-20T10:03:00+08:00
description: "大模型训练研究笔记第四卷之四：从 KL 约束的奖励最大化推出偏好对上的闭式目标，手算两词表上的损失与梯度，说明参考策略、序列概率与 beta 各自控制什么，并列出已知失效模式。"
format: technical
tags:
  - 大模型训练研究笔记
  - 偏好学习
  - 直接偏好优化
category: 大模型训练研究笔记
series: "大模型训练研究笔记 · 偏好卷"
seriesOrder: 4
lang: zh
---

上一篇我们有一批[偏好对](/posts/2026-10-20-llm-training-preference-02-pairs/)和[奖励模型的分数](/posts/2026-10-20-llm-training-preference-03-reward-model/)。现在的问题是：能不能跳过"训练奖励模型，再跑 PPO"这两步，直接在这批偏好对上优化？

DPO 的回答是能。但"能"这句话本身不是答案。真正需要弄清的是：**被跳过的那两步去哪了，为什么它们可以变成一个损失函数里的几项。**

本篇从目标函数出发推导，手算一个两词表上的 DPO 损失，并说明 beta 控制什么、参考策略为什么必须存在、为什么公式作用在序列概率上。文中数字均为自写教学算例，不是任何模型的训练成绩。

## 从三段式流水线到一个闭式目标

InstructGPT 把对齐分成监督微调、奖励建模、强化学习三个阶段；第三阶段在强化学习阶段用 PPO 优化一个带 KL 约束的奖励目标。[InstructGPT，第 3 节](https://arxiv.org/html/2203.02155v1#S3)

这个目标通常写成：

$$
\max_{\pi_\theta}\;\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(y\mid x)}\big[r_\phi(x,y)\big]-\beta\,\mathrm{KL}\big[\pi_\theta(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)\big]
$$

其中 $\pi_{\mathrm{ref}}$ 是参考策略，通常就是微调后的模型 $\pi^{\mathrm{SFT}}$，$\beta$ 控制策略允许偏离参考策略的程度。DPO 论文把这个式子作为出发点，并指出它带来两个麻烦：奖励模型的拟合误差会传进策略更新，而 actor-critic 方法在实践中不稳定。[DPO，第 3 节与第 5 节](https://arxiv.org/html/2305.18290v3#S3)

关键的一步是：这个约束问题的最优解有闭式表达。

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\exp\big(r^*(x,y)/\beta\big)
$$

把两边取对数并整理，可以把奖励写成策略与参考策略之比：

$$
r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)
$$

这一步是整个 DPO 的支点。它说明：在最优策略处，奖励与"策略相对参考策略移动了多少"是同一件事。奖励模型不再是一个必须单独拟合的网络，而是一个可以写出来的函数。

## Bradley–Terry 把偏好变成概率

要谈"偏好"，先要有一个描述"人多大概率选哪一个"的模型。Bradley 与 Terry 提出的成对比较模型给出：

$$
P(y_1\succ y_2\mid x)=\sigma\big(r^*(x,y_1)-r^*(x,y_2)\big)
$$

其中 $\sigma(z)=1/(1+e^{-z})$。这个模型最早出现在 1952 年的《Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons》。[Bradley 与 Terry，Biometrika 39(3-4):324-345, 1952](https://doi.org/10.1093/biomet/39.3-4.324)

把上一节的奖励重参数化代进去，配分函数 $Z(x)$ 在相减时抵消：

$$
P(y_1\succ y_2\mid x)=\sigma\Big(\beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\mathrm{ref}}(y_1\mid x)}-\beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\mathrm{ref}}(y_2\mid x)}\Big)
$$

于是最大似然目标就是一个二分类的交叉熵：

$$
\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\big(\beta\,\Delta_\theta(y_w)-\beta\,\Delta_\theta(y_l)\big)\Big],\quad \Delta_\theta(y)=\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

这就是 DPO 的全部损失。没有采样，没有奖励模型，没有价值函数；训练时每个样本需要策略模型和参考模型各前向一次。[DPO，第 4 节与附录 A.2](https://arxiv.org/html/2305.18290v3#S4)

注意这里做了两件事：一是把奖励换成策略比值，二是把"生成再打分"换成"对已有的一对回答打分类分"。

## 参考策略不是可选项

$\pi_{\mathrm{ref}}$ 出现在分子的对数里，也出现在分母里。它不是初始化技巧，而是目标的组成部分。

去掉它，损失会退化成"让 chosen 的概率高过 rejected"，模型可以靠整体压低所有序列的概率来作弊。保留它之后，损失衡量的是**相对位移**：策略是否把概率从别处挪到了 chosen 上。

实现上，参考模型通常冻结。TRL 0.23.1 的 DPOTrainer 在未传入 `ref_model` 时会用同结构模型作为参考，也提供 `precompute_ref_log_probs` 选项把参考对数概率提前算好、训练时不再加载参考模型。[TRL 0.23.1 DPO Trainer](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)

参考策略选谁会影响结果的含义。用 SFT 模型作参考，意味着目标是"在不破坏 SFT 行为的前提下按偏好调整"；DPO 论文也提到，当没有现成的 SFT 模型时，可以退而用最大化 chosen 似然来构造 $\pi_{\mathrm{ref}}$，这会引入参考分布与真实参考分布之间的偏差。[DPO，第 4 节](https://arxiv.org/html/2305.18290v3#S4)

## 手算一个两词表的 DPO 损失

词表只有甲、乙，回复都是两个 token：chosen 是"甲乙"，rejected 是"乙甲"。给出一个提示 $x$，策略与参考策略的条件概率如下：

| 位置 | $\pi_\theta$ | $\pi_{\mathrm{ref}}$ |
| --- | --- | --- |
| 第一个 token | 甲 0.6，乙 0.4 | 甲 0.5，乙 0.5 |
| 甲 之后 | 甲 0.3，乙 0.7 | 甲 0.4，乙 0.6 |
| 乙 之后 | 甲 0.55，乙 0.45 | 甲 0.4，乙 0.6 |

于是两条序列的概率是：

$$
\pi_\theta(y_w)=0.6\times0.7=0.42,\quad \pi_\theta(y_l)=0.4\times0.55=0.22
$$

$$
\pi_{\mathrm{ref}}(y_w)=0.5\times0.6=0.30,\quad \pi_{\mathrm{ref}}(y_l)=0.5\times0.4=0.20
$$

比值分别是 $1.4$ 与 $1.1$，对数比值 $\Delta_w=\log 1.4\approx0.3365$、$\Delta_l=\log 1.1\approx0.0953$，两者之差：

$$
h=\Delta_w-\Delta_l=\log\frac{1.4}{1.1}=\log\frac{14}{11}\approx0.2412
$$

代入损失 $\ell=-\log\sigma(\beta h)$：

| $\beta$ | $\beta h$ | $\sigma(\beta h)$ | $\ell$ | $\partial\ell/\partial h$ |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.0241 | 0.5060 | 0.6812 | $-0.0494$ |
| 0.5 | 0.1206 | 0.5301 | 0.6347 | $-0.2349$ |
| 1.0 | 0.2412 | 0.5600 | 0.5798 | $-0.4400$ |

$\beta=1$ 这一行可以完全手算：$\sigma(\log(14/11))=14/25=0.56$，损失是 $-\log(14/25)\approx0.5798$。

梯度方向也可以验证。$\partial\ell/\partial h=-\beta\big(1-\sigma(\beta h)\big)$ 在任何 $\beta\gt0$ 时都为负，所以梯度下降总是把 $h$ 推大，即抬高 chosen 相对 rejected 的位置。在 $\beta=0.05,0.1,0.25,0.5,1,2$ 上逐一计算，符号一致，而绝对值从 $0.0248$ 单调增长到 $0.7634$。

<div data-code-sandbox data-title="两词表 DPO 损失与梯度手算验证" data-title-en="Two-Vocab DPO Loss & Gradient Verification">
<pre><code>import math

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

# 教学算例概率设置
pi_yw, pi_yl = 0.42, 0.22
ref_yw, ref_yl = 0.30, 0.20

h = math.log(pi_yw / ref_yw) - math.log(pi_yl / ref_yl)
print(f"对数比值差 h = log(14/11) = {h:.4f}")

print("\n不同 beta 下的 DPO 损失与梯度检验:")
for beta in [0.1, 0.5, 1.0]:
    sig = sigmoid(beta * h)
    loss = -math.log(sig)
    grad = -beta * (1.0 - sig)
    print(f"  beta={beta:.1f} | 隐式奖励差 beta*h={beta*h:.4f} | 损失={loss:.4f} | 梯度={grad:.4f}")
</code></pre>
</div>

## beta 到底控制什么

beta 的作用常被一句"KL 系数"带过。它同时出现在两处，需要分开看。

第一处是原始目标里的 KL 惩罚。给定奖励函数，最优策略满足：

$$
\pi^*(y\mid x)\propto\pi_{\mathrm{ref}}(y\mid x)\exp\big(r(y\mid x)/\beta\big)
$$

$\beta$ 越大，指数倾斜越弱，最优解越靠近参考策略。取两个回复、奖励差为 1、参考概率为 $(0.6,0.4)$，可以算出：

| $\beta$ | $\pi^*(y_w)$ | $\mathrm{KL}(\pi^*\|\pi_{\mathrm{ref}})$ |
| ---: | ---: | ---: |
| 0.1 | 0.99997 | 0.5105 |
| 1.0 | 0.8030 | 0.0945 |
| 10 | 0.6237 | 0.0012 |

$\beta$ 增大时最优解离参考策略越来越近。这正是"不许跑太远"的含义。

第二处是损失里的缩放。在策略状态固定时，$\partial\ell/\partial h=-\beta\big(1-\sigma(\beta h)\big)$ 随 $\beta$ 增大而增大：表格里从 $-0.0494$ 变到 $-0.4400$，差了近九倍。

把两处放在一起，beta 并不是"对齐强度"旋钮。它同时缩放了隐式奖励和 KL 惩罚，真正被设定的是"奖励尺度相对于允许偏离程度"的比例。TRL 0.23.1 的 `beta` 默认值是 0.1，文档把它的作用描述为"偏离参考模型的程度，高 $\beta$ 意味着更少偏离"。[TRL 0.23.1 DPO Trainer](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)

本站的[偏好对齐实验台](/lab/preference-alignment/)把 beta 与代理奖励偏差放在一起，可以直接看到代理奖励继续上升而真实效用越过拐点后回落。

<div data-reactive-prose="dpo"></div>

## 为什么作用在序列概率上

$\log\pi_\theta(y\mid x)$ 是整段回复各 token 对数概率之和：

$$
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{\lt t})
$$

偏好是一条关于整段回复的判断，不是关于某一个位置的判断。如果改成逐 token 比较，就等于要回答"哪个 token 承担了这次偏好"，而这个问题在偏好数据里没有答案。序列求和把责任均匀摊到各 token 上：上例两个 token，各承担 $\Delta$ 的一半。

这不是唯一选择。TRL 0.23.1 的损失列表里注明 IPO 变体对完成部分的 log 似然取平均，而标准 DPO 只做求和；两种归一化会改变长回复上的梯度尺度。[TRL 0.23.1 DPO Trainer](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)

## 已知失效模式

**两侧同时下降。** 损失只约束两个对数比值之差，绝对水平是自由的。把上例的策略概率整体乘以 0.9：

$$
\pi_\theta'(y_w)=0.378,\quad \pi_\theta'(y_l)=0.198
$$

比值变成 $1.26$ 与 $0.99$，$h$ 仍是 $\log(14/11)$，损失在 $\beta=0.1$ 下仍是 $0.6812$，一点没变。但两侧的绝对对数概率都下降了 $\log 0.9\approx-0.1054$。这种"chosen 的绝对概率也在下降"的现象后来被专门研究，并被命名为 likelihood displacement。[Unintentional Unalignment，arXiv:2410.08847v4，ICLR 2025](https://arxiv.org/abs/2410.08847)

**分布漂移。** 偏好数据是在参考策略上采样并标注的，训练推进后策略已经离开那片区域，此时数据上的标签不再可靠。DPO 论文自己指出，实际使用中人们更愿意复用公开的偏好数据集，而不是重新采样并收集人类偏好。[DPO，第 4 节](https://arxiv.org/html/2305.18290v3#S4)

**长度被利用。** 有专门工作分析 DPO 中的长度问题，把长度利用与分布外自举联系起来，并指出 GPT-4 评审存在众所周知的冗长偏好。[Disentangling Length from Quality in DPO，arXiv:2403.19159v2](https://arxiv.org/abs/2403.19159)

**偏好不等于正确。** Bradley–Terry 描述的是选择概率，不是事实正确率。上一篇说过的那条边界在这里依然成立。

## 假设与边界

这套推导依赖几个假设：偏好可以由 Bradley–Terry 形式刻画；奖励的期望与策略无关；最优解有闭式表达；参考策略与数据采集时的分布足够接近。任一条不成立，结论就要重新检查。

"闭式解"还意味着 DPO 优化的是一个代理目标，而不是原始 KL 约束问题本身。两者在最优解处一致，在远离最优解处未必。

## 一个对照设计

要区分"DPO 提高了 chosen 的绝对概率"和"DPO 只抬高了 chosen 相对 rejected 的位置"，可以做一个便宜的对照：训练后同时记录 $\log\pi_\theta(y_w)$、$\log\pi_\theta(y_l)$ 各自的绝对变化，而不只是它们的差。

如果差值在上升而两侧的绝对值同时下降，那么模型学到的是相对偏好排序，不是"更可能生成更好的回答"。这个区分只有分开记录才能看到，报告单一胜率会把它抹掉。

## 一手来源

- Rafailov 等，Direct Preference Optimization: Your Language Model is Secretly a Reward Model，arXiv:2305.18290v3（2024-07-29 修订，NeurIPS 2023），方法与附录推导：https://arxiv.org/html/2305.18290v3
- Ouyang 等，Training language models to follow instructions with human feedback，arXiv:2203.02155v1（InstructGPT），RLHF 三阶段定义：https://arxiv.org/html/2203.02155v1#S3
- Bradley 与 Terry，Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons，Biometrika 39(3-4):324-345，1952，DOI 10.1093/biomet/39.3-4.324：https://doi.org/10.1093/biomet/39.3-4.324
- TRL 0.23.1 DPO Trainer 文档（`beta` 默认值、损失类型、参考模型与预计算选项）：https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer

## 两个检查理解的问题

第一，把上面算例里的 $\pi_\theta(y_w)$ 和 $\pi_\theta(y_l)$ 同时乘以 0.9，损失会变吗？两侧的绝对概率呢？为什么这说明"损失下降"不能直接读作"更可能生成好回答"？

第二，若把参考策略换成 SFT 之前的基础预训练模型，$\beta$ 的含义会变吗？你会记录哪些量，才能判断新参考策略是不是把 KL 约束挪到了错误的位置？

下一篇讨论[离线偏好优化与在线采样有什么区别](/posts/2026-10-20-llm-training-preference-05-offline-vs-online/)，看这个闭式目标在固定答案池之外还能走多远。

本篇完成的是目标推导与极小算例核对，没有训练任何模型，也没有在中文任务上比较 DPO 与 PPO 的效果。表中的概率与损失是按公式手算的教学设定；[前向推演白盒实验室](/lab/transformer-forward/)可以配合阅读序列概率的形成过程。
