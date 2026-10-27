# REINFORCE：直接调整动作概率

事件时间：1992（publication）。状态：指定主张已审校；阅读级别：指定片段。

[原文入口](https://gwern.net/doc/reinforcement-learning/model-free/1992-williams.pdf) · Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning；Ronald J. Williams；Machine Learning8,229–256

实际阅读：§3–4，PDF5–7；即时奖励情形；未审读延迟奖励扩展和证明附录

## 问题

怎样在没有可微环境模型时沿期望奖励梯度学习？

## 机制与前提

使用奖励减基线，乘以动作概率对参数的对数导数，形成随机更新。

核心定理使用平稳的输入与奖励分布；基线须满足给定参数和输入时与所抽样输出条件独立等条件。

## 主张与原文定位

- **e-williams-1992-c1 / 作者报告**：相同步长时平均更新等于期望奖励梯度乘步长；这是期望方向性质。
  - 原文定位：§3–4，印刷页233–235，定理1。本轮未独立复现。

## 边界与批评

不保证每次更新都改善，也不保证非凸目标全局最优；方差与探索仍需处理。

本站审校：基线不是可以任意依赖所选动作而仍保持无偏的数值。

## 教学例子与后续问题

教学构想（除[已运行的三个算例](../demos.md)外，不宣称已运行）：两个动作的随机奖励，枚举各动作概率与更新，验证期望梯度。

下一问：如何在降低估计方差的同时避免引入偏差？

## 版本与维护

- 阅读记录日期：2026-10-25。仅核对所列原文主张；未独立复现论文结果。
- 本机阅读文件 SHA-256：e1029df5459f4a83bf33a876df018ee4f7b628b210bb9d047a6d60ec13f7c838
- 第一轮对应事件：本轮新增。本轮记录不改变第一轮当时的阅读状态。
- 图表和笔记统一由 [research.json](../research.json) 生成。原PDF临时阅读，不随笔记重发。

[路线叙事](../cybernetics-rl.md) · [阅读工作台](../atlas.html#e-williams-1992)
