<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2019 · 如何从像素学到足以在线规划的动态？

[概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [文献夹](../literature.md)

**原始问题：** 部分可观测环境中，怎样保持记忆并预测多种未来，用有限经验选择行动？

**方法与机制：** RSSM组合确定性记忆与随机状态；学习动态、观察和奖励模型，用交叉熵搜索在潜在空间规划，每次接收新观察后重新规划并继续采集数据。

**成立条件：** 连续行动、给定奖励与固定动作重复；六项模拟控制任务使用64×64图像，搜索依赖学到的模型及有限时域。

## 主张与原文定位

- **原文主张**：规划在潜在空间搜索行动序列；观察模型用于训练信号，规划无需解码图像，也没有策略或价值网络。
  - 依据：[Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)；§2，2–3页；§3式4，4页。独立验证：未做独立复现。

- **原文主张**：正文提出latent overshooting以训练多步一致性，但明确最终采用RSSM的智能体不需要它。
  - 依据：[Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)；§4，4–6页，特别是5页开头。独立验证：未做独立复现。

- **原文主张**：六任务结果比较纯确定性、纯随机模型及在线收集／随机收集；图中线为中位数、带为5–95百分位，汇总5个种子与10条轨迹。表1均值与曲线统计口径不同。
  - 依据：[Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)；§5，6页；7–8页图4–5与表1。独立验证：未做独立复现。

## 阅读记录

### Learning Latent Dynamics for Planning from Pixels

Danijar Hafner；Timothy Lillicrap；Ian Fischer；Ruben Villegas；David Ha；Honglak Lee；James Davidson，2019。ICML 2019，PMLR 97:2555–2565；本文件页码1–11。事件按2019会议版定位，预印本2018另记

- 实读：PDF1–8全部正文；4页图2与7页图4、表1已视觉核对。
- 未覆盖：9–11页参考文献未逐条阅读；独立补充PDF的附录、实现与训练实验未审查；未复现性能或多步目标证明。
- 状态：selected-sections；核对日期：2026-10-26
- 来源：[Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)
- SHA-256：`1c3aee51ab99649e970815a7f8bccae8c7711addbfc8d50c4853a0d2e742f26a`

## 边界与下一问

模型准确性和探索仍限制结果；多步预测正则不是最终RSSM的必要组成。数据效率比较采用特定任务与他文基线，不能推出所有任务或计算成本上的通用优势。

奖励、外观或动态变化后，模型的哪些部分可复用，哪些必须重新学习？
