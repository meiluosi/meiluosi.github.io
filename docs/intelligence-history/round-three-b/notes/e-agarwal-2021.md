<!-- 由 research.json 生成；修改数据后重生成。B批已验收；不代表完整路线史。 -->

# 2021 · 有限重复次数下的评价不确定性

[行为表现与操作性路线](../routes/behavioral.md) · [文献夹](../literature.md)

**原始问题：** 少量随机运行得到的领先分数，能否稳定支持算法优劣判断？

**方法与机制：** 同时报告按任务分层重采样的区间、跨任务与运行的表现分布、稳健聚合；核对训练末端得分与训练期间最大得分等协议差别。

**成立条件：** 需声明任务集合、归一化参考、独立运行和统计目标；有限样本bootstrap也可能覆盖不足。

## 主张与原文定位

- **原文主张**：少量运行与随机条件使汇总分数具有统计不确定性；同一随机种子也未必控制GPU等全部非确定性。
  - 依据：[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)；§1–2，PDF 1–3及脚注3。独立验证：未做独立复现。

- **原文主张**：按任务分别重采样运行估计聚合统计量的区间；作者在少于足够运行时仍观察到覆盖不足，不能把bootstrap当保证。
  - 依据：[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)；§4.1和图6说明，PDF 5–6。独立验证：未做独立复现。

- **原文主张**：IQM汇总合并运行分数的中间50%；表现分布补充单一指标，并指出取训练中最大成绩与最终成绩会造成协议不可比。
  - 依据：[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)；PDF 5；§4.2–4.3，PDF 6–7。独立验证：未做独立复现。

## 评价视角

| 项目 | 条件与边界 |
| --- | --- |
| 评价对象／构念 | 固定任务集合上算法表现的统计估计 |
| 任务与环境 | RL基准；重点讨论Atari 100k |
| 样本／适用主体 | 每任务有限独立运行；不把随机种子数等同独立性保证 |
| 输出指标 | 归一化分数分布、区间、IQM等；聚合规则须明示 |
| 信度与不确定性 | 少量随机运行的不确定性是核心问题；bootstrap也有小样本覆盖限制。 |
| 效度与解释范围 | 可靠估计不等于构念有效；协议差异可改变比较 |
| 先验与经验 | 训练协议、任务标准化和挑选最佳运行会改变比较。 |
| 计算与时间预算 | 重复训练成本制约可估精度，需公开运行数量 |

## 阅读记录

### Deep Reinforcement Learning at the Edge of the Statistical Precipice

Rishabh Agarwal；Max Schwarzer；Pablo Samuel Castro；Aaron Courville；Marc G. Bellemare，2021。NeurIPS 2021主文PDF，17页；不是对rliable当前软件版本的评测。

- 实读：PDF 1–3、5–7：问题与形式化、协议变化、§4分层bootstrap、表现分布与IQM定义。
- 未覆盖：PDF 4的完整实验设定、后续跨基准分析与补充材料未完整审读；未重跑作者RL实验或验证覆盖率数值。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
- SHA-256：`7553e280d4a1560860bc5f6be2d70539fc5b510e0e3bb2d556824e51e9a80252`

## 边界与下一问

该文解决指定基准与协议中的统计报告问题，不证明任务集合代表通用智能。本站未复现作者对具体算法的排序结论。

任务采样的不确定性、运行的不确定性和构念遗漏，如何分别报告？
