<!-- 由 research.json 生成；修改数据后重生成。C批已验收；不代表完整路线史。 -->

# 2007 · 学习者怎样挑选值得探索的经验？

[发展与开放式学习路线](../routes/developmental-openended.md) · [文献夹](../literature.md)

**原始问题：** 持续误差大的经验是否总值得探索？

**方法与机制：** 按感觉运动区域追踪预测误差下降，把学习进展作为内部奖励。

**成立条件：** 依赖区域划分、平滑窗口、预测器与可行动空间。

## 主张与原文定位

- **原文主张**：学习进展由平滑误差变化计算；区域在划分时继承误差历史。
  - 依据：[Intrinsic Motivation Systems for Autonomous Mental Development](https://www.pyoudeyer.com/ims.pdf)；270–271页§III.E，公式1–2。独立验证：未做独立复现。

- **原文主张**：内部奖励接入行动选择；原文将其表述为可用强化学习处理的目标。
  - 依据：[Intrinsic Motivation Systems for Autonomous Mental Development](https://www.pyoudeyer.com/ims.pdf)；271页§III.F。独立验证：未做独立复现。

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
| 证据性质 | 算法设计与机器人研究（方法选读） |
| 分析单位 | 区域预测器与行动选择 |
| 观察／论证依据 | 学习进展公式及内部奖励接口 |
| 需要区分的解释 | 不可约噪声、暂时遗忘与可学结构。 |

## 阅读记录

### Intrinsic Motivation Systems for Autonomous Mental Development

Pierre-Yves Oudeyer；Frédéric Kaplan；Verena V. Hafner，2007。IEEE TEC 11(2):265–286；作者公开论文PDF。

- 实读：267页相关工作限制；270–271页区域专家与学习进展，公式1–2和图2扫描；281页图式讨论。
- 未覆盖：完整机器人结果、超参数敏感性及代码未重跑。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Intrinsic Motivation Systems for Autonomous Mental Development](https://www.pyoudeyer.com/ims.pdf)
- SHA-256：`58100a5bbae5f119778951724bae0f15e7167b7c3bfb5fd5041a864d03cffb53`

## 边界与下一问

学习进展指标仍是设计者设定；高噪声、表示失配与长期价值不能由局部下降自动解决。

在相同预算下，学习进展比误差、新颖性或外部奖励带来何种迁移？
