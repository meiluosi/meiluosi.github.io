<!-- 由 research.json 生成；修改数据后重生成。C批已验收；不代表完整路线史。 -->

# 2011（方法始于2008） · 目标分数会不会遮住有用路径？

[发展与开放式学习路线](../routes/developmental-openended.md) · [文献夹](../literature.md)

**原始问题：** 优化眼前目标时为何会错过必要的中间行为？

**方法与机制：** 按行为差异奖励新颖性，保存探索出的行为作为进一步搜索的起点。

**成立条件：** 行为描述和距离仍由设计者选定；搜索时间尺度属于演化算法。

## 主张与原文定位

- **原文主张**：§3将新颖行为当作中间踏脚石，并区别于单一目标适应度。
  - 依据：[Abandoning Objectives: Evolution Through the Search for Novelty Alone](https://gwern.net/doc/reinforcement-learning/exploration/2011-lehman.pdf)；§3开篇，198页／PDF10。独立验证：未做独立复现。

- **原文主张**：作者明确反对把结果解读为新颖性总比目标搜索更好，并将失败条件列为后续问题。
  - 依据：[Abandoning Objectives: Evolution Through the Search for Novelty Alone](https://gwern.net/doc/reinforcement-learning/exploration/2011-lehman.pdf)；§9.1后段，214页／PDF26。独立验证：未做独立复现。

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
| 证据性质 | 演化算法与限域实验 |
| 分析单位 | 行为搜索空间及种群 |
| 观察／论证依据 | 方法论证与作者自限 |
| 需要区分的解释 | 欺骗性目标、描述偏置与探索预算。 |

## 阅读记录

### Abandoning Objectives: Evolution Through the Search for Novelty Alone

Joel Lehman；Kenneth O. Stanley，2011。Evolutionary Computation 19(2):189–223；2011综合论文，最初方法2008提出。

- 实读：§1片段、§2.1、§2.3开篇、§3开篇（198页）、§9.1后段（214页）与§9.3–9.4片段。
- 未覆盖：新颖性距离公式细节、全部迷宫／双足实验和统计未复算。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Abandoning Objectives: Evolution Through the Search for Novelty Alone](https://gwern.net/doc/reinforcement-learning/exploration/2011-lehman.pdf)
- SHA-256：`4dccb7eab84e459fc310026e6178bab3d77d67aebe941c4e883cbf8e5e33766f`

## 边界与下一问

未证明所有目标都妨碍进步，也未证明无限复杂性增长；本批未复现实验。

改变行为描述与预算之后优势是否仍存在？
