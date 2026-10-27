<!-- 由 research.json 生成；修改数据后重生成。E批已验收；不代表完整路线史。 -->

# 1964 · 从短描述到归纳预测先验

[信息、压缩、复杂系统与物理路线](../routes/information-physics.md) · [文献夹](../literature.md)

**原始问题：** 如何从有限序列为后续事件分配概率？

**方法与机制：** 用描述或生成序列的程序来组织归纳模型。

**成立条件：** 依赖参考机器／描述语言；理想化归纳与有限计算预算分开。

## 主张与原文定位

- **原文主张**：Part I提出多种模型，以序列描述的长度和数量关联其先验倾向。
  - 依据：[A Formal Theory of Inductive Inference. Part I](https://www.raysolomonoff.com/publications/1964pt1.pdf)；PDF 1摘要；3–4导论。独立验证：未做独立复现。

- **原文主张**：作者明确提示§3.1–4的若干论证是启发式，尚无正式证明，不保证整个思路正确。
  - 依据：[A Formal Theory of Inductive Inference. Part I](https://www.raysolomonoff.com/publications/1964pt1.pdf)；PDF 4，关于heuristic reasoning的段落。独立验证：未做独立复现。

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
| 证据性质 | 理想化形式模型与启发式论证 |
| 分析单位 | 符号序列及程序描述 |
| 观察／论证依据 | Part I导论选读 |
| 需要区分的解释 | 压缩先验与可计算学习程序 |

## 阅读记录

### A Formal Theory of Inductive Inference. Part I

Ray J. Solomonoff，1964。Information and Control 7:1–22，作者站原刊PDF。

- 实读：PDF 1摘要、3–4导论与启发式说明。
- 未覆盖：Part I其余模型与推导、Part II及后来的收敛结果未系统核验。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[A Formal Theory of Inductive Inference. Part I](https://www.raysolomonoff.com/publications/1964pt1.pdf)
- SHA-256：`bad9d0d3e7c889d35df94b2738f4c2e3a7648901bd6b78198897d4992f9b5e0c`

## 边界与下一问

本批不把后来严格证明倒填进1964；未提供本站可运行的通用归纳器。

加入时间、样本和模型选择成本后，还保留哪些归纳保证？
