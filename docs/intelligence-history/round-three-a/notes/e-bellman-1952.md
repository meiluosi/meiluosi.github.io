<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1952 · 如何把行动序列优化变成递推问题？

[控制论、强化学习与主动推断路线](../routes/cybernetics-rl.md) · [概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [文献夹](../literature.md)

**原始问题：** 每步操作的结果会影响后续决策，怎样求整个序列的最优收益或代价？

**方法与机制：** 用当前状态、当前收益与下一状态的续接价值构造取最大或最小的函数方程；引言用搜盒取球和资源分配说明。

**成立条件：** 定理1要求状态变换留在定义域，收益有界、续接系数绝对值统一小于1，且原文还列出非负条件；不能删掉条件只剩“最优”。

## 主张与原文定位

- **原文主张**：引言把许多最优操作序列问题化为选择最优第一步，并给出 f(p)=max/min T_k(f) 的形式。
  - 依据：[On the Theory of Dynamic Programming](http://www.derongliu.org/adp/adp-cdrom/refs/bellman19520716.pdf)；印刷716页，§1式1.1–1.2。独立验证：未做独立复现。

- **原文主张**：定理1在所列有界性、续接系数和定义域条件下声称有唯一有界解；本文说明证明使用逐次逼近，详细证明另文给出。
  - 依据：[On the Theory of Dynamic Programming](http://www.derongliu.org/adp/adp-cdrom/refs/bellman19520716.pdf)；印刷717页定理1、718页说明、719页末注。独立验证：未做独立复现。

## 阅读记录

### On the Theory of Dynamic Programming

Richard Bellman，1952。PNAS 38(8):716–719；1952-06-05为通讯日期，非发表月日

- 实读：四页文本已读取；视觉核对PDF1–2／印刷716–717页的引言和定理1
- 未覆盖：定理2–7公式未逐项视觉核验；详细证明原文另指后续出版，未复核；1957专著未读
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[On the Theory of Dynamic Programming](http://www.derongliu.org/adp/adp-cdrom/refs/bellman19520716.pdf)；DOI：10.1073/pnas.38.8.716
- SHA-256：`58ccf880b967862f7b2c909844b71a0a195cef6306429b637955f67fefc3b204`

## 边界与下一问

论文主要给出方程与定理陈述，详细证明另行发表；已知算子下求解价值与从未知环境经验学习是不同问题。1957专著仍未读。

哪些环境信息必须由模型提供，哪些可以由样本估计？共享参数后，原方程的性质还剩哪些？
