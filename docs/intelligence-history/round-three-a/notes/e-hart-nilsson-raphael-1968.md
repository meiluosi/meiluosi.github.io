<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1968 · 启发式知识怎样帮助搜索而保留最优解？

[符号、逻辑、搜索与组合路线](../routes/symbolic-logic.md) · [文献夹](../literature.md)

**原始问题：** 怎样同时利用领域知识并约束搜索结果的路径代价？

**方法与机制：** 按已发现路径代价与剩余代价估计之和选择开放节点；找到更低代价时允许重开关闭节点。以剩余真实代价的下界约束启发式。

**成立条件：** 原文使用边代价有统一正下界的图。本批解释保守限定有限图、可达目标及非负启发式下界；不声称核验原文所有无限图细节。

## 主张与原文定位

- **原文主张**：原算法选择评价值最小的开放节点，并在发现更优路径时重开关闭节点；评价函数相加已知路径成本与剩余成本估计。
  - 依据：[A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf)；102页§II.A–C，算法步骤2–4和式2。独立验证：未做独立复现。

- **原文主张**：定理1将启发式不高估剩余最优代价作为可采纳性的关键条件；该主张仍依赖前文图与算法设置。
  - 依据：[A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf)；101页§I.B、103页定理1。独立验证：未做独立复现。

- **原文主张**：作者明确指出扩展节点最少不等于总资源消耗最少，启发式本身也有计算成本。
  - 依据：[A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf)；107页§IV.B。独立验证：未做独立复现。

## 阅读记录

### A Formal Basis for the Heuristic Determination of Minimum Cost Paths

Peter E. Hart；Nils J. Nilsson；Bertram Raphael，1968。IEEE Transactions on Systems Science and Cybernetics SSC-4(2):100–107，1968年7月；1967年11月24日为收稿日

- 实读：PDF2–4／101–103页：图定义、A*步骤、评价函数、引理1和定理1；PDF8／107页计算代价讨论；102–103页已视觉核对
- 未覆盖：原文104–106页扩展节点最优性论证、后续更正与争议未复核；未独立验证无限图终止论证
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf)；DOI：10.1109/TSSC.1968.300136
- SHA-256：`68905780affa771f22b0b6b83dde669bbce9246a62384d7d77ab9494f046e46e`

## 边界与下一问

保证路径最优、扩展节点最少、总计算时间最少是不同命题。本批不将原文扩展节点最优性论证视作已复核结论。

启发式的计算开销何时抵消减少搜索节点的收益？
