<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2011 · 少量经验如何支持丰富的结构推断？

[概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [学习效率与泛化路线](../routes/learning-efficiency.md) · [文献夹](../literature.md)

**原始问题：** 从稀疏数据作出广泛概括时，所需的先验和假设空间从何而来？

**方法与机制：** 把树、图、语法等结构放入概率模型；在分层模型中联合学习具体结构及约束它的抽象形式，跨实例汇集证据。

**成立条件：** 结构形式、元语法及先验族仍需规定；归纳能力取决于真实规律与假设空间的关系。此文是综述，实例来自其引用研究。

## 主张与原文定位

- **原文主张**：文章将概率推断与结构化表示结合，图2用上层结构形式约束下层图，并联合推断形式与具体结构。
  - 依据：[How to Grow a Mind: Statistics, Structure, and Abstraction](https://web.stanford.edu/~ngoodman/papers/tkgg-science11-reprint.pdf)；1280–1282页及图2。独立验证：未做独立复现。

- **原文主张**：图3展示分层图模式如何与具体网络从同一数据学习；高层共享约束可汇集低层证据，但模型仍有规定的表示及先验族。
  - 依据：[How to Grow a Mind: Statistics, Structure, and Abstraction](https://web.stanford.edu/~ngoodman/papers/tkgg-science11-reprint.pdf)；1283–1284页，图3及blessing of abstraction段。独立验证：未做独立复现。

- **原文主张**：开放问题包括初始概念、灵活程序表示如何约束学习、近似推断效率及神经实现；计算层解释不等于实现层已解决。
  - 依据：[How to Grow a Mind: Statistics, Structure, and Abstraction](https://web.stanford.edu/~ngoodman/papers/tkgg-science11-reprint.pdf)；1284–1285页Open Questions。独立验证：未做独立复现。

## 阅读记录

### How to Grow a Mind: Statistics, Structure, and Abstraction

Joshua B. Tenenbaum；Charles Kemp；Thomas L. Griffiths；Noah D. Goodman，2011。Science 331:1279–1285，2011-03-11；8页作者公开重印，PDF首页为下载封面

- 实读：正文1279–1285页及参考文献；1282–1283页图2–3已视觉核对。
- 未覆盖：综述所引用的各项原始实验、在线补充材料和程序未独立审查；不把综述复述计为原研究独立复现。
- 状态：full-text；核对日期：2026-10-26
- 来源：[How to Grow a Mind: Statistics, Structure, and Abstraction](https://web.stanford.edu/~ngoodman/papers/tkgg-science11-reprint.pdf)；DOI：10.1126/science.1192788
- SHA-256：`d42537f90a75727e8e0108ad85dd33f7e50cbb3276cb5feff1f2667b9bda3a70`

## 边界与下一问

分层学习不会消除所有先验；计算成本、初始概念与神经实现仍是原文开放问题。对连接主义的概括属于作者立场，不据此忽略已有分布式结构表示。

在相近经验与计算条件下，结构先验何时帮助迁移，何时造成系统性误判？
