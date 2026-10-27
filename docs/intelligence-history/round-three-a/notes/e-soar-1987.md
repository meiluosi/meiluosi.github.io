<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1987 · 遇到僵局时如何求解并记住经验？

[认知架构与多机制协作路线](../routes/cognitive-architectures.md) · [文献夹](../literature.md)

**原始问题：** 如何让决策、子目标和学习在同一架构内持续协作？

**方法与机制：** 知识产生偏好，决策僵局触发子目标；chunking追踪结果所依赖的条件，生成可复用的产生式。

**成立条件：** 原文描述SOAR4：长期知识统一存入产生式；任务表示与初始知识由具体模型提供。

## 主张与原文定位

- **原文主张**：决策知识不足或冲突产生僵局，架构据此生成子目标；chunking通过依赖分析保存解决子目标所需的条件与结果。
  - 依据：[SOAR: An Architecture for General Intelligence](https://www.researchgate.net/publication/346700056_SOAR_An_architecture_for_general_intelligence)；§1.4–1.5；§2.5开头；§2.6.1。独立验证：未做独立复现。

- **原文主张**：规则泛化依赖变量化、保留相关条件和任务表示的分解方式；逻辑上足够的表示也可能妨碍忽略无关细节。
  - 依据：[SOAR: An Architecture for General Intelligence](https://www.researchgate.net/publication/346700056_SOAR_An_architecture_for_general_intelligence)；§3.3.3，55–57页。独立验证：未做独立复现。

- **原文主张**：本文的统一产生式长期记忆和统一chunking学习是具体架构假说；作者不保证当时架构已完整或稳定。
  - 依据：[SOAR: An Architecture for General Intelligence](https://www.researchgate.net/publication/346700056_SOAR_An_architecture_for_general_intelligence)；§1.3；§4，58–59页。独立验证：未做独立复现。

## 阅读记录

### SOAR: An Architecture for General Intelligence

John E. Laird；Allen Newell；Paul S. Rosenbloom，1987。Artificial Intelligence 33(1):1–64，1987；实读为John E. Laird于2024-10-22上传的期刊扫描件网页转录，正文描述SOAR4

- 实读：引言2–4页、§1.1及§1.3–1.7；§2.5开头29–30页与§2.6–2.6.1的35–39页相关段落；§3.3.2负迁移段、§3.3.3的55–57页表示与泛化论证；§4结论58–59页。经网页正文转录阅读，未逐页视觉核验
- 未覆盖：其余完整算法、示例轨迹、实验表格和图形未逐项核对；没有复现学习收益，不把1987架构当作后来的Soar版本
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[SOAR: An Architecture for General Intelligence](https://www.researchgate.net/publication/346700056_SOAR_An_architecture_for_general_intelligence)；DOI：10.1016/0004-3702(87)90050-6；[书目核对](https://www.sciencedirect.com/science/article/abs/pii/0004370287900506)
- SHA-256：无：未取得可校验的本地PDF；实际阅读载体与范围见本条来源记录。

## 边界与下一问

作者报告过度泛化可能导致负迁移，并承认外部知识获取等能力仍缺失；有逻辑表达能力也不保证学出足够一般的规则。

保持任务不变，只改变相关属性与无关细节的编码方式，迁移范围怎样改变？
