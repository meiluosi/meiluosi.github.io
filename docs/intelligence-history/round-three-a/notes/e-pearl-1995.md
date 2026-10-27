<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1995 · 观察分布何时足以回答干预问题？

[概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [文献夹](../literature.md)

**原始问题：** 没有随机实验时，什么额外假设使干预效果能从观察数据唯一确定？

**方法与机制：** 把因果图解释为自治结构方程；干预替换指定机制并保留其他机制。借助后门、前门条件及干预演算，把符合条件的目标转成观察分布表达式。

**成立条件：** 图及其遗漏的边是领域假设；原文可识别性定义要求与图相容的正概率观察分布。前门要求中介截断所有有向路径、X至中介无后门、中介至Y的后门由X阻断。

## 主张与原文定位

- **原文主张**：结构方程表达自治机制；原子干预通过替换被干预变量的方程，保留其他机制来定义干预分布。
  - 依据：[Causal Diagrams for Empirical Research](https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf)；§2.2，672–674页，式3–5与定义2。独立验证：未做独立复现。

- **原文主张**：后门与前门定理在各自图条件下给出效应识别公式；非识别可由观察分布相同而干预效果不同的两个模型说明。
  - 依据：[Causal Diagrams for Empirical Research](https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf)；§3，674–676页，定义3–4、定理1–2及式6、9。独立验证：未做独立复现。

- **原文主张**：三条演算规则分别允许在相应删边图的分离条件下增删观察、交换观察与行动、增删行动。
  - 依据：[Causal Diagrams for Empirical Research](https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf)；§4.3–4.4，677–679页，定理3与式10–21。独立验证：未做独立复现。

- **原文主张**：Cox与Wermuth肯定计算框架，同时质疑后门／前门假设在实践中如何可靠成立，并要求说明设置变量时其他变量怎样变化。
  - 依据：[Discussion of “Causal Diagrams for Empirical Research” by J. Pearl](https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf)；688–689页，完整评论。独立验证：未做独立复现。

- **原文主张**：Pearl在回应中区分假想原子干预与实际操作；实际操作若影响其他机制，相关副作用需要明确建模。
  - 依据：[Causal Diagrams for Empirical Research](https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf)；回应§4，705–706页。独立验证：未做独立复现。

## 阅读记录

### Causal Diagrams for Empirical Research

Judea Pearl，1995。Biometrika 82(4)，669–710页整组论文、评论及回应扫描；本文669–688页，评论单独署名

- 实读：本文669–679页§1–4，另读680页§5.1；677页图4及式10–12、676页前门条件与式9已视觉核对。回应705–706页§4实际／假想干预及§5开头已读。
- 未覆盖：§5其余部分、§6、附录证明及大部分作者回应未完整读。曾读36页扩展附录稿R218-B-L.pdf的前18页，但本节点定位统一采用期刊版，不混用两版页码。
- 状态：selected-sections；核对日期：2026-10-26
- 来源：[Causal Diagrams for Empirical Research](https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf)
- SHA-256：`0a9dd3218e541acbf26aeff5865e3c70cb662cd7de50e1234d016f1af5da8927`

### Discussion of “Causal Diagrams for Empirical Research” by J. Pearl

D. R. Cox；Nanny Wermuth，1995。与Pearl论文同一期的独立署名讨论，688–689页；文件指纹与整组期刊扫描相同

- 实读：完整阅读688页讨论标题起至689页Received May 1995止。
- 未覆盖：未据此声称整组其他评论全部已读；本文引述的流行病学和实验设计文献未独立核查。
- 状态：full-text；核对日期：2026-10-26
- 来源：[Discussion of “Causal Diagrams for Empirical Research” by J. Pearl](https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf)
- SHA-256：`0a9dd3218e541acbf26aeff5865e3c70cb662cd7de50e1234d016f1af5da8927`

## 边界与下一问

图不是仅凭拟合自动验证的真相。Cox与Wermuth质疑现实中满足条件的把握；Pearl回应要求明确实际操作的副作用。677–678页把演算完备性列为当时开放问题，本记录不外推为今日状态。

哪些实验或背景证据能检验模型中最影响结论的缺失边与机制不变性？
