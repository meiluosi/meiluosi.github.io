<!-- 由 research.json 生成；修改数据后重生成。C批已验收；不代表完整路线史。 -->

# 2017 · 必要部件为何仍不是计算解释？

[神经科学路线](../routes/neuroscience.md) · [文献夹](../literature.md)

**原始问题：** 熟悉的神经分析工具能否恢复已知系统的机制？

**方法与机制：** 把MOS6502视为模型系统，进行单晶体管损伤等分析，比较结果与真实计算结构。

**成立条件：** 只能直接检验所选处理器、任务和分析方式；作者承认脑与处理器差异。

## 主张与原文定位

- **原文主张**：只对某一游戏必需的晶体管并非该游戏的专属语义单元；作者用损伤案例说明功能定位的歧义。
  - 依据：[Could a Neuroscientist Understand a Microprocessor?](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005268)；Lesion a single transistor at a time；Fig.4。独立验证：未做独立复现。

- **原文主张**：讨论指出结果针对方法的朴素使用，建议更能隔离单个计算的任务及层级模型。
  - 依据：[Could a Neuroscientist Understand a Microprocessor?](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005268)；Discussion开篇与结尾。独立验证：未做独立复现。

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
| 证据性质 | 人工系统方法检验 |
| 分析单位 | 处理器及三种游戏任务 |
| 观察／论证依据 | 部件损伤与已知功能分解比较 |
| 需要区分的解释 | 表现必要性与可迁移的计算角色。 |

## 阅读记录

### Could a Neuroscientist Understand a Microprocessor?

Eric Jonas；Konrad P. Kording，2017。PLoS Computational Biology 13:e1005268；期刊开放正文。

- 实读：Lesion a single transistor at a time与Fig.4图注；Discussion前三段及结尾方法建议。
- 未覆盖：其他分析的完整方法、数据和代码未执行。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Could a Neuroscientist Understand a Microprocessor?](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005268)
- SHA-256：`b47567c5daaf304fcad61221c739c84c6f5fd3baddb832edc681b272f4afa6ac`

## 边界与下一问

不能从处理器反例推出所有神经实验无效；也不能忽略已经知道的生物约束。

怎样设计有机制真值、能区分替代算法的分析基准？
