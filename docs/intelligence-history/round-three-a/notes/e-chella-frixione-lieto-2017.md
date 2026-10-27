<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2017 · 共同架构怎样协调异质知识？

[认知架构与多机制协作路线](../routes/cognitive-architectures.md) · [文献夹](../literature.md)

**原始问题：** 增加知识量以后，原型、实例和规则之间的冲突如何处理？

**方法与机制：** 作者建议以概念空间连接不同表征，并显式设计分类过程之间的协调。

**成立条件：** 这是采用异质表征立场的研究建议；架构批评限定于文中讨论的当时方案。

## 主张与原文定位

- **原文主张**：作者区分知识规模不足与知识类型单一，认为扩充大型知识库不足以单独解决后者。
  - 依据：[Representational Issues in the Debate on the Standard Model of the Mind](https://cdn.aaai.org/ocs/15990/15990-69900-1-PB.pdf)；302页及306页 Extended Declarative Memories。独立验证：未做独立复现。

- **原文主张**：对ACT-R的批评侧重原型与实例推理的联合协调和自主选择；作者同时承认所需组件可能具备，尚未充分利用。
  - 依据：[Representational Issues in the Debate on the Standard Model of the Mind](https://cdn.aaai.org/ocs/15990/15990-69900-1-PB.pdf)；305页 ACT-R，末段保留意见。独立验证：未做独立复现。

## 阅读记录

### Representational Issues in the Debate on the Standard Model of the Mind

Antonio Chella；Marcello Frixione；Antonio Lieto，2017。AAAI Technical Report FS-17-05，302–307页；2018年的arXiv上传与2020年重印不是原始发表年

- 实读：正文302–306页及307页参考文献；核对脚注1、2、4、5及ACT-R批评中的保留意见
- 未覆盖：所引DUAL PECCS、分类实验与神经研究未逐篇核验；未运行架构或验证作者对当时版本的全部功能判断
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Representational Issues in the Debate on the Standard Model of the Mind](https://cdn.aaai.org/ocs/15990/15990-69900-1-PB.pdf)
- SHA-256：`029b821305ba4a35cc82e5cef2c3256a49e052d645f9525598141ff61473c47b`

## 边界与下一问

扩充知识库不自动解决异质推理；概念空间的规模仍有缺口。文章未证明唯一必要架构，也未否定ACT-R所有任务结果。

保持知识内容不变，改变冲突协调机制，能否预测规则、原型和旧实例相冲突时的选择？
