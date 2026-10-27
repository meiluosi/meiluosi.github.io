<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1988 · 相关认知能力为什么成组出现？

[学习效率与泛化路线](../routes/learning-efficiency.md) · [符号、逻辑、搜索与组合路线](../routes/symbolic-logic.md) · [连接主义、统计学习与表征路线](../routes/connectionist.md) · [文献夹](../literature.md)

**原始问题：** 理解某种关系为什么通常伴随理解相关的角色交换，而不是互不相关的能力清单？

**方法与机制：** 从系统性、组合性与推理一致性论证：表征需要组成结构，处理过程需要对该结构敏感；系统性论证不依赖无界生产性的理想化。

**成立条件：** 作者把相关能力成组出现视作需要解释的经验前提；从语言到思想还使用表达思想的桥接前提；关于非语言动物的论证包含待经验检验的主张。

## 主张与原文定位

- **原文主张**：§3.2将系统性表述为某些理解或表达能力之间的内在联系，并明确它不需要无界能力的理想化。
  - 依据：[Connectionism and cognitive architecture: A critical analysis](https://neural-mechanics.baulab.info/papers/fodor-1988-connectionism-critique.pdf)；印刷36–40页，§3.2。独立验证：未做独立复现。

- **原文主张**：§3.4和§3.5承认可以构造表现系统性的网络；作者要求说明为什么相关能力由机制成组产生，而非仅额外规定其存在。
  - 依据：[Connectionism and cognitive architecture: A critical analysis](https://neural-mechanics.baulab.info/papers/fodor-1988-connectionism-critique.pdf)；印刷47–50页，尤其50页。独立验证：未做独立复现。

- **本站分析**：把SCAN或MLC某一切分得分直接视为整场架构争论的裁决，会跨过能力定义、训练机制及泛化范围的对齐步骤。
  - 依据：[Connectionism and cognitive architecture: A critical analysis](https://neural-mechanics.baulab.info/papers/fodor-1988-connectionism-critique.pdf)；Fodor与Pylyshyn §3.2–3.5；比较依据另见关联来源SCAN PDF5–6及MLC 119页Discussion；本站分析。独立验证：未做独立复现。

- **原文主张**：作者将节点的说法扩展到聚合活动模式，并区分认知架构与神经实现；不能把其论证简单还原为只批评局部单元表征。
  - 依据：[Connectionism and cognitive architecture: A critical analysis](https://neural-mechanics.baulab.info/papers/fodor-1988-connectionism-critique.pdf)；印刷5页脚注1、10–11页层次区分、15页脚注10。独立验证：未做独立复现。

## 阅读记录

### Connectionism and cognitive architecture: A critical analysis

Jerry A. Fodor；Zenon W. Pylyshyn，1988。Cognition 28(1–2):3–71；扫描期刊版

- 实读：PDF1–49／印刷3–51页：§1、§2、完整§3.1–3.5与§4开头；扫描文字识别辅助阅读，图形及少量识别不清符号不据OCR重建
- 未覆盖：PDF50–69／印刷52–71页尚未完整读；不能称全文完成
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Connectionism and cognitive architecture: A critical analysis](https://neural-mechanics.baulab.info/papers/fodor-1988-connectionism-critique.pdf)；DOI：10.1016/0010-0277(88)90031-5
- SHA-256：`fcee0873886d02d961fa798b418cefcf22498f1d0997f0034b15f2510bd9bdf5`

### Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks

Brenden M. Lake；Marco Baroni，2018。Generalization without systematicity；Lake & Baroni；ICML/PMLR80

- 实读：§2及§4实验，PDF2、4–7；未逐页读全部附录；本批复核PDF2、5–6：任务定义、隔离原语实验及与Fodor思想实验的联系
- 未覆盖：第二轮所列未覆盖范围仍保留；本批未补读全部Methods、附录、代码，也未复现实验
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf)
- SHA-256：`7961052c8c0a113ac7b7c8ba992f58eb88b5665d777b1d695df3eb23b8c56350`
- 阅读记录延续自第二轮；本批复核范围另列。

### Human-like systematic generalization through a meta-learning neural network

Brenden M. Lake；Marco Baroni，2023。Human-like systematic generalization through a meta-learning neural network；Lake & Baroni；Nature623,115–121

- 实读：正文PDF2–6；未全面核对Methods与补充材料；本批复核PDF5／119页表2与Discussion：直接讨论Fodor与Pylyshyn，并保留分布外及新符号限制
- 未覆盖：第二轮所列未覆盖范围仍保留；本批未补读全部Methods、附录、代码，也未复现实验
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Human-like systematic generalization through a meta-learning neural network](https://www.nature.com/articles/s41586-023-06668-3.pdf)；DOI：10.1038/s41586-023-06668-3
- SHA-256：`0f7720ff2037c704806781a311172b9f55e4b2edc8ee82ff1a2336f83ac09b50`
- 阅读记录延续自第二轮；本批复核范围另列。

## 边界与下一问

这不是证明任何神经网络都无法完成组合任务。作者承认网络可以表现系统性，争点是架构如何解释并产生这种规律；人类系统性的范围也需任务与表现约束。

训练诱导的规则能否在新角色、新长度与新分布下持续工作？这些检验分别触及原论证哪一前提？
