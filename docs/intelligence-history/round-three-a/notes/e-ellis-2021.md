<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2021 · 能否同时学习程序库和搜索程序的方法？

[符号、逻辑、搜索与组合路线](../routes/symbolic-logic.md) · [连接主义、统计学习与表征路线](../routes/connectionist.md) · [学习效率与泛化路线](../routes/learning-efficiency.md) · [文献夹](../literature.md)

**原始问题：** 预先设计专用语言限制了程序合成，怎样从已解任务中学出可复用抽象？

**方法与机制：** 清醒阶段搜索程序；抽象阶段重构已找到的程序形成库；梦阶段用已解任务和库中采样程序训练神经搜索策略。

**成立条件：** 输入仍需基本语言和具有共享结构、难度递进的任务语料；后验计算使用有限候选束等近似，不是从无先验开始。

## 主张与原文定位

- **原文主张**：§2明确要求基本语言足以表达任务，并要求容易与困难任务共享可复用抽象。
  - 依据：[DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf)；837页§2。独立验证：未做独立复现。

- **原文主张**：算法将程序搜索、抽象库学习和识别网络训练分成三阶段，并使用有限候选束近似程序空间的求和。
  - 依据：[DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf)；837–839页§2.1–2.4。独立验证：未做独立复现。

- **原文主张**：作者报告主评测中两类学习部件均有贡献；列表任务测试集和EC2批处理方式改变，不能直接与旧论文数字比较。
  - 依据：[DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf)；843–844页§5。独立验证：未做独立复现。

- **原文主张**：特殊递归语言实验禁用识别网络；作者仍将自动学习DSL的数据类型、惯用法和设计模式列为开放问题。
  - 依据：[DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf)；847页特殊实验与§6 Outlook。独立验证：未做独立复现。

- **原文主张**：图7生成文本面板报告预测似然；误差条为运行间标准差，EC2的运行次数存在领域差异。
  - 依据：[DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf)；845页图7及图说。独立验证：未做独立复现。

## 阅读记录

### DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning

Kevin Ellis；Catherine Wong；Maxwell Nye；Mathias Sablé-Meyer；Lucas Morales；Luke Hewitt；Luc Cary；Armando Solar-Lezama；Joshua B. Tenenbaum，2021。PLDI 2021正式论文，835–850页；2020预印本题名与组织不同，本批不混用

- 实读：PDF1–5／835–839页的问题、算法及近似目标；PDF9–10／843–844页评测设置与结果；PDF13–14／847–848页特殊实验、讨论和图说；PDF11／845页图7曲线、图例、误差条与图说已视觉核对，并读该页正文
- 未覆盖：PDF6–8重构细节、PDF12特殊实验中段、15–16参考文献未完整读；另册附录和代码未核验，未复现结果
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf)；DOI：10.1145/3453483.3454080
- SHA-256：`dc551c57c4d7b46d378f6449209c3a8355507603e47b69003cdfc2f686cafa1f`

## 边界与下一问

主消融覆盖六个领域，另有两个特殊实验；递归基础学习实验没有使用识别网络。少量任务例不等于低总训练成本，完整DSL学习仍是开放问题。 图7中的生成文本任务使用每字符预测似然，不能与其他任务的解题率混为同一指标；EC2各领域运行次数也不同。

在新数据类型、不同任务分布和更现实的噪声下，库与搜索策略能否继续迁移？
