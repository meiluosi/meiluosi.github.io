<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2004 · 并行模块如何形成连贯的认知过程？

[认知架构与多机制协作路线](../routes/cognitive-architectures.md) · [文献夹](../literature.md)

**原始问题：** 视觉、动作、目标和记忆各自工作时，怎样约束它们的相互协调？

**方法与机制：** ACT-R 5.0让模块经有限缓冲区与产生式系统交互；模块可异步并行，但单个缓冲区和每周期产生式选择形成串行约束。

**成立条件：** 约50毫秒周期及缓冲区容量是模型假设；脑区对应为作者提出的机制解释，不是直接测得的软件式模块边界。

## 主张与原文定位

- **原文主张**：本文明确采用ACT-R 5.0；产生式系统只访问模块缓冲区的有限信息，模块内部及模块之间可并行。
  - 依据：[An Integrated Theory of the Mind](https://web.iitd.ac.in/~sumeet/and04.pdf)；1036–1038页。独立验证：未做独立复现。

- **原文主张**：双任务几乎无干扰并不自动排除中央串行瓶颈：该模型用两任务对瓶颈的需求在时间上错开来解释。
  - 依据：[An Integrated Theory of the Mind](https://web.iitd.ac.in/~sumeet/and04.pdf)；1039–1041页，图2–3及后续讨论。独立验证：未做独立复现。

- **原文主张**：总讨论承认全部神经处理都经基底节的解释不成立，并列出未解释的学习相关反应。
  - 依据：[An Integrated Theory of the Mind](https://web.iitd.ac.in/~sumeet/and04.pdf)；1057页 General Discussion。独立验证：未做独立复现。

## 阅读记录

### An Integrated Theory of the Mind

John R. Anderson；Daniel Bothell；Michael D. Byrne；Scott Douglass；Christian Lebiere；Yulin Qin，2004。Psychological Review 111(4):1036–1060；本文描述ACT-R 5.0，不能当作所有版本的固定结构

- 实读：PDF1–6／1036–1041页：架构、缓冲区、串并行与双任务例；PDF22／1057页讨论；PDF23结论末段；PDF5／1040页图2时序与图3预测已视觉核对
- 未覆盖：记忆与程序学习详细方程、复杂任务建模及fMRI实验方法未完整读；引用的行为实验和替代模型未复现
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[An Integrated Theory of the Mind](https://web.iitd.ac.in/~sumeet/and04.pdf)；DOI：10.1037/0033-295X.111.4.1036
- SHA-256：`536f4ca80fe940875954fcbf4a655162723a0faa528726837d906f263addb781`

## 边界与下一问

作者承认并非所有神经处理都经过基底节，且部分学习相关影像反应尚无完整解释。单一任务拟合不足以验证整套架构。

固定架构和参数后，改变双任务相对时序能否预测干扰变化？
