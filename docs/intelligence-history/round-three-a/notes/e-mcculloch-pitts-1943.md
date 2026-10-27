<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1943 · 固定神经网络能够实现哪些逻辑关系？

[连接主义、统计学习与表征路线](../routes/connectionist.md) · [文献夹](../literature.md)

**原始问题：** 怎样从连接结构计算网络行为，又从指定的逻辑条件构造网络？

**方法与机制：** 以全或无活动、阈值、抑制与离散延迟构造时序命题表达式，并分析无环网络的可实现性。

**成立条件：** 原文采用五项理想化假设，包括结构不随时间改变、抑制绝对生效；扩展等价允许时间延迟不同。

## 主张与原文定位

- **原文主张**：无环网络与原文定义的时序命题表达式之间存在构造性对应；定理III要求所有输入命题为假时结果也为假。
  - 依据：[A Logical Calculus of the Ideas Immanent in Nervous Activity](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf)；§II，118–121页，定理I–III。独立验证：未做独立复现。

- **原文主张**：网络结构固定是所用假设之一；以固定网络替代可变过程的形式等价，作者明确不视为事实性的生理解释。
  - 依据：[A Logical Calculus of the Ideas Immanent in Nervous Activity](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf)；117–118页。独立验证：未做独立复现。

- **原文主张**：具有相同行为的网络可以拓扑不同；仅从外显行为反推实际神经网络并不唯一。
  - 依据：[A Logical Calculus of the Ideas Immanent in Nervous Activity](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf)；121页定理II后；132页。独立验证：未做独立复现。

## 阅读记录

### A Logical Calculus of the Ideas Immanent in Nervous Activity

Warren S. McCulloch；Walter Pitts，1943。The Bulletin of Mathematical Biophysics 5:115–133；19页原刊扫描，非1990重印本

- 实读：PDF1–8／115–122页：引言、五项假设、时序命题表达式及定理I–III；另读129、131–133页。121页定理III已视觉核对。
- 未覆盖：123–128页的等价变换及循环网络详细推导未读；图1未逐项复核；129页关于图灵计算的结论未独立证明，不将有限网络直接等同于无界图灵机。
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[A Logical Calculus of the Ideas Immanent in Nervous Activity](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf)；DOI：10.1007/BF02478259
- SHA-256：`f977a9c0278dacf7b45dd144c089eb363a955e905ffef1fdb96ba0622f311151`

## 边界与下一问

形式等价不等于生理机制解释；这里没有从数据自动学出网络的算法。作者对心理过程的广泛解释超出已核查定理，不作为现代经验事实。

如果结构事先未知，怎样通过经验改变网络，并区分可以表达与可以学会？
