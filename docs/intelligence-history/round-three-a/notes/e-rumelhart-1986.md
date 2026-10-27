<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1986 · 隐藏单元怎样获得学习信号？

[连接主义、统计学习与表征路线](../routes/connectionist.md) · [学习效率与泛化路线](../routes/learning-efficiency.md) · [文献夹](../literature.md)

**原始问题：** 任务只指定输入和目标输出时，怎样学习未被直接指定的内部表示？

**方法与机制：** 前向计算活动，反向用链式法则传播误差导数，再以梯度及动量更新连接；内部特征随任务形成。

**成立条件：** 本文采用可微单元和给定输入输出样例；训练结果依赖架构、随机初值及优化设置。

## 主张与原文定位

- **原文主张**：误差导数可以逐层反传，使隐藏单元输入连接得到训练信号；正文给出梯度与动量更新。
  - 依据：[Learning Representations by Back-propagating Errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)；533–535页，式1–9。独立验证：未做独立复现。

- **原文主张**：家族关系任务使用两个同构家族树；作者报告训练104个关系三元组中的100个后，对留出的4个正确泛化。
  - 依据：[Learning Representations by Back-propagating Errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)；534–535页，图2–4及正文。独立验证：未做独立复现。

- **原文主张**：原文承认局部极小值与生物可信性限制，并注明Parker与Le Cun独立发现相关变体。
  - 依据：[Learning Representations by Back-propagating Errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)；535–536页。独立验证：未做独立复现。

## 阅读记录

### Learning Representations by Back-propagating Errors

David E. Rumelhart；Geoffrey E. Hinton；Ronald J. Williams，1986。Nature 323:533–536，1986-10-09；作者网站提供的4页期刊扫描

- 实读：533页本文标题起至536页本文参考文献止，全文及式1–9、图1–5已逐页视觉阅读；同页其他文章不属于本文。
- 未覆盖：未重跑对称性与家族关系实验；参考文献中的PDP长章、Le Cun 1985及更早历史原文未读，不据本文宣称反向传播首次发明。
- 状态：full-text；核对日期：2026-10-25
- 来源：[Learning Representations by Back-propagating Errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)；DOI：10.1038/323533a0；[书目核对](https://doi.org/10.1038/323533a0)
- SHA-256：`d26997baf588222109d32545604a2a2ed400dc769a21fd49a5acdc4a955396ae`

## 边界与下一问

下降方向不保证全局最优；四个留出家族关系的成功不是任意组合泛化的证明。作者明确否认当时算法已是可信的脑学习模型。

相同优化方法在不同结构先验和训练分布下，会学出什么表示和外推行为？
