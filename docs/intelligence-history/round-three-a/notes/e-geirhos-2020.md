<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2020 · 高测试分数究竟来自哪条决策规则？

[连接主义、统计学习与表征路线](../routes/connectionist.md) · [学习效率与泛化路线](../routes/learning-efficiency.md) · [文献夹](../literature.md)

**原始问题：** 当多种规则都能拟合训练与同分布测试时，怎样判断学到了预期能力？

**方法与机制：** 改变候选线索与标签的关系，比较同分布与分布外表现；联合审查架构、数据、损失和优化带来的偏向。

**成立条件：** 需要明确目标能力、可解的分布变化及对应对照；本文是观点文章，实例证据与普遍解释应分别判断。

## 主张与原文定位

- **原文主张**：同分布测试可能无法区分形状、像素数量与位置等候选规则；分布外测试须明确预期解和改变的条件。
  - 依据：[Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780v5)；§3，4–6页；§6.2，12–13页。独立验证：未做独立复现。

- **原文主张**：星月玩具实验中，位置与类别在训练时相关；附录报告位置随机化测试下MLP为51.0%，特定CNN为100%，两者训练准确率均为100%。
  - 依据：[Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780v5)；5页图2；28页Appendix A。独立验证：未做独立复现。

- **原文主张**：作者以架构、训练数据、损失和优化四方面分析学习偏向，并强调大规模数据仍可保留捷径机会。
  - 依据：[Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780v5)；§4.1，7页；Box II及§6.3，15页。独立验证：未做独立复现。

## 阅读记录

### Shortcut Learning in Deep Neural Networks

Robert Geirhos；Jörn-Henrik Jacobsen；Claudio Michaelis；Richard Zemel；Wieland Brendel；Matthias Bethge；Felix A. Wichmann，2020。2020年首次预印本及期刊文章；实际读取arXiv:2004.07780v5（2023-11-21），29页，页码不采用期刊665–673

- 实读：PDF1–18，§1–8及Boxes I–II；PDF28 Appendix A玩具实验方法；图2与附录A已视觉核对。固定v5下载指纹与第二轮暂存但未读的文件相同。
- 未覆盖：未逐篇核查其引用的视觉、语言、医疗、教育等研究；未运行作者代码、核对多种子稳定性或比较v1到v5的改动；不能用本次阅读声称2020初稿逐字相同。
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[Shortcut Learning in Deep Neural Networks](https://arxiv.org/pdf/2004.07780v5)；DOI：10.1038/s42256-020-00257-z；[书目核对](https://arxiv.org/abs/2004.07780v5)
- SHA-256：`da93a1633b56f6590836afaf3794c6131d38d53629e2cf9d219deb274d91ec15`

## 边界与下一问

玩具任务不能证明卷积网络总能避开捷径；“总选择最容易解”的表述是作者概括，不是本文证明的一般定理。所读为2023年v5，未对照2020各版。

控制预期规则与捷径规则的可用性后，哪些训练条件改变模型依赖的线索？
