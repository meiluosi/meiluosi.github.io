<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1958 · 识别偏向怎样随经验改变？

[连接主义、统计学习与表征路线](../routes/connectionist.md) · [学习效率与泛化路线](../routes/learning-efficiency.md) · [文献夹](../literature.md)

**原始问题：** 在连接结构含随机性的系统中，记忆与泛化能否由物理参数和环境统计关系解释？

**方法与机制：** 感觉、联想及响应单元组织为网络；活动和强化改变联想单元的输出效力，进而改变响应竞争。

**成立条件：** 本文比较多种增益、判别及强化设置；相似性依赖网络组织和刺激环境，不能简化为一个不带条件的现代线性分类器。

## 主张与原文定位

- **原文主张**：原文的感知机包含感觉、联想和响应阶段，并讨论不同反馈和增益规则；学习改变的是响应倾向。
  - 依据：[The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://web.engr.oregonstate.edu/~huanlian/teaching/ML/2020spring/extra/rosenblatt-1958.pdf)；389–392页，图1、表1。独立验证：未做独立复现。

- **原文主张**：记忆测试重现训练刺激；泛化测试使用独立位置等变化的新刺激。两两响应偏好概率也不同于在全部响应中正确选择的概率。
  - 依据：[The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://web.engr.oregonstate.edu/~huanlian/teaching/ML/2020spring/extra/rosenblatt-1958.pdf)；395–396页。独立验证：未做独立复现。

- **原文主张**：作者把相对判断与时空关系抽象列为当前感知机的能力边界，而未宣称已解释完整人类学习。
  - 依据：[The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://web.engr.oregonstate.edu/~huanlian/teaching/ML/2020spring/extra/rosenblatt-1958.pdf)；404–406页。独立验证：未做独立复现。

## 阅读记录

### The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain

Frank Rosenblatt，1958。Psychological Review 65(6):386–408；23页原刊扫描

- 实读：PDF1–11／386–396页的动机、架构、增益方式及记忆／泛化协议；PDF19–23／404–408页的关系抽象边界、结论和参考文献。图1与表1已视觉核对。
- 未覆盖：397–403页的详细学习曲线、推导及扩展尚未完整读；未复现仿真，也未将后来的感知机收敛定理或教材更新式倒填到本篇。
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://web.engr.oregonstate.edu/~huanlian/teaching/ML/2020spring/extra/rosenblatt-1958.pdf)；DOI：10.1037/h0042519；[书目核对](https://pubmed.ncbi.nlm.nih.gov/13602029/)
- SHA-256：`9c2228eeb866cd8ce4b45ad02ecdac9fcf34f196c12075a2eab0ba82bf995d37`

## 边界与下一问

作者区分多种模型与学习协议，并承认关系抽象困难；本次没有复核完整统计推导或重跑仿真。生物类比仍是模型假设。

固定输入特征与可学习内部特征各自限制了哪些新任务和关系泛化？
