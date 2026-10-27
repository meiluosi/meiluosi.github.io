<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2010 · 感知更新与行动能否共享一个优化框架？

[控制论、强化学习与主动推断路线](../routes/cybernetics-rl.md) · [概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [文献夹](../literature.md)

**原始问题：** 怎样把感知、学习和行动放进同一生成模型框架，同时说明其生物学解释条件？

**方法与机制：** 用变分自由能界定模型下感官数据的惊异；感知改变识别分布以逼近后验，行动改变采样数据。特定高斯与层级假设下以精度加权预测误差实现更新。

**成立条件：** 需要生成模型、可实现的识别分布及状态／运动先验；把自维持与这种优化联系起来还依赖该文提出的额外理论前提。

## 主张与原文定位

- **原文主张**：自由能可分解为模型下数据的负对数证据与识别分布到后验的KL散度；固定数据时，优化识别分布降低后者。
  - 依据：[The Free-energy Principle: A Unified Brain Theory?](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)；AOP2页Box 1、3页Implications；按正文及分解解释，图框方向差异另记。独立验证：未做独立复现。

- **原文主张**：预测编码的具体神经实现采用高斯／Laplace及层级模型假设；由此得到精度加权误差与上下层消息传递。
  - 依据：[The Free-energy Principle: A Unified Brain Theory?](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)；AOP4页及Box 2。独立验证：未做独立复现。

- **原文主张**：山地车例通过对状态运动的先验，使目标外固定点不稳定、目标处稳定；作者也说明仅期待到达某状态并不足以保证绕过障碍或满足动力学约束。
  - 依据：[The Free-energy Principle: A Unified Brain Theory?](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)；AOP8页控制讨论、9页图3。独立验证：未做独立复现。

- **原文主张**：文章将跨理论统一作为研究提案；开篇说明统一能力尚待建立，结论要求进一步说明其在脑中的具体实现。
  - 依据：[The Free-energy Principle: A Unified Brain Theory?](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)；AOP1页引言、9–10页结论。独立验证：未做独立复现。

## 阅读记录

### The Free-energy Principle: A Unified Brain Theory?

Karl Friston，2010。作者网站12页Advance Online Publication，2010-01-13；定位使用AOP页码，不套用期刊127–138页

- 实读：AOP1–4及7–10页，覆盖原则、Box 1–2、控制与结论；11页参考文献40已核对。Box 1已视觉核对，发现图中文字与正文的优化方向不一致。
- 未覆盖：5–6页未完整读；补充框S1–S5、原始模拟、所引实验及普遍性推导未审查。Box 1下方图框写arg max Divergence，而同页正文与KL分解要求降低该项；本记录按正文与分解解释，未取得作者勘误。参考文献40误列Rao–Ballard为1998，本项目采用原刊1999日期。
- 状态：selected-sections；核对日期：2026-10-26
- 来源：[The Free-energy Principle: A Unified Brain Theory?](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf)；DOI：10.1038/nrn2787
- SHA-256：`aeb47a0a36ada532bb5632108b1931c8ae4ab11c4e62b012bf0c3548ffd86e1f`

## 边界与下一问

变分分解本身不证明生物系统必然按该过程运行。运动先验承载目标与探索约束；本文的到达和山地车是指定模拟，不能称无先验发现任务。作者开篇即承认统一能力尚待建立。

在模型、先验和计算预算明确后，哪些干预能区分该解释与其他反馈或信念状态控制模型？
