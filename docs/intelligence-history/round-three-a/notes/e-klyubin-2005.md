<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 2005 · 行动能带来多少可区分的感知结果？

[控制论、强化学习与主动推断路线](../routes/cybernetics-rl.md) · [具身、生态、生成与动力系统路线](../routes/embodied-dynamical.md) · [文献夹](../literature.md)

**原始问题：** 能否以较通用的内在准则，评价一个主体潜在的可感知控制能力？

**方法与机制：** 把一段可自主选择的行动作为通道输入、未来感觉作为输出，最大化两者互信息得到通道容量；用它评价可区分的潜在行动效果。

**成立条件：** 动作序列可被干预选择；需要行动到感知的通道模型、观察尺度和时间跨度。本文实验还固定起点、形态约束及小的感知／动作数量代价。

## 主张与原文定位

- **原文主张**：赋能度定义为行动序列到未来感知通道的容量，即在允许的行动分布上最大化互信息；它评价潜在可控性，而非实际执行频率。
  - 依据：[Empowerment: A Universal Agent-Centric Measure of Control](https://uhra.herts.ac.uk/id/eprint/282/1/901241.pdf)；§2.1–2.2，129–130页，式1–3；§4，134页。独立验证：未做独立复现。

- **原文主张**：本文两项演化实验分别固定动作集合优化传感器、固定传感器优化动作；以四步赋能并加小的规模惩罚评价形态。
  - 依据：[Empowerment: A Universal Agent-Centric Measure of Control](https://uhra.herts.ac.uk/id/eprint/282/1/901241.pdf)；§2.3及§3，131–133页；式4。独立验证：未做独立复现。

- **原文主张**：作者明确指出，只要主体能发出并看到闪光，赋能准则会保留相应能力，无论该能力是否在常识意义上有用。
  - 依据：[Empowerment: A Universal Agent-Centric Measure of Control](https://uhra.herts.ac.uk/id/eprint/282/1/901241.pdf)；§2.3，131页。独立验证：未做独立复现。

## 阅读记录

### Empowerment: A Universal Agent-Centric Measure of Control

Alexander S. Klyubin；Daniel Polani；Chrystopher L. Nehaniv，2005。2005 IEEE Congress on Evolutionary Computation，128–135页；作者大学资料库公开原刊扫描

- 实读：全文128–135页；130页式3及图1–2、132页式4和实验设置已视觉核对。
- 未覆盖：没有重跑演化实验或复核Blahut容量计算；作者对Gibson、Ashby的历史解释不替代这些原作的独立阅读。
- 状态：full-text；核对日期：2026-10-26
- 来源：[Empowerment: A Universal Agent-Centric Measure of Control](https://uhra.herts.ac.uk/id/eprint/282/1/901241.pdf)
- SHA-256：`19bd4594ea68f6aafd44ce2c929de315ab78c5b14d82a4c930b3cbe76bc47ba6`

## 边界与下一问

容量描述潜能，不能用既有策略下互信息直接替代；高赋能不保证任务奖励或生存。作者的黑暗中自产闪光例表明，只要可控且可感知，常识上无用的能力也会被保留。

固定可达状态后，改变传感器、行动代价和时间跨度，赋能偏好会怎样改变？
