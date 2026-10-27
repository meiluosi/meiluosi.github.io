<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1982 · 局部消息怎样完成全局信念更新？

[概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [文献夹](../literature.md)

**原始问题：** 如何在分层推理系统中传播新证据，同时避免证据被重复计算？

**方法与机制：** 在给定概率树中分离来自下方的似然支持与上方的先验支持；每条边传递局部消息，向下更新时移除子节点此前贡献。

**成立条件：** 每个节点至多一个父节点，离散状态及条件概率已给定；子树在给定父状态时条件独立。

## 主张与原文定位

- **原文主张**：本文从单父节点概率树及局部条件独立出发，把每个节点后验分解为上下两部分支持的归一化乘积。
  - 依据：[Reverend Bayes on Inference Engines: A Distributed Hierarchical Approach](https://cdn.aaai.org/AAAI/1982/AAAI82-032.pdf)；133–134页，式1–6。独立验证：未做独立复现。

- **原文主张**：父节点向子节点发送信息时，需要除去该子节点上一次已纳入的消息，避免同一证据回流后重复计算。
  - 依据：[Reverend Bayes on Inference Engines: A Distributed Hierarchical Approach](https://cdn.aaai.org/AAAI/1982/AAAI82-032.pdf)；135页，式10及其前文。独立验证：未做独立复现。

- **原文主张**：作者用先自下而上、再自上而下的归纳论证树上的平衡值正确，并报告新信息单遍扩散的性质。
  - 依据：[Reverend Bayes on Inference Engines: A Distributed Hierarchical Approach](https://cdn.aaai.org/AAAI/1982/AAAI82-032.pdf)；136页 Characteristics／A Summary of Proofs。独立验证：未做独立复现。

## 阅读记录

### Reverend Bayes on Inference Engines: A Distributed Hierarchical Approach

Judea Pearl，1982。AAAI-82，133–136页；4页会议原文

- 实读：全文133–136页；135页式7–10与消息传递图已视觉核对。
- 未覆盖：未实现异步更新；未将树上的结论扩展到有环网络；未以本篇替代1988年专著的阅读。
- 状态：full-text；核对日期：2026-10-26
- 来源：[Reverend Bayes on Inference Engines: A Distributed Hierarchical Approach](https://cdn.aaai.org/AAAI/1982/AAAI82-032.pdf)
- SHA-256：`8bfd6c94c5263ec0a7a6336c9aa5a6562221e4b7847440a927f7a292ba5bcfe6`

## 边界与下一问

单遍传播与有限收敛结论针对本文树结构；没有解决结构学习、任意有环推理或因果识别。

当依赖图包含环或概率未知时，哪些正确性与收敛保证仍成立？
