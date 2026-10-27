<!-- 由 research.json 生成；修改数据后重生成。E批已验收；不代表完整路线史。 -->

# 1999 · 信息瓶颈：先指定相关变量

[信息、压缩、复杂系统与物理路线](../routes/information-physics.md) · [文献夹](../literature.md)

**原始问题：** 压缩时应保留什么，而非把一切信息同等保留？

**方法与机制：** 压缩X到T，同时保留T关于Y的互信息；目标I(T;X)-βI(T;Y)。

**成立条件：** 需给定相关变量Y与联合分布p(x,y)，以及表示集合。

## 主张与原文定位

- **原文主张**：相关性通过另一变量Y定义；估计联合分布所需样本的问题被明确留在论文范围之外。
  - 依据：[The information bottleneck method](https://www.princeton.edu/~wbialek/our_papers/tishby%2Bal_99.pdf)；PDF 8，§3及脚注1。独立验证：未做独立复现。

- **原文主张**：式15权衡压缩与保留Y信息，没有唯一正确的权衡；后文的收敛说明不保证解唯一。
  - 依据：[The information bottleneck method](https://www.princeton.edu/~wbialek/our_papers/tishby%2Bal_99.pdf)；PDF 9，式15；14，§3.3结尾。独立验证：未做独立复现。

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
| 证据性质 | 变分目标与形式方法 |
| 分析单位 | X、T、Y的联合分布 |
| 观察／论证依据 | 目标及范围选读 |
| 需要区分的解释 | 任务相关性与普遍意义 |

## 阅读记录

### The information bottleneck method

Naftali Tishby；Fernando C. Pereira；William Bialek，1999。稿内日期1999-09-30；所读arXiv physics/0004057 v1于2000-04-24提交，16页。

- 实读：PDF 1、3–4背景；8–9 §3及式15；14收敛不保证唯一性的段落。
- 未覆盖：全部变分证明、实现和后续神经网络解释未独立复验。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[The information bottleneck method](https://www.princeton.edu/~wbialek/our_papers/tishby%2Bal_99.pdf)
- SHA-256：`45dd6ccfe4fa85bc6a82460b9de4b2b579ee31b7c535f7c8d86561e3cf5c6f39`

## 边界与下一问

这里的“meaningful”指相对Y的相关信息，不直接等于哲学语义或意识。

任务改变或Y不完整时，被压掉的信息是否成为失败原因？
