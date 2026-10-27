<!-- 由 research.json 生成；修改数据后重生成。E批已验收；不代表完整路线史。 -->

# 2003 · 神经雪崩：临界传播的候选证据

[信息、压缩、复杂系统与物理路线](../routes/information-physics.md) · [文献夹](../literature.md)

**原始问题：** 局部神经活动如何兼顾传播与稳定？

**方法与机制：** 比较皮层局部场电位雪崩大小与临界分枝模型，并用简化网络模拟信息传播。

**成立条件：** 大鼠培养／急性皮层切片，60通道；事件阈值、时间箱和采样决定观察。

## 主张与原文定位

- **原文主张**：作者报告切片雪崩大小近似-3/2幂律、分枝参数近1，解释为临界传播的证据。
  - 依据：[Neuronal Avalanches in Neocortical Circuits](https://pmc.ncbi.nlm.nih.gov/articles/PMC6741045/)；Abstract；Discussion前3段。独立验证：未做独立复现。

- **原文主张**：信息传输优势来自配套模型模拟；研究对象并非人类智能任务表现。
  - 依据：[Neuronal Avalanches in Neocortical Circuits](https://pmc.ncbi.nlm.nih.gov/articles/PMC6741045/)；Introduction末段；Abstract及Discussion。独立验证：未做独立复现。

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
| 证据性质 | 切片实验与配套模拟 |
| 分析单位 | 局部场电位事件／网络传播 |
| 观察／论证依据 | 原文报告及方法范围 |
| 需要区分的解释 | 统计尾形与生成机制 |

## 阅读记录

### Neuronal Avalanches in Neocortical Circuits

John M. Beggs；Dietmar Plenz，2003。J Neurosci 23:11167–11177，PMC全文HTML。

- 实读：Abstract；Materials and Methods的培养与急性切片段落；Discussion前部及临界性小节开头。
- 未覆盖：完整统计重算、网络仿真实现和后续重复研究未做。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Neuronal Avalanches in Neocortical Circuits](https://pmc.ncbi.nlm.nih.gov/articles/PMC6741045/)
- SHA-256：`85afef090b7ff33c82cda3767b786153917648033611bb2471bb892d41c2ea06`

## 边界与下一问

原文实验未独立复现；幂律与临界机制的对应需竞争模型和统计检验。

阈值、采样和非临界随机模型能否产生相似表象？
