<!-- 由 research.json 生成；修改数据后重生成。E批已验收；不代表完整路线史。 -->

# 1948 · 通信的不确定性：明确排除语义问题

[信息、压缩、复杂系统与物理路线](../routes/information-physics.md) · [文献夹](../literature.md)

**原始问题：** 怎样量化消息选择并设计可靠通信？

**方法与机制：** 用概率分布、对数单位与熵刻画不确定性。

**成立条件：** 给定消息集合和概率；通信工程目标不等于意义或价值。

## 主张与原文定位

- **原文主张**：导论将工程问题限定为在另一点再现所选消息，并将语义层面排除在该问题之外。
  - 依据：[A Mathematical Theory of Communication](https://marco-dalai.unibs.it/teach/IT/shannon_1948.pdf)；PDF 1导论。独立验证：未做独立复现。

- **原文主张**：§6以连续性、等概率时单调性和选择分解条件引出H=-KΣp log p；作者也说明这套公理化说明不是整套理论的必要前提。
  - 依据：[A Mathematical Theory of Communication](https://marco-dalai.unibs.it/teach/IT/shannon_1948.pdf)；PDF 14–15，§6／Theorem 2。独立验证：未做独立复现。

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
| 证据性质 | 数学定义与通信理论 |
| 分析单位 | 消息分布 |
| 观察／论证依据 | 定义及定理陈述选读，未重验全证明 |
| 需要区分的解释 | 不确定性、任务相关性与意义 |

## 阅读记录

### A Mathematical Theory of Communication

Claude E. Shannon，1948。原作1948；实读79页重印PDF，首页标Reprinted December, 1957。

- 实读：PDF 1导论；14–15（印刷18–19）§6熵定义；10–11语言序列例。
- 未覆盖：编码定理及附录证明未系统审校；不将语义解释视为本论文已解决。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[A Mathematical Theory of Communication](https://marco-dalai.unibs.it/teach/IT/shannon_1948.pdf)
- SHA-256：`42166a1d255e91141de04df87f9bb881b227e36836928c8901e52f57a5d20fb0`

## 边界与下一问

高熵不自动表示理解；论文没有给出通用智能量表。

何种信息对特定任务有用，能否与仅仅不可预测区分？
