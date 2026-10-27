<!-- 由 research.json 生成；修改数据后重生成。B批已验收；不代表完整路线史。 -->

# 2007 · 跨环境的通用智能度量

[理性、适应与资源约束路线](../routes/rational-resource.md) · [行为表现与操作性路线](../routes/behavioral.md) · [文献夹](../literature.md)

**原始问题：** 怎样将不同环境的目标达成汇总成一个形式指标？

**方法与机制：** 对可计算、收益总和有界的环境，以2^-K权重累加代理的预期收益。

**成立条件：** 环境含目标机制；参考机决定复杂度，时间偏好进入收益设置。

## 主张与原文定位

- **原文主张**：度量采用收益总和不超过1的环境，并依描述复杂度加权，而非对所有任务均匀平均。
  - 依据：[Universal Intelligence: A Definition of Machine Intelligence](https://www.hutter1.net/ai/iorx.pdf)；§3.2–3.3，19–23。独立验证：未做独立复现。

- **原文主张**：作者明确区分不可计算的定义与有限样本实际测验；参考机改变会影响复杂度常数。
  - 依据：[Universal Intelligence: A Definition of Machine Intelligence](https://www.hutter1.net/ai/iorx.pdf)；23、27–28、39。独立验证：未做独立复现。

- **原文主张**：对查表和中文房间的回应采取功能立场，不以实现效率或不可观测理解单独否定高分。
  - 依据：[Universal Intelligence: A Definition of Machine Intelligence](https://www.hutter1.net/ai/iorx.pdf)；§5.2，42。独立验证：未做独立复现。

## 评价视角

| 项目 | 条件与边界 |
| --- | --- |
| 评价对象／构念 | 广泛环境中的目标达成 |
| 任务与环境 | 可计算且收益可求和的交互环境 |
| 样本／适用主体 | 形式环境类；实践抽样待设计 |
| 输出指标 | 加权预期总收益Υ |
| 信度与不确定性 | 理想量不可计算；有限环境抽样另有估计误差。 |
| 效度与解释范围 | 定义与测验分开；不识别内部机制 |
| 先验与经验 | 固定参考机和复杂度权重 |
| 计算与时间预算 | 原指标不直接惩罚代理内部计算 |

## 阅读记录

### Universal Intelligence: A Definition of Machine Intelligence

Shane Legg；Marcus Hutter，2007。作者站49页稿，封面December 2007；本文页码按该PDF。

- 实读：17–23、27–28、39–42：交互、收益有界、定义、性质与常见批评；23页公式像素核对。
- 未覆盖：未逐条审读全部历史综述、AIXI最优性证明与批评回应尾部；未实现通用测验。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Universal Intelligence: A Definition of Machine Intelligence](https://www.hutter1.net/ai/iorx.pdf)
- SHA-256：`4227cbd1e4959779db875c4c78fad5b78976bb19c7ec01ddbe463cb92b7ba77d`

## 边界与下一问

不是现成的通用排行榜；复杂环境不等于困难任务，本文22页明确区分二者。

有限任务抽样、先验和预算怎样改变实际排序？
