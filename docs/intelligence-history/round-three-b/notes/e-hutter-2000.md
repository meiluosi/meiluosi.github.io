<!-- 由 research.json 生成；修改数据后重生成。B批已验收；不代表完整路线史。 -->

# 2000 · AIXI与有条件的资源有界近似

[理性、适应与资源约束路线](../routes/rational-resource.md) · [文献夹](../literature.md)

**原始问题：** 未知环境中怎样形式化选择，以及能否在有限时间实现？

**方法与机制：** 把通用程序先验与序贯决策结合；资源有界构造枚举可证明有效的候选程序。

**成立条件：** 固定参考机、交互编码和规划视野；有限近似另有程序、时间与证明长度界。

## 主张与原文定位

- **原文主张**：AIXI以程序长度加权环境并选择预期收益动作；核心定义不可计算。
  - 依据：[A Theory of Universal Artificial Intelligence based on Algorithmic Complexity](https://www.hutter1.net/ai/pkcunai.pdf)；摘要；§4，19–22。独立验证：未做独立复现。

- **原文主张**：§10比较的是带收益下界证明的候选，其排序涉及可证明的自评收益，不是任意程序的真实表现统治。
  - 依据：[A Theory of Universal Artificial Intelligence based on Algorithmic Complexity](https://www.hutter1.net/ai/pkcunai.pdf)；54–56，式67–68及主定理。独立验证：未做独立复现。

- **原文主张**：每周期开销含2^l乘t，初始化还含证明枚举；有效但不可证明或证明过长的程序不在保证内。
  - 依据：[A Theory of Universal Artificial Intelligence based on Algorithmic Complexity](https://www.hutter1.net/ai/pkcunai.pdf)；56–57，Limitations and open questions。独立验证：未做独立复现。

## 评价视角

| 项目 | 条件与边界 |
| --- | --- |
| 评价对象／构念 | 给定形式环境类中的预期收益 |
| 任务与环境 | 动作—观察—收益交互 |
| 样本／适用主体 | 形式模型，无实验样本 |
| 输出指标 | 预期收益；近似另用可证明的收益下界 |
| 信度与不确定性 | 保证是有条件的形式结果，不是经验置信区间。 |
| 效度与解释范围 | 参考机、环境类、视野与证明系统均属条件 |
| 先验与经验 | 以程序长度构造先验 |
| 计算与时间预算 | 不可计算理想；近似有指数枚举成本 |

## 阅读记录

### A Theory of Universal Artificial Intelligence based on Algorithmic Complexity

Marcus Hutter，2000。作者站64页稿，封面Munich 31.03.2000／cs.AI/0004001；不是2005专著。

- 实读：封面目录；§4印刷页19–22；§10的53–57，含算法、主定理与限制；55页像素已核对。
- 未覆盖：未通读所有证明、示例、视野问题及后续版本；未运行AIXI近似。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[A Theory of Universal Artificial Intelligence based on Algorithmic Complexity](https://www.hutter1.net/ai/pkcunai.pdf)
- SHA-256：`41eec0ff1078133dfacfa1ef1fbf18b0838b70dba95234a15de2f77525ee179e`

## 边界与下一问

理论构造不提供现实可用性保证；2005书尚未审读，不借用其结果补齐本稿证明。

在明确资源预算和证明限制下，可计算近似还能保留哪些性质？
