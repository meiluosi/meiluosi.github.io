<!-- 由 research.json 生成；修改数据后重生成。B批已验收；不代表完整路线史。 -->

# 2020 · 把认知策略的收益与成本共同建模

[理性、适应与资源约束路线](../routes/rational-resource.md) · [文献夹](../literature.md)

**原始问题：** 怎样由任务功能和认知约束推导可检验的过程模型？

**方法与机制：** 在可执行策略类中权衡预期结果与计算机会成本，再将预测与经验比较。

**成立条件：** 需独立约束能力、代价和环境分布；不假设每次决定都现场解决最优化。

## 主张与原文定位

- **原文主张**：作者把优化作用于认知机制；简单规则可通过学习与演化获得，而非每次重新求解。
  - 依据：[Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources](https://cocosci.princeton.edu/papers/liederresource.pdf)；5–6，式2–4与Box 2。独立验证：未做独立复现。

- **原文主张**：未知环境通过分布进入模型；有限适应意味着人不必完全资源理性。
  - 依据：[Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources](https://cocosci.princeton.edu/papers/liederresource.pdf)；5，式4后。独立验证：未做独立复现。

- **原文主张**：自由调整限制可能过拟合；作者要求独立容量测量或预测建模时未使用的现象。
  - 依据：[Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources](https://cocosci.princeton.edu/papers/liederresource.pdf)；§6，14–15。独立验证：未做独立复现。

## 评价视角

| 项目 | 条件与边界 |
| --- | --- |
| 评价对象／构念 | 认知机制如何使用有限资源 |
| 任务与环境 | 理论与实验范式综合 |
| 样本／适用主体 | 未在本批重建经验样本 |
| 输出指标 | 结果效用减计算机会成本；行为和反应时预测 |
| 信度与不确定性 | 约束和成本测量的不确定性会传递到最优策略预测。 |
| 效度与解释范围 | 独立约束与样本外预测防止循环解释 |
| 先验与经验 | 经验与演化环境用于适应解释 |
| 计算与时间预算 | 可用操作、速度、记忆、机会成本 |

## 阅读记录

### Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources

Falk Lieder；Thomas L. Griffiths，2020。BBS 43 e1；2019在线稿、2020卷年；61页合订PDF含评论与作者回应。

- 实读：目标文章1、4开头至bounded optimality段、5–6、14–15；5页公式2–4像素核对。
- 未覆盖：未逐项读所有案例与实验证据；正文16结尾及全部评论不算已读；未运行作者模型。
- 状态：selected-sections；核对日期：2026-10-27
- 来源：[Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources](https://cocosci.princeton.edu/papers/liederresource.pdf)；DOI：10.1017/S0140525X1900061X
- SHA-256：`9ca5002d52aa7a7fb2264d259b3e51319f1dcb7064674ba5d7bde74321f855f9`

## 边界与下一问

这是一套生成假说的方法，成功拟合不证明所有人类偏差都最优；真实环境与约束测量仍困难。

在独立测得约束后，模型能否预言新的代价—准确率变化？
