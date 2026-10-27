<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 1971 · 如何在世界模型之间搜索行动计划？

[符号、逻辑、搜索与组合路线](../routes/symbolic-logic.md) · [概率推断、因果模型与世界模型路线](../routes/causal-world-models.md) · [文献夹](../literature.md)

**原始问题：** 怎样减少把整个行动序列都交给定理证明器造成的搜索困难？

**方法与机制：** 把模型内的证明与模型之间的搜索分开；用前提和增删列表描述算子，以未完成证明形成差异，再寻找相关算子及其子目标。

**成立条件：** 需给定初始逻辑模型、算子及目标；实际行动程序与模型中的算子应用分开；派生事实还需跟踪其依赖的原始事实。

## 主张与原文定位

- **原文主张**：作者明确分离模型内定理证明和模型间搜索，并区分规划算子与实际执行的行动程序。
  - 依据：[STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving](https://www.ijcai.org/Proceedings/71/Papers/055.pdf)；608页引言及脚注。独立验证：未做独立复现。

- **原文主张**：算子以适用条件和增删列表描述；删除原始事实后，还需检查依赖它的非原始事实是否仍有效。
  - 依据：[STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving](https://www.ijcai.org/Proceedings/71/Papers/055.pdf)；609页§II.A、612–613页§III.B。独立验证：未做独立复现。

- **原文主张**：615页报告灯开关问题解非最优，并注明当时SRI机器人不能爬箱和开灯。
  - 依据：[STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving](https://www.ijcai.org/Proceedings/71/Papers/055.pdf)；615页§IV及脚注。独立验证：未做独立复现。

## 阅读记录

### STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving

Richard E. Fikes；Nils J. Nilsson，1971。IJCAI 1971会议录608–620页；不是同年Artificial Intelligence 189–208页期刊版，后者DOI不套用于此文件

- 实读：PDF1–8／608–615页，含问题空间、搜索、参数、算子、模型存储和三箱例；PDF13参考文献；PDF5的push算子已视觉核对
- 未覆盖：PDF9–12表格、图3和结尾讨论未完整读；期刊版未比对；原代码、机器人执行与性能表未复现
- 状态：selected-sections；核对日期：2026-10-25
- 来源：[STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving](https://www.ijcai.org/Proceedings/71/Papers/055.pdf)
- SHA-256：`293e2f5613b940bfa7f0f053d5669423fe08813cb23e19f1a356a8c47ea1c4bc`

## 边界与下一问

原文世界模型允许一阶公式，不能直接缩写成所有现代教材中的简化STRIPS定义。所得计划不保证最短；能规划爬箱不等于当时机器人能执行。

如何在行动效果不确定、观测不完整或模型错误时修正计划？
