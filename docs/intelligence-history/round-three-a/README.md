<!-- 由 research.json 生成；修改数据后重生成。A批已验收；不代表完整路线史。 -->

# 第三轮 A 批：表征与认知机制（已验收）

目标：补三个样板的关键断层，展开四条新路线，完成三张跨路线问题卡和一项正式复现。**已完成执行计划列明的A批范围；第三轮B—E批仍待开展。 节点数不构成整条路线已覆盖的声明。**

[交互研究工作台](atlas.html) · [完整范围与验收计划](plan.md) · [验收记录](validation.md) · [文献夹](literature.md) · [主张台账](claims.md) · [关系台账](relations.md) · [研究数据](research.json)

[Brooks：1985备忘录与1986期刊](versions/brooks-1985-1986.md) · [MuZero：2019 v1与2020期刊](versions/muzero-2019-2020.md)

当前增量：34个节点、41份有阅读范围的来源、109条定位主张、33条关系。

| 路线 | 当前增量节点 |
| --- | ---: |
| [学习效率与泛化路线](routes/learning-efficiency.md) | 9 |
| [符号、逻辑、搜索与组合路线](routes/symbolic-logic.md) | 9 |
| [连接主义、统计学习与表征路线](routes/connectionist.md) | 10 |
| [概率推断、因果模型与世界模型路线](routes/causal-world-models.md) | 10 |
| [控制论、强化学习与主动推断路线](routes/cybernetics-rl.md) | 6 |
| [认知架构与多机制协作路线](routes/cognitive-architectures.md) | 6 |
| [具身、生态、生成与动力系统路线](routes/embodied-dynamical.md) | 5 |

一个节点可以属于多条路线，上表不能相加作为独立节点数。

## 跨路线问题卡

- [模型与表征：同一个词承担哪些不同任务？](questions/q-model-representation.md)
- [学习与泛化：究竟把什么能力推广到了哪里？](questions/q-learning-generalization.md)
- [主体与环境：贡献、依赖和系统构成如何区分？](questions/q-agent-environment.md)

问题卡区分原文结论与本站比较；所提区分实验尚未运行。时间线、关系图、问题卡和笔记从同一份数据重建。

## 正式复现

[Baird六状态反例](replication/README.md)：原文版本、预先协议、独立实现、数值结果、可编辑曲线、参数敏感性和原文差异。复现范围限于图1的六状态预测问题，图5速度实验与一般强化学习结论不在本项范围内。

## 维护

修改research.json后运行 `node scripts/generate-intelligence-history-round-three-a.mjs`；加 `--check` 检查基线指纹、引用和生成一致性。上述校验不能代替文献论证审校或完整范围验收。第一轮与第二轮数据不回写当时阅读状态。

[第一轮](../README.md) · [第二轮](../round-two/README.md)
