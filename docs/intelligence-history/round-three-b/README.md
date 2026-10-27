<!-- 由 research.json 生成；修改数据后重生成。B批已验收；不代表完整路线史。 -->

# 第三轮 B 批：能力与评价（已验收）

怎样判断系统具有什么能力，这种判断依赖哪些任务、样本、环境与资源？本批围绕三条路线整理关键转折，并把评价条件带回A批的学习与表征问题。

**已完成plan.md所列B批范围；第三轮C—E批待开展。 这是有明确选读范围的研究增量，不是完整路线史。**

[交互研究工作台](atlas.html) · [执行计划](plan.md) · [验收记录](validation.md) · [文献夹](literature.md) · [主张台账](claims.md) · [关系证据](relations.md) · [评价条件索引](evaluation.md) · [A批回访](revisits.md) · [研究数据](research.json)

当前：25个独立节点、29份来源记录、73条定位主张、20条有类型的关系。下载、选读与独立验证分别记账。

| 路线 | 本批节点 |
| --- | ---: |
| [行为表现与操作性路线](routes/behavioral.md) | 8 |
| [心理测量与能力结构路线](routes/psychometric.md) | 11 |
| [理性、适应与资源约束路线](routes/rational-resource.md) | 7 |

节点可属于多条路线，表中数量不可直接相加。

## 跨路线问题卡

- [表现与能力：高分支持了哪一种解释？](questions/q-performance-capability.md)
- [共同因素与机制：相关结构能确定什么？](questions/q-factor-mechanism.md)
- [效用与资源：对谁、在何种环境中合理？](questions/q-utility-resources.md)

问题卡的后续干预实验均未运行；已经完成的数学例子另列如下。

## 评价方法核验

[同一观察结构与不同潜在构造](method-check/README.md)：预先协议、精确有理数实现、独立标量核对、负对照与均值解释例。结果限定于合成线性高斯模型，不是POT论文复现或真实人群实验。

## 维护

先修改research.json，再运行 `node scripts/generate-intelligence-history-round-three-b.mjs`，加 `--check` 检查生成一致性与引用。核验程序为 `python3 scripts/check-intelligence-evaluation.py --check`。时间线SVG、关系SVG、Mermaid与笔记可重建；直接修改生成文件会被覆盖。

[访问记录与缺口](evidence/access-log.json) · [第一轮](../README.md) · [第二轮](../round-two/README.md) · [第三轮A批](../round-three-a/README.md)
