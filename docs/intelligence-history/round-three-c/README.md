<!-- 由 research.json 生成；修改数据后重生成。C批已验收；不代表完整路线史。 -->

# 第三轮 C 批：生物与发展（已验收）

智能怎样在神经系统、个体发展和生命组织中形成？本批将机制假设、损伤证据、发展解释、开放式搜索与生命判准分开记录。

**已完成plan.md所列范围；第四轮另行开展。 这是有明确选读范围的研究增量，不是完整路线史。**

[全局总览](../index.html) · [交互研究工作台](atlas.html) · [执行计划](plan.md) · [验收记录](validation.md) · [文献夹](literature.md) · [主张台账](claims.md) · [关系证据](relations.md) · [证据条件索引](evaluation.md) · [前批回访](revisits.md) · [研究数据](research.json)

当前：19条本批节点记录（含延续）、19份来源记录、40条定位主张、15条有类型的关系。下载、选读与独立验证分别记账。

| 路线 | 本批节点 |
| --- | ---: |
| [神经科学路线](routes/neuroscience.md) | 7 |
| [发展与开放式学习路线](routes/developmental-openended.md) | 6 |
| [生命、自维持与演化路线](routes/autopoiesis-evolution.md) | 6 |

节点可属于多条路线，表中数量不可直接相加。

## 跨路线问题卡

- [相关、必要性与机制解释](questions/q-neural-mechanism.md)
- [发展、探索与开放式增长](questions/q-development-openended.md)
- [自维持、适应与认知](questions/q-life-cognition.md)

问题卡的后续干预实验均未运行；已经完成的数学例子另列如下。

## 方法核验

[观察相关相同，干预结果仍可不同](method-check/README.md)：预先协议、精确计算、独立解析对照与负控制。结果只覆盖协议定义的合成模型；不计原论文或生物实验复现。

## 维护

先修改research.json，再运行 `node scripts/generate-intelligence-history-round-three-cde.mjs c`，加 `--check` 检查生成一致性与引用。核验程序为 `python3 scripts/check-intelligence-cde.py c --check`。时间线SVG、关系SVG、Mermaid与笔记可重建；直接修改生成文件会被覆盖。

[访问记录与缺口](evidence/access-log.json) · [第一轮](../README.md) · [第二轮](../round-two/README.md) · [第三轮A批](../round-three-a/README.md)
