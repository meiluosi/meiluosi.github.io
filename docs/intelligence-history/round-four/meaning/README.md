<!-- 由meaning/research.json和method-check/results.json生成；编辑源数据后重生成。 -->

# 第四轮第三阶段：意义与意识解释

当系统能说、能推理、能行动，或者显示特定脑信号时，还需要哪些前提，才能谈理解与体验？

比较形式与语义、访问与体验、理论与测量之间的桥接条件。纳入原典、明确反方、预注册检验与理论方回应；不裁定当代某个模型是否有意识，也不把所有意识理论或非经典物理解释计为已覆盖。

[工作台](index.html) · [综合论述](synthesis.md) · [比较矩阵](comparison.md) · [桥接框架](framework.md) · [结论与反方](conclusions.md) · [文献夹](literature.md) · [演进图](timeline.md) · [预测台账](predictions.md) · [版本台账](versions.md) · [方法核验](method-check/README.md) · [未执行设计](research-designs.md) · [验收](validation.md)

本阶段12份来源记录、40条定位主张、6条条件性结论、5类比较案例、3组可编辑图、11项精确检查。12份记录包括同一COGITATE项目的协议、锁定文件、主文与补充材料，不是12项独立研究；均为指定范围选读。

## 维护

编辑[research.json](research.json)，随后运行：

```sh
python3 scripts/check-intelligence-meaning.py --check
node scripts/generate-intelligence-meaning.mjs
node scripts/generate-intelligence-history-index.mjs
python3 scripts/audit-intelligence-meaning.py
```

保留[方法协议](method-check/protocol.md)、[历史基线](baselines.json)和[前两阶段核心内容基线](previous-content-baselines.json)。SVG与Mermaid可编辑，但直接修改生成物会被覆盖；新增节点、桥接与预测应优先修改源数据。正式网站联动进入[第四阶段](../plan.md)。
