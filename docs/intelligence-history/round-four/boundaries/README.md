<!-- 由boundaries/research.json与method-check/results.json生成；修改数据后重生成。 -->

# 第四轮第二阶段：系统边界与发展

工具、互动伙伴与生命过程在什么条件下进入解释单位？当下贡献、发展依赖与认知构成如何区分？

对齐五类案例，补读九份原始来源的指定片段，保留支持、反对与回应；精确核验限于自建模型。尚未开展人类实验、原始数据复算或生命模拟。

[交互工作台](index.html) · [完整综述](synthesis.md) · [比较矩阵](comparison.md) · [结论与反方](conclusions.md) · [文献夹](literature.md) · [演进与关系](timeline.md) · [方法核验](method-check/README.md) · [研究方案](research-designs.md) · [阅读增量](reading-updates.md) · [验收](validation.md)

本阶段9份来源记录均为指定片段选读，形成27条定位主张、6条条件性结论、5个比较案例、3组可编辑图、8项精确检查和3项未执行研究设计。篇数不是阅读完整度或独立实验数。

## 维护入口

以[research.json](research.json)为内容源，[protocol.md](method-check/protocol.md)保留先于实现保存的协议。图的文字与关系由源数据和生成器维护，SVG、Mermaid与HTML可编辑；直接改生成物会在重生成时被覆盖。

从仓库根目录运行：

```sh
python3 scripts/check-intelligence-boundaries.py --check
node scripts/generate-intelligence-boundaries.mjs
node scripts/generate-intelligence-history-index.mjs
python3 scripts/audit-intelligence-boundaries.py
```

[历史基线](baselines.json)和[第一阶段内容基线](phase-one-baselines.json)用于防止改写旧研究结论。第三阶段成果见[意义与意识工作台](../meaning/index.html)，第四轮当前进度见[阶段计划](../plan.md)。
