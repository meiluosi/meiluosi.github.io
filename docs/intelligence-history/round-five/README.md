# 第五轮：路线演进时间线

[执行计划](plan.md) · [正式网站入口](http://127.0.0.1:4327/research/intelligence/timelines/) · [验收记录](validation.md)

沿20条路线阅读重要问题、方法、定位主张、限制和下一问。以学习效率、心理测量与心灵哲学三种不同研究类型作为样板，再按相同规则推广。时间先后不自动意味着历史影响；年代未定的古典文本单列。

第二轮至第三轮历史索引共150条批次节点记录、144个去重ID。网站在20条路线中展示169条路线节点记录，同一ID可在多条路线出现。每条展示都指向索引选择的相应原批次，来源与阅读范围不跨批次合并；当前索引中的书目更正以显式注释展示，不改历史快照。

## 阅读与维护

- 从[正式研究总览](http://127.0.0.1:4327/research/intelligence/)或任意路线介绍进入对应时间线，点按图中节点进入详情与原文定位。
- 每页提供SVG、Mermaid和路线数据下载。图形导出可编辑；长期维护应编辑对应批次的 `research.json`，更新 `research-index.json`，再运行 `node scripts/generate-intelligence-timelines.mjs`。直接改生成文件会在下次重建时被覆盖。
- 本阶段没有新增原典阅读、原始数据复算或新的历史影响关系。各路线仍保留已有缺口、来源未读范围以及首轮未映射候选。
- 内容一致性由 `scripts/audit-intelligence-timelines.py` 核对；浏览器交互由 `scripts/verify-intelligence-timelines.mjs` 验证。详情见[验收记录](validation.md)。
