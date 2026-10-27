# 第二轮：三条样板路线

2026-10-25。已形成 **25 个指定主张审校节点、16 条分类型关系、25份文献笔记、三组时间线与分支图**。这是样板批次，不是三条路线已穷尽。全部来源均诚实记录为指定片段阅读；没有声称论文全文、证明附录或实验全部验证。

首选入口：[交互研究工作台](atlas.html)。

| 路线 | 节点 | 阅读入口 |
| --- | --- | --- |
| 学习效率与泛化 | 9 | [路线叙事与两种图](learning-efficiency.md) |
| 控制论、强化学习与主动推断 | 8 | [路线叙事与两种图](cybernetics-rl.md) |
| 具身、生态与动力系统 | 8 | [路线叙事与两种图](embodied-dynamical.md) |

## 文献与证据

- [文献夹](literature.md)：逐篇阅读范围、版本、原文定位、批评和后续问题。
- [主张台账](claims.md)、[关系台账](relations.md)：作者报告、本站比较与独立验证分开。
- [三个教学算例](demos.md)：训练样例与外推、TD传播、条件信息与协同；均不冒称论文复现。
- [审校修订与异议](review.md)：版本变化、未选候选和下一轮优先问题。
- [验收记录](validation.md)：数据、图、浏览器与项目检查。

## 编辑与增补

本轮唯一编辑源是 [research.json](research.json)。节点复用第一轮事件ID，新增来源有独立ID。第一轮 [atlas.json](../atlas.json)作为历史基线保存；来源现在读过，不会把第一轮当时的状态改为已读。

```sh
node scripts/intelligence-history-demos.mjs
node scripts/generate-intelligence-history-round-two.mjs
node scripts/intelligence-history-demos.mjs --check
node scripts/generate-intelligence-history-round-two.mjs --check
```

新增时补齐事件、实际阅读范围、带定位的主张和边界；再按证据添加关系。图、卡片与台账同步重生成。Mermaid与SVG是可编辑源图；长期维护应修改研究数据，避免重生成覆盖手工图形修改。

本轮延续项目研究工程方法：问题→机制与前提→证据→边界与批评→下一问。专著按章阅读，不以下载数冒充阅读数；PDF仅临时用于阅读，笔记提供公开入口及可用指纹。

[长期方案](../../intelligence-history-plan.md) · [第一轮底稿](../round-one.md)
