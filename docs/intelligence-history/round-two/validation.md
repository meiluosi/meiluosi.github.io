# 第二轮验收记录

2026-10-25。第二轮样板交付已完成：学习效率与泛化9个节点，控制论、强化学习与主动推断8个节点，具身、生态与动力系统8个节点。完成的是指定主张审校批次，不代表路线已穷尽或25份文献均已全文读完。

## 研究数据与生成

- 25个节点、25条带原文定位的核心主张、25份文献笔记；节点另附机制、前提、批评、例子和下一问。
- 16条关系：13条有原文定位，3条为明确标记的本站比较。事件与主张ID唯一，关系两端和来源可解析，方向符合所采用的年代。
- 21份本机阅读文件的SHA-256与笔记匹配；4份经网页正文或转录阅读，没有伪造文件指纹。实际读取范围逐篇保留，访问失败独立登记。
- 第一轮106个候选事件与来源数据指纹不变；第一轮生成一致性检查通过。
- 第二轮46个生成文件可从研究数据重建，包含三份路线叙事、六张SVG、六份Mermaid源图、文献笔记、证据台账和交互工作台。
- 六张SVG通过XML解析、节点数量和文字边界检查；内部文档链接检查通过。Mermaid文件保留为可编辑文本源，本轮浏览器渲染验收使用SVG，未运行独立Mermaid引擎。

可重跑：

```sh
node scripts/intelligence-history-demos.mjs --check
node scripts/generate-intelligence-history-round-two.mjs --check
node scripts/generate-intelligence-history.mjs --check
```

三个教学算例分别核对训练外推、确定性TD预测的解析目标及异或变量的精确信息量。均通过；不计为论文复现。

## 浏览器与图形

使用独立临时会话的Microsoft Edge，通过Playwright检查本机工作台。三条路线的时间线和分支图均已浏览并保存完整截图；所有25个目录节点打开正确详情。跨路线深链接、地址锚点变化、图中节点点击和键盘选择通过；390像素窄屏无页面横向溢出，宽图在图内滚动。未捕获页面脚本异常。

修复了初始化覆盖节点深链接、时间线行内空白不能点击，以及部分关系编号重叠；最终截图和自动检查均来自修订后的图。

- [工作台截图](screenshots/workbench.png)
- [学习效率时间线](screenshots/learning-efficiency-timeline.png) · [分支图](screenshots/learning-efficiency-branches.png)
- [控制论与RL时间线](screenshots/cybernetics-rl-timeline.png) · [分支图](screenshots/cybernetics-rl-branches.png)
- [具身路线时间线](screenshots/embodied-dynamical-timeline.png) · [分支图](screenshots/embodied-dynamical-branches.png)
- [窄屏节点详情](screenshots/mobile-detail.png)

## 项目检查

| 检查 | 结果 |
| --- | --- |
| `pnpm check` | 188个文件，0错误、0警告、0提示；内容同步另有中文数学模式提示。 |
| `pnpm type-check` | 通过。 |
| `AI_SUMMARY_API_KEY= pnpm build` | 通过，243页；关闭本次构建的外部摘要调用。 |
| `pnpm preview` | 已启动并检查生产路线页和文献架；研究工作台在独立文档预览中验收。 |
| 构建生成资源 | `icons.ts`与`lqips.json`构建前后指纹一致。 |

构建仍提示部分包超过500kB、静态robots.txt覆盖同名路由及npm配置弃用；没有因此失败。本轮成果位于项目文档中，不自动发布为博客解读。已有网站改动保持原样。

机器记录：[数据与构建](evidence/validation.json)、[浏览器检查](evidence/browser-checks.json)、[生产预览](evidence/preview-checks.json)、[算例结果](evidence/demos.json)。研究内容的限制与下轮缺口见[审校记录](review.md)。
