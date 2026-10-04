# 📚 项目文档

> 枫语个人品牌网站的规划、调研与开发文档。执行优先级见路线图；旧博客功能点仅作历史备选。

## 文档索引

| 文档 | 说明 |
|------|------|
| [small-model-research-studio.md](./small-model-research-studio.md) | 研究岛 2.0、旗舰全链路文章与独立训练项目规划 |
| [studio-validation.md](./studio-validation.md) | 本轮网站交互、构建、包体和性能边界核验 |
| [small-model-training-validation.md](./small-model-training-validation.md) | tiny 全链路实测与正式训练的边界 |
| [features.md](./features.md) | 🎯 功能创意清单 — 当前方向与历史备选功能 |
| [integration-ideas.md](./integration-ideas.md) | 🔬 开源项目调研 — 可集成的开源项目详情与集成方案 |
| [architecture.md](./architecture.md) | 🏗️ 项目架构说明 — 技术栈、目录结构、集成模式 |
| [roadmap.md](./roadmap.md) | 🗺️ 开发路线图 — 分阶段的实施计划 |
| [case-study-evidence.md](./case-study-evidence.md) | 🧾 研究档案证据 — 项目事实、原文依据与待补充细节 |

## 研究阅读体验的维护入口

| 内容 | 维护位置 | 约定 |
|------|----------|------|
| RSI 导读 | `src/pages/rsi.astro` | 保留开放问题；个人判断需由作者提供 |
| 三条阅读路线 | `src/data/readingPaths.ts` | 使用发布文章的稳定 `sourceId`；失效条目会在构建时报告 |
| 文献架 | `src/data/library.ts` | 使用原始来源链接；导读不写成未经提供的个人读后评价 |
| MCTS 实验 | `src/utils/mcts.ts`、`MctsPlayground.svelte` | 种子固定时可复现；奖励统一从单智能体视角回传 |
| 概念预览 | `src/data/concepts.ts`、`ConceptPreview.astro` | 双语短定义与继续阅读目标；不侵入代码、公式和现有链接 |
| 反向链接 | `src/utils/backlinks.ts`、`ArticleConnections.astro` | 构建时解析已发布正文的 Markdown 链接；不将主题相似当成引用 |
| 阅读收藏 | `src/utils/reading-storage.ts`、`SaveNote.svelte`、`SavedReading.svelte` | 只保存文章 ID；中英文共用；本地存储键 `feng-yu-reading-v1` |

新增研究岛 `/world/` 与旗舰工作台 `/lab/model-pipeline/`；可独立复制发布的训练骨架位于 [`projects/small-model-lab/`](../projects/small-model-lab/README.md)，实际运行证据见 [训练验证记录](./small-model-training-validation.md)。

入口为 `/rsi/`、`/paths/`、`/library/`、`/lab/`、`/reading/`；共用 `ExploreLayout.astro`，公共导航与首页通过 `ExploreMenu.astro` 连接。

## 核心约束

- **托管平台**：GitHub Pages（纯静态）
- **无后端**：所有动态功能必须通过以下任一方式实现：
  - 🔨 **构建时预生成**（数据 → 静态 HTML，零运行时成本）
  - 🔗 **Serverless 代理**（Cloudflare Workers / Vercel Functions）
  - 🌐 **第三方嵌入式服务**（iframe / Web Component）
- **包管理器**：pnpm
- **框架**：Astro 7 + Svelte 5
- **格式化**：Biome（tab 缩进，双引号）

## 快速链接

- [项目 README](../README.md)
- [AGENTS.md](../AGENTS.md) — AI 编码助手指南
- [CLAUDE.md](../CLAUDE.md) — Claude Code 指南
