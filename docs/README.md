# 📚 项目文档

> 枫语个人品牌网站的规划、调研与开发文档。执行优先级见路线图；旧博客功能点仅作历史备选。

## 文档索引

| 文档 | 说明 |
|------|------|
| [site-editorial-cleanup.md](./site-editorial-cleanup.md) | 标题表达、导航精简与交互文献卡片 |
| [article-concept-network.md](./article-concept-network.md) | 文章多层概念网络、来源关联与评论关闭记录 |
| [pretraining-data-research.md](./pretraining-data-research.md) | 大模型训练研究笔记第一卷：六篇文章、来源台账与验收记录 |
| [humanities-and-technology.md](./humanities-and-technology.md) | 人文＋科技方向、编辑原则、实现与验收 |
| [small-model-research-studio.md](./small-model-research-studio.md) | 研究岛 2.0、旗舰全链路文章与独立训练项目规划 |
| [studio-validation.md](./studio-validation.md) | 本轮网站交互、构建、包体和性能边界核验 |
| [small-model-training-validation.md](./small-model-training-validation.md) | tiny 全链路实测与正式训练的边界 |
| [features.md](./features.md) | 🎯 功能创意清单 — 当前方向与历史备选功能 |
| [integration-ideas.md](./integration-ideas.md) | 🔬 开源项目调研 — 可集成的开源项目详情与集成方案 |
| [architecture.md](./architecture.md) | 🏗️ 项目架构说明 — 技术栈、目录结构、集成模式 |
| [roadmap.md](./roadmap.md) | 🗺️ 开发路线图 — 分阶段的实施计划 |
| [design-archetypes.md](./design-archetypes.md) | 🏛️ 顶级个人学术站四大流派与设计哲学（解释性计算、数字花园、微交互工艺、长卷叙事） |
| [case-study-evidence.md](./case-study-evidence.md) | 🧾 研究档案证据 — 项目事实、原文依据与待补充细节 |

## 构建期校验

| 脚本 | 作用 | 何时跑 |
|------|------|--------|
| `scripts/validate-frontmatter.js` | 拦下未加引号却含冒号的 YAML 标量（`title` / `description` 尤其容易踩） | 已接入 `pnpm build`，在 `astro build` 之前执行；也可单独 `pnpm validate:frontmatter` |
| `scripts/audit-internal-links.js` | 扫描 `dist/` 里的站内链接，报告失效目标 | 构建后 `pnpm audit:links` |

英文界面的本地化约定：`data-i18n-zh` / `data-i18n-en` 两条翻译路径都以 `data-i18n` 键作为选择器钩子，只有语言属性、没有 `data-i18n` 键的元素在客户端切换语言时不会跟着变。给这类元素补键时，纯文本用 `data-i18n` + 语言属性即可，含富文本的（如数字花园抽屉里的阶段列表）另加 `data-i18n-html="true"`，否则替换时会丢掉内部标签。

## 研究阅读体验的维护入口

| 内容 | 维护位置 | 约定 |
|------|----------|------|
| RSI 导读 | `src/pages/rsi.astro` | 保留开放问题；个人判断需由作者提供 |
| 文献架 | `src/data/library.ts`、`src/pages/library.astro` | 按研究主题筛选，展开交互卡片查阅原始来源 |
| MCTS 实验 | `src/utils/mcts.ts`、`MctsPlayground.svelte` | 种子固定时可复现；奖励统一从单智能体视角回传 |
| 概念预览 | `src/data/concepts.ts`、`ConceptPreview.astro` | 双语短定义与继续阅读目标；不侵入代码、公式和现有链接 |

新增研究岛 `/world/` 与旗舰工作台 `/lab/model-pipeline/`；可独立复制发布的训练骨架位于 [`projects/small-model-lab/`](../projects/small-model-lab/README.md)，实际运行证据见 [训练验证记录](./small-model-training-validation.md)。

入口为 `/rsi/`、`/library/`、`/lab/`、`/world/`；共用 `ExploreLayout.astro`，公共导航与首页通过 `ExploreMenu.astro` 连接。

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

- [智能手记与文献解读](./rsi-reading-notebook.md)：RSI／AGI 互动叙事、逐篇解读标准与 dot 文献清单状态。

- [大模型训练研究笔记栏目规划](./llm-training-series-plan.md)：第一卷改写原则与第二至第七卷主题、篇目及交付目标。

- [七卷执行记录](./llm-training-series-progress.md)：正文、来源核对和页面验收的实际进展。

- [第二卷审校记录](./pretraining-foundations-research.md)：一手来源、实现边界、教学算例与整卷页面验收。

- [第三卷 SFT 审校记录](./sft-research.md)：任务示范、模板与监督范围的来源、算例和验收。
