# 📚 项目文档

> 枫语个人品牌网站的规划、调研与开发文档。执行优先级见路线图；旧博客功能点仅作历史备选。

## 文档索引

[第五卷审校记录](./reasoning-rl-research.md)：强化学习问题建模、教学算例和页面验收。

[第六卷审校记录](./tool-training-research.md)：工具交互轨迹、环境契约与页面验收。

[第七卷审校记录](./evaluation-reproducibility-research.md)：实验设计、评测证据与工程复现。

| 文档 | 说明 |
|------|------|
| [intelligence-routes-specification.md](./intelligence-routes-specification.md) | 🧭 「智能是什么」20 大探索路线家族与 20 个全量交互认知透镜架构规范 |
| [intelligence-history-plan.md](./intelligence-history-plan.md) | 智能问题演进史：研究工程方法、四轮交付与证据标准 |
| [intelligence-history/round-two/README.md](./intelligence-history/round-two/README.md) | 第二轮成果：三条样板、25 个审校节点、文献笔记与可编辑图 |
| [intelligence-history/round-three-a/README.md](./intelligence-history/round-three-a/README.md) | 第三轮A批已验收：四条路线、样板补链、问题卡、版本对照及正式复现 |
| [intelligence-history/round-three-b/README.md](./intelligence-history/round-three-b/README.md) | 第三轮B批已验收：三条能力评价路线、问题卡、A批回访及合成核验 |
| [intelligence-history/index.html](./intelligence-history/index.html) | 20路线统一研究入口：144个去重节点、15张问题卡、11项回访 |
| [intelligence-history/round-six/README.md](./intelligence-history/round-six/README.md) | 第六轮：24项缺口排序、4项原典片段补读、2项竞争解释及结论回访 |
| [intelligence-history/round-five/README.md](./intelligence-history/round-five/README.md) | 第五轮：20条路线的演进时间线、原批次来源追溯和可编辑图 |
| [intelligence-history/round-four/integration/README.md](./intelligence-history/round-four/integration/README.md) | 第四轮第四阶段：18条结论共同复核、31条来源统一、正式网站原生入口与9项设计排序 |
| [intelligence-history/round-four/meaning/README.md](./intelligence-history/round-four/meaning/README.md) | 第四轮第三阶段已验收：意义与意识，12份选读、40条主张、6条结论、预测与版本台账、3组可编辑图及11项精确核验 |
| [intelligence-history/round-four/boundaries/README.md](./intelligence-history/round-four/boundaries/README.md) | 第四轮第二阶段已验收：系统边界与发展，9份选读、6条结论、三组可编辑图和8项精确核验 |
| [intelligence-history/round-four/README.md](./intelligence-history/round-four/README.md) | 第四轮第一阶段已验收：比较框架、6条条件性结论、10份来源记录、两组可编辑图及方法核验 |
| [intelligence-history/cde-validation.md](./intelligence-history/cde-validation.md) | C—D—E已验收：10条路线、66条节点记录、三项合成核验 |
| [intelligence-history/coverage.md](./intelligence-history/coverage.md) | 全局覆盖、原典／版本缺口与跨批节点目录 |
| [intelligence-history/questions-index.md](./intelligence-history/questions-index.md) | 共同问题、证据比较和回访索引 |
| [intelligence-history/README.md](./intelligence-history/README.md) | 第一轮基线：20 条路线、候选时间线、可编辑图与文献工作台 |
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
| `scripts/generate-intelligence-history-round-two.mjs` | 从第二轮研究数据同步生成节点笔记、台账与六张图 | 修改 `round-two/research.json` 后运行；加 `--check` 验证一致性 |
| `scripts/generate-intelligence-history-round-three-a.mjs` | 从A批数据同步生成笔记、问题卡、SVG、Mermaid与交互工作台 | 修改 `round-three-a/research.json` 后运行；加 `--check` 核对引用、基线与生成一致性 |
| `scripts/generate-intelligence-history-round-three-b.mjs` | 同步生成B批时间线、关系图、评价条件与文献笔记 | 修改B批 `research.json` 后运行；加 `--check` 验证生成与前三批快照 |
| `scripts/generate-intelligence-history-round-three-cde.mjs` | 同步生成C／D／E批路线、证据、图与文献夹 | 参数`c`、`d`或`e`；加`--check`验证生成一致性 |
| `scripts/generate-intelligence-boundaries.mjs` | 生成系统边界综述、文献、三组图与交互工作台 | 修改boundaries/research.json后运行；`--check`核验基线与生成一致性 |
| `scripts/check-intelligence-boundaries.py` | 精确核验历史／支持、状态重置与反馈对照 | `--check`复核先存协议、16种条件及8项断言 |
| `scripts/generate-intelligence-meaning.mjs` | 生成意义与意识的综述、文献、预测／版本台账、图和工作台 | `--check`核对源数据、历史基线和37个生成文件 |
| `scripts/check-intelligence-meaning.py` | 11项预先协议下的有限模型精确检查 | `--check`核对保存结果，不检测真实意识 |
| `scripts/audit-intelligence-meaning.py` | 审计来源、结论、预测、链接、锚点和历史快照 | `--check`核对报告；需Python XML模块 |
| `scripts/verify-intelligence-meaning.mjs` | 浏览器核验图、精确交互、桌面／手机和导航 | 可配置Playwright、浏览器及本地预览地址 |
| `scripts/audit-intelligence-boundaries.py` | 审计第二阶段证据结构、路径、锚点与旧快照 | `--check`核对保存报告；需Python XML模块 |
| `scripts/verify-intelligence-boundaries.mjs` | 浏览器交互、精确显示、桌面／手机布局与截图核验 | 对本地预览运行；Playwright与浏览器位置可通过环境变量指定 |
| `scripts/generate-intelligence-history-round-four.mjs` | 同步生成第四轮综述、文献、结论审计、SVG／Mermaid和交互页 | 修改第四轮research.json后运行；`--check`核对历史基线、引用与生成一致性 |
| `scripts/check-intelligence-round-four.py` | 按预先协议精确核验观察等价、干预与电路反例 | `--check`核对结果、协议和实现指纹 |
| `scripts/audit-intelligence-history-round-four.py` | 核验第四轮和当前索引的本地路径、显式锚点及证据结构 | 无参数更新报告；`--check`核对已保存报告。需Python的XML模块可用 |
| `scripts/generate-intelligence-history-index.mjs` | 汇总20路线、去重节点、共同问题及缺口 | 修改任意批次后运行；加`--check`核对全局索引 |
| `scripts/check-intelligence-cde.py` | 精确计算三项合成核验，记录协议与实现指纹 | 参数`c`、`d`或`e`；`--check`核对结果，`--plot`生成SVG |
| `scripts/audit-intelligence-history-cde.py` | 检查CDE及全局索引的本地链接、SVG与证据字段 | 无参数更新审计记录；`--check`验证既存记录一致 |
| `scripts/check-intelligence-evaluation.py` | 精确核验观察等价与均值解释的合成例子 | `--check` 核对结果；`--plot` 重建SVG，绘图需要Matplotlib与NumPy |
| `scripts/intelligence-history-demos.mjs` | 运行三个教学算例并核对解析结果 | 加 `--check` 检查结果是否一致；不计为论文复现 |
| `scripts/generate-intelligence-history.mjs` | 从同一份研究数据生成时间线、路线底稿与文献队列 | 修改 `docs/intelligence-history/atlas.json` 后运行；加 `--check` 检查引用和生成一致性 |

英文界面的本地化约定：`data-i18n-zh` / `data-i18n-en` 两条翻译路径都以 `data-i18n` 键作为选择器钩子，只有语言属性、没有 `data-i18n` 键的元素在客户端切换语言时不会跟着变。给这类元素补键时，纯文本用 `data-i18n` + 语言属性即可，含富文本的（如数字花园抽屉里的阶段列表）另加 `data-i18n-html="true"`，否则替换时会丢掉内部标签。

## 研究阅读体验的维护入口

| 内容 | 维护位置 | 约定 |
|------|----------|------|
| 智能问题演进史 | [全局索引](./intelligence-history/research-index.json)、[研究方案](./intelligence-history-plan.md) | 第一轮候选基线独立保留；本轮主张、阅读范围和关系分别核验；图与笔记同步生成 |
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

综合研究维护：`node scripts/generate-intelligence-integration.mjs` 生成修订文档及网站下载资源；`--check` 验证93份保留材料与68份生成结果。网站入口由 `src/pages/research/intelligence/` 读取研究数据与修订层。

路线时间线维护：更新相应批次 `research.json` 并重建当前研究索引，再运行 `node scripts/generate-intelligence-timelines.mjs`；`--check` 核验20条路线的61项生成结果。`scripts/audit-intelligence-timelines.py` 检查原批次与网站内容的一致性。
