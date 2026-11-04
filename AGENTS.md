# Repository Guidelines

## Project Structure & Module Organization

This is an Astro 7 site with Svelte islands and TypeScript configuration. Main source code lives in `src/`: routes in `src/pages`, layouts in `src/layouts`, reusable UI in `src/components`, styles in `src/styles`, content in `src/content`, helpers in `src/utils`, and Markdown/HTML plugins in `src/plugins`. Site configuration is split across `src/config` with matching type definitions in `src/types`; prefer imports from `@/config` when available. Static files served directly belong in `public`, source-managed images in `src/assets`, docs in `docs`, and automation in `scripts`.

## Build, Test, and Development Commands

Use `pnpm`; the `preinstall` script enforces it.

- `pnpm dev` or `pnpm start`: run the local Astro dev server.
- `pnpm check`: run Astro diagnostics.
- `pnpm type-check`: run TypeScript with `--noEmit`.
- `pnpm format`: format `src` with Biome.
- `pnpm lint`: run Biome checks and safe fixes on `src`.
- `pnpm build`: generate icons, LQIPs, the Astro build, font subsets, and Pagefind search output in `dist`.
- `pnpm preview`: preview the production build locally.
- `pnpm new-post`: scaffold a new content post.

## Coding Style & Naming Conventions

Biome is the formatter and linter. It uses tabs for indentation and double quotes for JavaScript/TypeScript strings. Keep Astro and Svelte components in `PascalCase` (`PostCard.astro`, `ArchivePanel.svelte`), config modules in `camelCase` ending with `Config.ts`, and utilities in descriptive kebab case such as `date-utils.ts`. Keep `src/types` aligned with `src/config`. Avoid unrelated formatting churn.

## Testing Guidelines

There is no dedicated unit-test framework configured. Before submitting changes, run `pnpm check`, `pnpm type-check`, and `pnpm build` for rendering, content, or generated asset work. For visual or interactive changes, verify with `pnpm dev` or `pnpm preview` and include screenshots in the PR. Name future tests near the feature they cover, using the local file name as the stem.

## Commit & Pull Request Guidelines

Use Conventional Commits, matching the current history: `feat: ...`, `fix: ...`, and `chore: ...`. Keep commits and PRs focused on one concern. PRs should include a concise summary, linked issues when relevant, validation commands run, and screenshots for UI changes. Discuss major features or design changes in an issue or discussion before implementation.

## Security & Configuration Tips

Do not commit secrets, tokens, or service keys in config files. Keep deployment-specific settings in the target platform environment, and review generated files such as `dist`, `src/constants/lqips.json`, and `src/constants/icons.ts` before committing them.

## Project Documentation

Feature planning, integration research, and roadmap are maintained in `docs/`:

| Document | Purpose |
|----------|---------|
| `docs/README.md` | Documentation index and quick links |
| `docs/features.md` | Feature ideas with priority and feasibility |
| `docs/integration-ideas.md` | Open-source project research and integration plans |
| `docs/architecture.md` | Project architecture, directory structure, build pipeline |
| `docs/roadmap.md` | Phased development roadmap with Gantt chart |

When implementing new features, consult `docs/features.md` and `docs/roadmap.md` first. Update `docs/roadmap.md` when completing a phase.

## Definition of Done

一项改动（功能、文章、研究轮次）只有同时满足以下条件才算完成：

1. `pnpm lint`（或 `npx biome check ./src`）无错误；`pnpm check`、`pnpm type-check` 通过。
2. 涉及渲染、内容或生成物时，`pnpm build` 通过，且 `pnpm audit:links` 与 `pnpm audit:math` 无新增死链与裸露伪公式。
3. 新增文章通过 `pnpm validate:frontmatter`；中英文是否成对按下文「内容原则」处理。
4. UI 或交互改动已在 `pnpm dev` / `pnpm preview` 中实际看过，PR 附截图。
5. 完成路线图阶段时已更新 `docs/roadmap.md`；有重大决策时已新增 ADR（`docs/decisions/`）。
6. 提交遵循 Conventional Commits，一个提交只做一件事；PR 的 CI（`.github/workflows/ci.yml`）为绿。

## 内容原则

- **证据优先**：事实性论断需有可追溯来源（论文、官方文档、可复现实验）；无法核实的内容明确标注为推测或观点。
- **数学符号与公式规范**：凡涉及数学变量、超参数、模型权重、概率分布与统计量（如 $\theta, \alpha, \gamma, \pi, \mathcal{L}, \mathbb{E}, W_q$），一律采用标准行内 LaTeX（`$\theta$`、`$\pi_\theta$`）或块级 LaTeX（`$$ ... $$`），严禁在正文中使用裸露 Unicode 希腊字母（如 `θ`、`π_θ`）、行内代码块或 ASCII 字符拼接伪公式（如 `s=√(3/16)`、`R_t = Σ ...`）。公式内若包含汉字注释必须使用 `\text{...}` 包裹以确保 KaTeX 零警告解析并激活站内跨公式变量联动高亮（`InteractiveMathEnhancer`）。
- **区分阅读深度**：只读了摘要或片段的来源，不得表述为“完整研究”；篇数、来源数不等于证据强度。
- **叙事节奏与结构化密度**：深度长文与专著章节遵循「**文首导读契约 + 四拍小节节奏**」（详见 `.agents/skills/editorial-narrative/SKILL.md`）。文首优先提供核心文献锚点（`corePapers`）、阅读收获清单（`takeaways`）、适用读者（`targetAudience`）与时效审校说明（`verifiedNote`）；小节内部避免连续超过三段的大块纯文字铺陈，按「① 抛出真问题或朴素基线失效处 → ② 3~4 列结构化对照表（机制/作用/缺失或退化表现/真实案例） → ③ GitHub Alert（`[!IMPORTANT]` / `[!NOTE]`）一句话收束核心矛盾或机制公式 → ④ 架构图、算例或交互组件核验」推进。
- **不改写已验收结论**：已发布或已验收的研究结论需要变更时，保留原结论并追加勘误/修订说明，不静默覆盖。
- **生成物不手改**：由 `scripts/` 生成的文件，改源数据或生成器，再重新生成。
- **冻结脚本**：`scripts/` 中 `intelligence-*` 等研究轮次脚本与文档中的哈希、复现命令绑定，不移动、不重命名、不修改（详见 `scripts/README.md`）。
- **双语**：站点提供中英文界面；新增长期文章优先中英成对，若暂缺其一，在 frontmatter 或文档中注明，避免中文文案泄漏到英文界面。
- **一次性脚本**：新增的一次性脚本需在文件头写明输入、输出、是否写入 `docs/`，验收后登记并冻结，不复制旧脚本改名。

## 部署与决策记录

部署主线为 GitHub Pages（`.github/workflows/deploy.yml`），`vercel.json` 与 `wrangler.jsonc` 仅为备用配置，详见 `docs/decisions/0001-deployment-mainline.md`。CI 与部署一律使用 `pnpm install --frozen-lockfile`，pnpm 版本以 `package.json` 的 `packageManager` 为准。
