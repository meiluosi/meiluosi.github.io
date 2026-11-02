# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。提交遵循 Conventional Commits，
`package.json` 的 `version` 在有用户可见变化的发布时更新，并在此登记。

## [Unreleased]

### Added
- `.github/workflows/ci.yml`：PR / 非 master 分支检查（Biome、frontmatter、`astro check`、类型检查、构建、站内链接审计）。
- `scripts/README.md`：脚本分类（长期维护 vs 冻结的研究轮次脚本）。
- `docs/decisions/`：决策记录（ADR），首份为 0001 部署主线。
- `AGENTS.md`：Definition of Done、内容原则、部署与决策记录说明。
- `.github/workflows/perf-baseline.yml` + `lighthouserc.json`：每周/手动的 Lighthouse 与包体基线，只记录不拦截。
- `scripts/report-bundle-size.js`（`pnpm report:bundle`）、`scripts/report-missing-translations.js`（`pnpm report:translations`）。
- `src/utils/frame-loop.ts`：仅在可见时运行的逐帧循环。

### Changed
- `deploy.yml` 改用 `pnpm install --frozen-lockfile`，pnpm 版本改由 `packageManager` 字段决定。
- `MarginGadget3D`、`ScrollytellingTransformer` 的 canvas 循环在离屏/标签页隐藏时暂停；后者尊重 `prefers-reduced-motion`。
- 文章页底部的 `Mindmap` 与相关文章组件由 `client:load` 改为 `client:visible`。
- Markdown 正文图片默认 `loading="lazy" decoding="async"`（`rehype-figure`）。
- Strandbeest 两张 GIF 换成等帧等时序的动画 WebP（约 2.6 MB → 1.3 MB），删除原 GIF；`docs/validation/series-final-2026-10-23/build-manifest.json` 作为历史快照保留不改。

## 6.13.5 之前（按提交历史回溯，未逐版本拆分）

### 2026-10-28
- 20 条智能探索路线、Living Books、3D 数字花园 2.0。
- 智能研究轮次：整合进站点、证据关联时间线、缺口与反方回访、群体智能证据审计、问题地图综合。
- Strandbeest 第 5 部分（风力证据与负结果报告，中英）。

### 2026-10-25
- 七卷 LLM 训练研究系列完成。

### 2026-10-22
- 打印样式、英文 RSS、响应式正文模式、代码沙盒。
- Strandbeest 系列发布；essays 索引页；35 篇英文概览回填；偏好卷补全。
- 新增 frontmatter 校验与站内链接审计脚本。
- 修复中文界面文案泄漏到英文界面。

### 2026-10-21
- Transformer 前向实验室与滚动叙事；CFR / DPO / RoPE 实验室；数字花园知识宇宙。
- 果蝇连接组系列（11 篇）。
- 重构：精简布局、清理死资源。

### 2026-10-18
- 移除侧边栏子系统，清理无用 i18n 键与依赖。

更早的历史见 `git log` 与 `docs/roadmap.md`。
