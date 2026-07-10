# 📚 项目文档

> 枫语博客的功能规划、调研与开发文档。所有值得实现的想法和开源项目集成方案都维护在这里。

## 文档索引

| 文档 | 说明 |
|------|------|
| [features.md](./features.md) | 🎯 功能创意清单 — 按优先级排列的所有功能想法，含可行性评估 |
| [integration-ideas.md](./integration-ideas.md) | 🔬 开源项目调研 — 可集成的开源项目详情与集成方案 |
| [architecture.md](./architecture.md) | 🏗️ 项目架构说明 — 技术栈、目录结构、集成模式 |
| [roadmap.md](./roadmap.md) | 🗺️ 开发路线图 — 分阶段的实施计划 |

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