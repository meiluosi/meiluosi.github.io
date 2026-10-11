# 站点使用与维护

本文整理项目 README 中的开发、配置与写作说明。访客入口见 [项目首页](../README.md)，设计与工程细节见 [架构说明](architecture.md)。

## 开发命令

使用 pnpm；`preinstall` 会检查包管理器。当前 [GitHub Pages 工作流](../.github/workflows/deploy.yml) 使用 Node.js 22 和 pnpm 9.14.4。

| 命令 | 用途 |
| --- | --- |
| `pnpm install` | 安装依赖 |
| `pnpm dev` / `pnpm start` | 启动开发服务器，默认端口 4321 |
| `pnpm check` | Astro 诊断 |
| `pnpm type-check` | TypeScript 检查，不生成输出文件 |
| `pnpm build` | 生成资产、构建站点与搜索索引，输出到 `dist/` |
| `pnpm preview` | 预览生产构建 |
| `pnpm new-post my-article-slug` | 在 `src/content/posts/` 创建文章 |
| `pnpm format` | 用 Biome 格式化 `src/`，会修改文件 |
| `pnpm lint` | 用 Biome 检查并安全修复 `src/`，会修改文件 |

网站改动提交前运行 `pnpm check`、`pnpm type-check` 与 `pnpm build`。视觉或交互改动还需浏览器验证，见 [开发约定](../AGENTS.md)。避免因全目录格式化引入无关改动。

## 目录与配置

| 位置 | 用途 |
| --- | --- |
| [`src/pages/`](../src/pages/) | 页面路由 |
| [`src/layouts/`](../src/layouts/) 与 [`src/components/`](../src/components/) | 页面布局与复用组件 |
| [`src/content/`](../src/content/) | 中文文章、英文文章与特殊页面内容 |
| [`src/data/`](../src/data/) | 研究记录、阅读路线、文献与近期状态 |
| [`src/config/`](../src/config/README.md) 与 [`src/types/`](../src/types/) | 模块配置及对应类型 |
| [`src/styles/`](../src/styles/)、[`src/utils/`](../src/utils/)、[`src/plugins/`](../src/plugins/) | 样式、工具和 Markdown/HTML 插件 |
| [`src/assets/`](../src/assets/) 与 [`public/`](../public/) | 源资源与直接发布的静态资源 |
| [`scripts/`](../scripts/) | 文章脚手架与构建脚本 |
| [`projects/small-model-lab/`](../projects/small-model-lab/README.md) | 可独立运行的小模型训练项目 |

配置通过 `@/config` 统一导入，例如：

```ts
import { profileConfig, siteConfig } from "@/config";
```

| 配置 | 内容 |
| --- | --- |
| [`siteConfig.ts`](../src/config/siteConfig.ts) | 站点 URL、名称与文章页面选项 |
| [`profileConfig.ts`](../src/config/profileConfig.ts) | 名字、简介、头像与公开链接 |
| [`navBarConfig.ts`](../src/config/navBarConfig.ts) | 导航名称、链接与菜单 |
| [`sidebarConfig.ts`](../src/config/sidebarConfig.ts) | 侧边栏组件与显示策略 |
| [`commentConfig.ts`](../src/config/commentConfig.ts) | 评论服务 |
| [`fontConfig.ts`](../src/config/fontConfig.ts) | 字体来源与字体栈 |
| [`licenseConfig.ts`](../src/config/licenseConfig.ts) | 文章许可声明 |
| [`analyticsConfig.ts`](../src/config/analyticsConfig.ts) | 统计分析服务 |
| [`expressiveCodeConfig.ts`](../src/config/expressiveCodeConfig.ts) | 代码块主题、行号、折叠与语言标记 |

完整功能配置包括壁纸、音乐、看板娘、公告、赞助和特效，见 [配置索引](../src/config/README.md)。配置项存在不代表所有页面都启用该功能。品牌页面的配色统一由 [`brand-theme.css`](../src/styles/brand-theme.css) 管理；调整品牌颜色时先检查这里的共享变量。

新增导航应根据 [`navBarConfig.ts`](../src/config/navBarConfig.ts) 的现有结构添加，并核对中英文名称与目标 URL。密钥放在部署平台环境变量中，不写入仓库配置。

## 写作与双语内容

```bash
pnpm new-post my-article-slug
```

脚手架在 `src/content/posts/` 中创建带 frontmatter 的 Markdown 文件。文章字段以 [`src/content.config.ts`](../src/content.config.ts) 为准，例如：

```yaml
---
title: 文章标题
published: 2026-10-11
updated: 2026-10-11
description: 一句话说明文章讨论的问题
image: ""
tags: [强化学习, AlphaZero]
category: 技术研究
format: technical
draft: false
pinned: false
comment: true
lang: zh_CN
---
```

`format` 可选 `plan`、`code-example`、`application`、`technical`、`experiment` 或 `reflection`。密码保护字段为 `password` 和 `passwordHint`，默认留空。英文文章放在 `src/content/en-posts/`，以 `sourceId` 对应中文文章 ID，以 `slug` 指定英文 URL；可参考 [现有双语文章](../src/content/en-posts/local-1b-model-workflow.md)。页面界面文案与文章正文翻译分别维护。

支持 Markdown / MDX、KaTeX、Mermaid、PlantUML、代码高亮、仓库卡片与图片网格。下面是写作语法示例；网格中的图片路径需替换为文章自己的图片。

````md
<!-- KaTeX 数学公式 -->
$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

<!-- Mermaid 图表 -->
```mermaid
graph TD
    A[开始] --> B{判断}
    B -->|是| C[执行]
    B -->|否| D[结束]
```

<!-- GitHub 仓库卡片 -->
:github[https://github.com/meiluosi/meiluosi.github.io]

<!-- 图片网格 -->
:::grid
![图一](image1.png)
![图二](image2.png)
:::
````

## 构建与部署

[`package.json`](../package.json) 中的 `pnpm build` 依次运行：

1. `generate-icons.js`：生成 SVG 图标常量。
2. `generate-lqips.ts`：生成图片占位符。
3. `generate-summaries.ts`：生成文章摘要。
4. `generate-knowledge-graph.ts`：生成知识图谱数据。
5. `astro build`：构建页面。
6. `subset-fonts.ts`：生成字体子集。
7. `pagefind --site dist`：生成站内搜索索引。

构建会生成或更新资产，提交前检查 `git diff`，尤其是图标、LQIP 与知识图谱数据。`dist/` 是部署输出。

| 平台 | 仓库中的配置 |
| --- | --- |
| GitHub Pages | [部署工作流](../.github/workflows/deploy.yml) 在推送 `master` 或手动触发时构建并上传 `dist/` |
| Vercel | [`vercel.json`](../vercel.json) 指定安装、构建、输出目录与响应头 |
| Cloudflare | [`wrangler.jsonc`](../wrangler.jsonc) 指定静态资产目录；[`astro.config.mjs`](../astro.config.mjs) 在设置 `CF_WORKERS` 时启用 Cloudflare 适配器 |

切换部署平台时需核对平台环境与适配器配置；配置文件存在不代表该平台已部署或验收。站点使用的公开 URL 在 [`siteConfig.ts`](../src/config/siteConfig.ts) 中维护。
