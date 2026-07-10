# 🏗️ 项目架构说明

> 枫语博客的技术架构、目录结构、构建管线与集成模式。

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 框架 | [Astro 7](https://astro.build) | 静态站点生成，岛屿架构 |
| 交互 | [Svelte 5](https://svelte.dev) | 响应式 UI 组件 |
| 样式 | Tailwind CSS + 自定义属性 | 主题色系统 |
| 格式化 | [Biome](https://biomejs.dev) | Tab 缩进，双引号 |
| 包管理 | pnpm | 强制使用 |
| 部署 | GitHub Pages / Vercel / Cloudflare | 支持多平台 |
| 搜索 | [Pagefind](https://pagefind.app) | 构建时静态索引 |
| 评论 | [Giscus](https://giscus.app) | GitHub Discussions 驱动 |
| 数学 | [KaTeX](https://katex.org) | LaTeX 渲染 |
| 图表 | Mermaid + PlantUML | 文本驱动图表 |
| 页面过渡 | [Swup.js](https://swup.js.org) | SPA 式动画 |
| 音乐 | [Meting](https://github.com/metowolf/MetingJS) | 多平台音乐 API |

---

## 目录结构

```
meiluosi.github.io/
├── docs/                        # 📚 项目文档（本目录）
│   ├── README.md                # 文档索引
│   ├── features.md              # 功能规划
│   ├── integration-ideas.md     # 开源项目调研
│   ├── architecture.md          # 架构说明（本文档）
│   └── roadmap.md               # 开发路线图
│
├── src/                         # 源码
│   ├── pages/                   # 页面路由（基于文件路由）
│   │   ├── index.astro          # 首页（文章列表）
│   │   ├── [...page].astro      # 文章分页
│   │   ├── archive.astro        # 归档页
│   │   ├── about.astro          # 关于页
│   │   ├── friends.astro        # 友链页
│   │   ├── guestbook.astro      # 留言板
│   │   ├── sponsor.astro        # 打赏页
│   │   ├── search.astro         # 搜索页
│   │   ├── gallery/             # 相册
│   │   ├── posts/               # 文章详情页
│   │   ├── categories/          # 分类页
│   │   ├── tags/                # 标签页
│   │   ├── og/                  # OG 图片生成
│   │   ├── 404.astro            # 404 页面
│   │   ├── rss.astro            # RSS 订阅
│   │   └── api/                 # API 端点
│   │
│   ├── components/              # 组件库
│   │   ├── analytics/           # 分析组件（Google Analytics 等）
│   │   ├── comment/             # 评论组件（Giscus）
│   │   ├── common/              # 通用组件
│   │   ├── controls/            # 控件组件（主题切换等）
│   │   ├── features/            # 功能组件（看板娘、特效等）
│   │   ├── layout/              # 布局组件
│   │   ├── misc/                # 杂项组件
│   │   ├── pages/               # 页面级组件
│   │   └── widget/              # 小部件
│   │
│   ├── config/                  # 配置文件（TypeScript）
│   │   ├── siteConfig.ts        # 站点核心配置
│   │   ├── navBarConfig.ts      # 导航栏配置
│   │   ├── pioConfig.ts         # 看板娘配置
│   │   ├── musicConfig.ts       # 音乐播放器配置
│   │   ├── commentConfig.ts     # 评论系统配置
│   │   ├── analyticsConfig.ts   # 分析配置
│   │   ├── fontConfig.ts        # 字体配置
│   │   ├── backgroundWallpaper.ts # 壁纸配置
│   │   └── ...                  # 其他配置
│   │
│   ├── types/                   # TypeScript 类型定义
│   │   ├── siteConfig.ts
│   │   ├── navBarConfig.ts
│   │   └── ...
│   │
│   ├── i18n/                    # 国际化
│   │   ├── i18nKey.ts           # 翻译键枚举
│   │   ├── translation.ts       # 翻译查找
│   │   └── languages/           # 语言文件（zh_CN, en, ja, ru, zh_TW）
│   │
│   ├── content/                 # 内容集合
│   │   ├── posts/               # 博客文章（.md / .mdx）
│   │   └── spec/                # 特殊页面（about, guestbook）
│   │
│   ├── utils/                   # 工具函数
│   │   ├── content-utils.ts     # 内容排序/过滤
│   │   ├── date-utils.ts        # 日期格式化
│   │   ├── crypto.ts            # 加密文章
│   │   └── ...
│   │
│   ├── plugins/                 # 自定义 remark/rehype 插件
│   │   ├── remark-reading-time.mjs
│   │   ├── rehype-mermaid.mjs
│   │   ├── rehype-plantuml.mjs
│   │   └── ...                  # 15 个插件
│   │
│   ├── styles/                  # 全局样式
│   ├── assets/                  # 源管理的图片/资源
│   ├── constants/               # 构建生成的常量（icons, LQIPs）
│   └── layouts/                 # 布局组件
│       ├── Layout.astro         # 基础 HTML 壳
│       └── MainGridLayout.astro # 主网格布局
│
├── public/                      # 静态文件（直接服务）
│   ├── favicon/                 # 网站图标
│   ├── pio/                     # 看板娘模型资源
│   ├── gallery/                 # 相册图片
│   └── assets/                  # CSS/JS/字体
│
├── scripts/                     # 构建脚本
│   ├── generate-icons.js        # 图标生成
│   ├── generate-lqips.ts        # LQIP 生成
│   ├── new-post.js              # 新建文章脚手架
│   ├── subset-fonts.ts          # 字体子集化
│   └── quarantine-bad-posts.mjs # 文章隔离
│
├── astro.config.mjs             # Astro 配置
├── svelte.config.js             # Svelte 配置
├── biome.json                   # Biome 配置
├── tsconfig.json                # TypeScript 配置
├── package.json                 # 项目依赖
├── vercel.json                  # Vercel 部署
└── wrangler.jsonc               # Cloudflare Workers 部署
```

---

## 构建管线

```
pnpm build
    │
    ├── 1. scripts/generate-icons.js
    │       └── 生成 src/constants/icons.ts
    │
    ├── 2. scripts/generate-lqips.ts
    │       └── 生成 src/constants/lqips.json（低质量图片占位符）
    │
    ├── 3. astro build
    │       ├── 解析 content collections
    │       ├── 渲染所有页面为静态 HTML
    │       ├── 打包 Svelte 组件为 JS
    │       └── 输出到 dist/
    │
    ├── 4. scripts/subset-fonts.ts
    │       └── 字体子集化，减小字体文件大小
    │
    └── 5. pagefind --site dist
            └── 生成全文搜索索引
```

---

## 集成模式

所有新功能必须遵循以下三种集成模式之一：

### 模式 A：构建时预生成 🔨

```
数据源 → 构建脚本 → JSON/HTML → 静态部署
```

| 特点 | 说明 |
|------|------|
| 安全 | ✅ API Key 只存在于 CI 环境 |
| 成本 | 仅构建时一次性调用 |
| 实时性 | ❌ 数据仅在构建时更新 |
| 适合 | AI 摘要、数据面板、图表生成 |

**示例**：AI 文章摘要、AKShare 数据面板、Mermaid 思维导图

### 模式 B：Serverless 代理 🔗

```
用户浏览器 → Cloudflare Worker → 第三方 API → 流式返回
```

| 特点 | 说明 |
|------|------|
| 安全 | ✅ Key 藏在 Worker 环境变量中 |
| 成本 | 按调用量计费，需加 rate limit |
| 实时性 | ✅ 实时响应 |
| 适合 | AI 对话、实时问答 |

**示例**：AI 对话看板娘、文章 RAG 问答

### 模式 C：第三方嵌入式服务 🌐

```
用户浏览器 → iframe / Web Component → 第三方平台
```

| 特点 | 说明 |
|------|------|
| 安全 | ✅ 第三方平台管理认证 |
| 成本 | 第三方平台免费额度 |
| 实时性 | ✅ 实时 |
| 适合 | 聊天机器人、知识库问答 |

**示例**：Dify AI 知识库问答、Giscus 评论

---

## 主题系统

```
CSS 自定义属性
├── --hue: 主题色相（0-360）
├── --primary: 根据 hue 计算的主色
├── --bg: 背景色
├── --text: 文字色
└── ... 更多变量

主题模式
├── light: 亮色
├── dark: 暗色
└── system: 跟随系统
```

---

## 路径别名

| 别名 | 映射路径 |
|------|---------|
| `@components/*` | `./src/components/*` |
| `@assets/*` | `./src/assets/*` |
| `@constants/*` | `./src/constants/*` |
| `@utils/*` | `./src/utils/*` |
| `@i18n/*` | `./src/i18n/*` |
| `@layouts/*` | `./src/layouts/*` |
| `@/*` | `./src/*` |

---

## 关键约定

- **组件命名**：Astro/Svelte 组件使用 `PascalCase`（如 `PostCard.astro`）
- **配置模块**：`camelCase` 结尾加 `Config`（如 `siteConfig.ts`）
- **工具函数**：kebab-case（如 `date-utils.ts`）
- **类型定义**：与 `src/config` 对应，保持同步
- **提交规范**：Conventional Commits（`feat:`, `fix:`, `chore:`）
- **格式化**：Biome 自动处理，不手动格式化