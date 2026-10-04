# 🏗️ 项目架构说明

> 枫语 RSI / AGI 个人品牌站的技术架构、目录结构、构建管线与集成模式。文章仍由 Astro Content Collections 管理；首页、Now、研究档案和案例页是站点的主要品牌入口。

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
│   ├── case-study-evidence.md   # 案例事实依据与待补证据
│   ├── architecture.md          # 架构说明（本文档）
│   └── roadmap.md               # 开发路线图
│
├── src/                         # 源码
│   ├── pages/                   # 页面路由（基于文件路由）
│   │   ├── index.astro          # RSI → AGI 品牌首页、Now 与精选档案
│   │   ├── about.astro          # 个人介绍与研究问题
│   │   ├── notes.astro          # 笔记入口、分类和形式筛选
│   │   ├── work.astro           # 研究方向与研究档案
│   │   ├── work/[slug].astro    # 来源型研究案例详情
│   │   ├── contact.astro        # 联系页
│   │   ├── posts/               # 中文文章详情页
│   │   ├── en/posts/            # 英文文章详情页/概览
│   │   ├── [...page].astro      # 旧文章分页入口
│   │   ├── archive.astro        # 旧归档入口
│   │   ├── categories/, tags/   # 旧分类/标签入口
│   │   ├── graph.astro          # 旧知识图谱入口
│   │   ├── search.astro         # 站内搜索
│   │   ├── og/                  # OG 图片生成
│   │   ├── 404.astro            # 404 页面
│   │   ├── rss.astro, rss.xml.ts # RSS 订阅
│   │   └── api/                 # API 端点
│   │
│   ├── components/              # 组件库
│   │   ├── analytics/           # 分析组件（Google Analytics 等）
│   │   ├── comment/             # 评论组件（Giscus）
│   │   ├── common/              # 通用组件
│   │   ├── controls/            # 控件组件（主题切换等）
│   │   ├── features/            # 功能组件（3D 研究地图、语言切换等）
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
│   │   ├── brand.ts             # 品牌页面中英文文案
│   │   ├── i18nKey.ts           # 翻译键枚举
│   │   ├── translation.ts       # 翻译查找
│   │   └── languages/           # 语言文件（zh_CN, en, ja, ru, zh_TW）
│   │
│   ├── content/                 # 内容集合
│   │   ├── posts/               # 博客文章（.md / .mdx）
│   │   ├── en-posts/            # 与中文文章配对的英文译文或概览
│   │   └── spec/                # 特殊页面（about, guestbook）
│   │
│   ├── data/                    # 品牌站可维护的叙事数据
│   │   ├── now.ts               # 首页当前主张与更新日期
│   │   ├── researchThreads.ts   # 三条研究线、3D 站点和来源文章映射
│   │   ├── researchArchive.ts   # 精选项目摘要、状态与文章匹配
│   │   └── caseStudies.ts       # 案例详情页的双语内容与证据边界
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
│   └── assets/                  # CSS/JS/字体
│
├── scripts/                     # 构建脚本
│   ├── generate-icons.js        # 图标生成
│   ├── generate-lqips.ts        # LQIP 生成
│   ├── generate-summaries.ts    # 可选的构建时文章摘要
│   ├── generate-knowledge-graph.ts # 知识图谱数据生成
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

## 品牌内容维护

- 更新首页当前主张、简述和日期时，编辑 `src/data/now.ts`。
- 更新模型系统、反馈学习、因果评估的研究问题、Work 方向文案、3D 场景名称或来源链接时，编辑 `src/data/researchThreads.ts`。首页 Now 区块、`ExploreWorld.svelte` 与 Work 页方向锚点共用这份数据；双语方向说明和相关笔记检索词也在此维护。
- 更新精选档案的类型、摘要、状态或来源文章匹配时，编辑 `src/data/researchArchive.ts`。
- 更新案例详情的叙述时，编辑 `src/data/caseStudies.ts`。只写现有文章或用户确认的事实；计划配置、模拟数据、缺失的个人职责和未记录结果要继续标注边界。
- 新增案例时，同时在 `researchArchive.ts` 加 `caseSlug`，在 `caseStudies.ts` 加同名 `slug`；动态路由会按 `match` 找到对应中文文章并配对英文文章。

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
    ├── 3. scripts/generate-summaries.ts
    │       └── 有 AI_SUMMARY_API_KEY 时补充文章摘要；无 Key 时跳过
    │
    ├── 4. scripts/generate-knowledge-graph.ts
    │       └── 生成 src/constants/knowledge-graph.json
    │
    ├── 5. astro build
    │       ├── 解析 content collections
    │       ├── 渲染所有页面为静态 HTML
    │       ├── 打包 Svelte 组件为 JS
    │       └── 输出到 dist/
    │
    ├── 6. scripts/subset-fonts.ts
    │       └── 字体子集化，减小字体文件大小
    │
    └── 7. pagefind --site dist
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

**示例**：Giscus 评论

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

- **品牌配色（2026-09-30）**：用户选择 B「鼠尾草与桃色」。`src/styles/brand-theme.css` 统一定义纸色、正文、辅助文字、链接、桃色和分割线变量；`Layout.astro` 的 `data-brand-theme="sage"` 将品牌页面与中英文文章统一为浅色，旧本地深色偏好不覆盖这一品牌选择。3D 材质、灯光和驾驶在 `components/features/world/create-world.ts` 维护，`ExploreWorld.svelte` 负责静态预览、按需启动和内容控件，站点标注颜色由 `researchThreads.ts` 提供。分享卡片的 SVG 与 PNG 需同步更新。
- **品牌动效**：`BrandMotion.astro` 连接 `brand-motion.ts` 与 `brand-motion.css`，仅作用于 `data-brand-motion` 容器；标题用 `data-motion-intro`、章节用 `data-motion-reveal`、卡片反馈用 `data-motion-hover`。正文不加入逐段入场；减少动态效果、焦点、锚点和页面恢复均应保持可读。

- **组件命名**：Astro/Svelte 组件使用 `PascalCase`（如 `PostCard.astro`）
- **配置模块**：`camelCase` 结尾加 `Config`（如 `siteConfig.ts`）
- **工具函数**：kebab-case（如 `date-utils.ts`）
- **类型定义**：与 `src/config` 对应，保持同步
- **提交规范**：Conventional Commits（`feat:`, `fix:`, `chore:`）
- **格式化**：Biome 自动处理，不手动格式化


## RSI 研究工作室（2026-10-04）

首页 `StudioRoutes.astro` 连接三个实验与阅读路线，`TrainingBudgetPreview.svelte` 单独延迟水合。`ExploreWorld.svelte` 默认提供 SVG 静态岛，访客启动后才动态导入渲染器；`/world/` 以 immersive 模式空闲加载。路线规划在 `world/navigation.ts`，连续线段与海岸/台座碰撞规则通过957路线测试。

全链路工作台 `/lab/model-pipeline/` 使用独立 `ModelPipeline.svelte` 和可测试的 `utils/model-pipeline.ts`。浏览器只进行教学计算；真实 PyTorch 训练在可迁移的 `projects/small-model-lab/`。它具有自己的依赖、数据工具、配置、训练与推理CLI、CI和运行报告，不进入 Astro 客户端依赖图。模型检查点和数据缓存通过项目 `.gitignore` 排除；代码目录整体复制即可形成独立仓库。

项目示意图由 `ResearchVisual.astro` 共用；三个静态分享封面通过 `scripts/generate-research-covers.mjs` 离线生成，案例页使用 `Layout.astro` 的 `shareImage`。包体和浏览器验证范围见 [studio-validation.md](./studio-validation.md)，训练真实产物和边界见 [small-model-training-validation.md](./small-model-training-validation.md)。
