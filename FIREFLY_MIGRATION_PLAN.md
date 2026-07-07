# 🔥 Firefly 模板迁移执行计划

> **目标**：从 Jekyll/Hux Blog 迁移到 Astro/Firefly 模板，打造个人品牌技术博客
> **创建日期**：2025-07-20
> **当前状态**：⏳ 准备阶段

---

## 一、项目概览

### 1.1 基本信息

| 项目 | 详情 |
|------|------|
| **当前站点** | Jekyll + Hux Blog (2015) |
| **目标模板** | [Firefly](https://github.com/CuteLeaf/Firefly) — Astro 5 + Tailwind CSS 4 + Svelte + Material Design 3 |
| **作者** | 冯宇 (Feng Yu) |
| **域名** | https://meiluosi.github.io |
| **GitHub** | github.com/meiluosi |
| **定位** | AI 技术博客 — 强化学习 / 概率图模型 / 因果推断 / LLM |

### 1.2 Firefly 模板核心特性

- ✅ **Astro 5** — 静态站点生成，默认零 JS 输出
- ✅ **Tailwind CSS 4** — 原子化 CSS，Material Design 3 设计语言
- ✅ **Svelte** — 交互式组件（搜索、设置面板等）
- ✅ **KaTeX** — 数学公式渲染（关键：用户大量使用 `$$` 数学公式）
- ✅ **Giscus** — 基于 GitHub Discussions 的评论系统
- ✅ **Pagefind** — 客户端全文搜索
- ✅ **i18n** — 原生中文支持 (zh_CN)
- ✅ **双栏布局** — 左右侧边栏可配置
- ✅ **Expressive Code** — 代码高亮
- ✅ **明暗主题** — Light / Dark / System 三模式
- ✅ **瀑布流 / 网格 / 列表** — 多种文章布局

---

## 二、最终项目结构

```
meiluosi.github.io/                    # GitHub Pages 仓库根目录
├── .github/
│   └── workflows/
│       └── deploy.yml                 # GitHub Actions 自动部署
├── public/                            # 静态资源（构建后输出到 dist/）
│   ├── img/
│   │   ├── about_me.jpg               # 个人头像
│   │   └── post-bg-*.jpg              # 文章封面图
│   └── favicon.ico
├── src/
│   ├── components/                    # 组件（按需裁剪）
│   ├── config/                        # ⭐ 核心配置文件（见第三节）
│   ├── content/
│   │   ├── posts/                     # ⭐ 博客文章（30+ 篇迁移）
│   │   └── spec/                      # 特殊页面内容
│   ├── i18n/                          # 国际化
│   ├── layouts/                       # 布局模板
│   ├── pages/                         # 路由页面
│   ├── styles/                        # 全局样式
│   ├── types/                         # TypeScript 类型
│   └── utils/                         # 工具函数
├── astro.config.ts
├── tailwind.config.ts
├── tsconfig.json
└── package.json
```

---

## 三、ACG 元素清理清单 🔥

> 以下所有 ACG 元素需要在初始化后第一时间移除/禁用，共计 **21 项**。

### 3.1 配置文件清理（7 个文件）

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `src/config/pioConfig.ts` | **删除** | 看板娘/Live2D/Spine 模型配置 |
| 2 | `src/config/effectsConfig.ts` | **删除** | 樱花飘落特效配置 |
| 3 | `src/config/musicConfig.ts` | **删除** | 音乐播放器配置（崩坏：星穹铁道 OST） |
| 4 | `src/config/galleryConfig.ts` | **删除** | 相册配置（"可爱流萤"） |
| 5 | `src/config/backgroundWallpaper.ts` | **简化** | 移除 typewriter 文字（流萤台词）、视频背景、kenburns 等；只保留静态 banner 图或不显示 |
| 6 | `src/config/coverImageConfig.ts` | **修改** | 移除 ACG 随机图 API（dmoe.cc、uapis.cn），替换为 Unsplash 或本地图片 |
| 7 | `src/config/siteConfig.ts` | **修改** | 移除 keywords 中的 "ACGN"；禁用 bangumi/anime 相关配置 |

### 3.2 组件与页面删除（14 个文件/目录）

| # | 路径 | 操作 | 说明 |
|---|------|------|------|
| 8 | `src/components/features/SpineModel.astro` | **删除** | Spine 看板娘渲染器 |
| 9 | `src/components/features/Live2DWidget.astro` | **删除** | Live2D 看板娘组件 |
| 10 | `src/components/features/SakuraEffect.astro` | **删除** | 樱花飘落特效 |
| 11 | `src/components/features/MusicManager.astro` | **删除** | 全局音乐管理器 |
| 12 | `src/components/features/MusicPlayer.astro` | **删除** | 音乐播放器 UI |
| 13 | `src/components/features/BackgroundPlayer.astro` | **删除** | 背景视频播放器 |
| 14 | `src/components/features/TypewriterText.astro` | **删除** | 打字机文字动画 |
| 15 | `src/components/widget/SpineModel.astro` | **删除** | 侧边栏 Spine 组件 |
| 16 | `src/components/widget/Music.astro` | **删除** | 侧边栏音乐组件 |
| 17 | `src/components/common/PioMessageBox.astro` | **删除** | 看板娘消息气泡 |
| 18 | `src/components/pages/bangumi/` | **删除整个目录** | 番组计划页面组件 |
| 19 | `src/components/pages/anime/` | **删除整个目录** | 追番页面组件 |
| 20 | `src/components/pages/gallery/` | **删除整个目录** | 相册页面组件 |
| 21 | `src/pages/bangumi.astro` | **删除** | 番组计划路由 |
| 22 | `src/pages/anime.astro` | **删除** | 追番路由 |
| 23 | `src/pages/gallery/` | **删除整个目录** | 相册路由 |
| 24 | `src/types/bangumi.ts` | **删除** | Bangumi 类型定义 |
| 25 | `src/types/anime.ts` | **删除** | Anime 类型定义 |
| 26 | `src/components/controls/WallpaperSwitch.svelte` | **删除** | 壁纸模式切换按钮 |

### 3.3 依赖清理

```bash
# 移除 ACG 相关的 npm 包（在 package.json 中检查并移除）
npm uninstall l2d-widget    # Live2D 看板娘库
# 确认移除其他可能不需要的依赖
```

### 3.4 配置引用清理

在以下文件中移除对已删除组件的引用：
- `src/layouts/Layout.astro` — 移除 SpineModel、Live2DWidget、SakuraEffect、MusicManager、BackgroundPlayer 的 import 和渲染
- `src/layouts/MainGridLayout.astro` — 移除侧边栏音乐/Spine 组件引用
- `src/components/layout/SideBar.astro` — 移除 Music、SpineModel widget
- `src/config/index.ts` — 移除已删除配置的 export

---

## 四、个人品牌配置方案

### 4.1 `siteConfig.ts` — 站点核心配置

```ts
export const siteConfig = {
  title: "Feng Yu's Tech Blog",
  subTitle: "AI · 强化学习 · 因果推断 · 大语言模型",
  url: "https://meiluosi.github.io",
  keywords: [
    "人工智能", "强化学习", "因果推断", "概率图模型",
    "大语言模型", "机器学习", "深度学习", "数据科学",
    "DUCG", "AlphaZero", "LLM", "技术博客"
  ],
  // 主题色：保持蓝色调（专业学术风）
  themeColor: {
    light: "0 0% 100%",    // 可调整为更专业的色调
    dark: "222.2 84% 4.9%",
  },
  // 禁用不需要的功能
  page: {
    bangumi: false,
    anime: false,
    gallery: false,
    music: false,
  },
  // 其他配置...
};
```

### 4.2 `profileConfig.ts` — 个人信息

```ts
export const profileConfig = {
  avatar: "/img/about_me.jpg",
  name: "冯宇",
  bio: "AI 算法工程师 | 强化学习 & 因果推断研究者",
  socialLinks: [
    { name: "GitHub", url: "https://github.com/meiluosi", icon: "github" },
    { name: "Email", url: "mailto:1160337988@qq.com", icon: "email" },
  ],
};
```

### 4.3 `navBarConfig.ts` — 导航栏

```ts
export const navBarConfig = {
  links: [
    { name: "首页", url: "/", preset: "home" },
    { name: "文章", url: "/posts", preset: "posts" },
    {
      name: "系列",
      children: [
        { name: "强化学习", url: "/series/reinforcement-learning" },
        { name: "DUCG 理论", url: "/series/ducg-theory" },
        { name: "数据科学", url: "/series/data-science" },
        { name: "游戏 AI", url: "/series/game-ai" },
        { name: "LLM 实践", url: "/series/llm-practice" },
      ],
    },
    { name: "项目", url: "/projects" },
    { name: "论文", url: "/publications" },
    { name: "关于", url: "/about" },
    { name: "标签", url: "/tags" },
  ],
};
```

### 4.4 `sidebarConfig.ts` — 侧边栏

```ts
export const sidebarConfig = {
  position: "right",         // 单右边栏（专业风格）
  // 左侧组件列表
  left: {
    top: ["profile"],
    sticky: ["categories", "tags"],
  },
  // 右侧组件
  right: {
    top: ["siteInfo"],
    sticky: ["sidebarToc", "calendar"],
  },
};
```

### 4.5 `commentConfig.ts` — 评论系统

使用 **Giscus**（基于 GitHub Discussions，无需额外服务）：

```ts
export const commentConfig = {
  type: "giscus",
  giscus: {
    repo: "meiluosi/meiluosi.github.io",
    repoId: "<从 Giscus 官网获取>",
    category: "Announcements",
    categoryId: "<从 Giscus 官网获取>",
  },
};
```

### 4.6 `expressiveCodeConfig.ts` — 代码高亮

保持原有配置，支持暗色/亮色主题。

---

## 五、内容迁移方案

### 5.1 文章迁移（30+ 篇）

**Jekyll 格式 → Firefly (Astro) 格式转换：**

Jekyll 文件名格式：
```
_posts/2024-09-01-Transformer架构详解.md
```

Firefly 文件名格式（保持相同）：
```
src/content/posts/2024-09-01-Transformer架构详解.md
```

**Frontmatter 转换：**

Jekyll:
```yaml
---
layout: post
title: "Transformer架构详解"
subtitle: "从Attention机制到完整模型实现"
date: 2024-09-01
author: "Feng Yu"
header-img: "img/post-bg-deeplearning.jpg"
tags:
  - 深度学习
  - Transformer
  - NLP
---
```

Firefly:
```yaml
---
title: "Transformer架构详解"
description: "从Attention机制到完整模型实现"
published: 2024-09-01
updated: 2024-09-01
tags:
  - 深度学习
  - Transformer
  - NLP
category: 深度学习
lang: zh
---
```

### 5.2 待迁移文章清单

#### 强化学习系列 (8篇)
| 文件名 | 标题 |
|--------|------|
| `2024-6-30-深度Q网络案例实践.md` | 深度Q网络案例实践 |
| `2024-6-30-应用CFR实现德州扑克对战.md` | 应用CFR实现德州扑克对战 |
| `2024-6-30-Alpha Zero算法实现五子棋.md` | Alpha Zero算法实现五子棋 |
| `2024-06-09-强化学习笔记记录.md` | 强化学习笔记记录 |
| `2024-07-05-策略梯度方法详解.md` | 策略梯度方法详解 |
| `2024-07-12-Actor-Critic算法家族.md` | Actor-Critic算法家族 |
| `2024-07-28-模型基础强化学习.md` | 模型基础强化学习 |
| `2024-08-05-强化学习调参技巧.md` | 强化学习调参技巧 |

#### DUCG 系列 (8篇)
| 文件名 | 标题 |
|--------|------|
| `2024-6-30-动态不确定性因果图模型理论.md` | 动态不确定性因果图模型理论 |
| `2024-6-30-动态不确定性因果图模型理论在法律领域的应用.md` | DUCG在法律领域的应用 |
| `2024-6-30-动态不确定性因果图模型理论在金融领域的应用.md` | DUCG在金融领域的应用 |
| `2024-6-30-动态不确定性因果图模型理论在软件可靠性领域的应用.md` | DUCG在软件可靠性领域的应用 |
| `2024-6-30-动态不确定性因果图模型理论在网络安全领域的应用.md` | DUCG在网络安全领域的应用 |
| `2024-07-15-DUCG推理算法详解.md` | DUCG推理算法详解 |
| `2024-07-22-DUCG建模实战指南.md` | DUCG建模实战指南 |
| `2024-08-20-MCTS算法深度解析.md` | MCTS算法深度解析 |

#### LLM 系列 (10篇)
| 文件名 | 标题 |
|--------|------|
| `2024-09-01-Transformer架构详解.md` | Transformer架构详解 |
| `2024-09-15-LLM训练技术详解.md` | LLM训练技术详解 |
| `2024-09-22-高效微调技术LoRA与PEFT.md` | 高效微调技术LoRA与PEFT |
| `2024-09-29-提示工程最佳实践.md` | 提示工程最佳实践 |
| `2024-10-06-RAG检索增强生成实战.md` | RAG检索增强生成实战 |
| `2024-10-13-LLM-Agent开发指南.md` | LLM Agent开发指南 |
| `2024-10-20-多模态大模型技术.md` | 多模态大模型技术 |
| `2024-10-27-LLM推理优化与部署.md` | LLM推理优化与部署 |
| `2024-11-03-LLM安全与对齐.md` | LLM安全与对齐 |
| `2025-01-19-DeepSeek-R1论文精读.md` | DeepSeek R1论文精读 |

#### 数据科学 & 其他 (5篇)
| 文件名 | 标题 |
|--------|------|
| `2024-06-09-关于数据分析思考.md` | 关于数据分析思考 |
| `2024-6-30-数据分析实战—Dowhy酒店案例实战.md` | 数据分析实战—Dowhy酒店案例 |
| `2024-6-30-应用SHAP库实现机器学习可解释性.md` | 应用SHAP实现机器学习可解释性 |
| `2024-6-30-应用Ylearn框架实现因果推断.md` | 应用Ylearn框架实现因果推断 |
| `2024-8-15-统计学基础与假设检验.md` | 统计学基础与假设检验 |

#### 游戏 AI & 博弈论 (3篇)
| 文件名 | 标题 |
|--------|------|
| `2024-6-30-新型群体智能算法对比实验.md` | 新型群体智能算法对比实验 |
| `2024-09-03-多人博弈AI设计.md` | 多人博弈AI设计 |
| `2024-07-20-多智能体强化学习.md` | 多智能体强化学习 |

#### 其他 (1篇)
| 文件名 | 标题 |
|--------|------|
| `2024-9-16-R语言学习.md` | R语言学习 |

### 5.3 系列页面迁移

当前有 5 个系列页面（`series/` 目录），每个系列页面手动列出相关文章。Firefly 中可以：

1. **方案 A（推荐）**：使用标签/分类自动聚合 — 每篇文章打上对应标签，标签页自动展示系列文章
2. **方案 B**：创建自定义系列页面 — 类似当前 `series/*.html`，手动维护文章列表

建议采用**方案 A**，更符合 Astro 内容集合的设计理念。

### 5.4 页面迁移

| 当前页面 | Firefly 对应 | 操作 |
|----------|-------------|------|
| `pages/2-projects.html` | `src/content/spec/projects.md` | 新建，保存项目介绍 |
| `pages/3-publications.html` | `src/content/spec/publications.md` | 新建，保存论文列表 |
| `pages/4-tags.html` | `/tags` 路由（内置） | 无需迁移 |
| `index.html` | `/` 首页（内置） | 无需迁移 |
| `404.html` | `src/pages/404.astro`（内置） | 无需迁移 |
| `about` (sidebar) | `src/content/spec/about.md` | 新建个人介绍 |

---

## 六、开发阶段与节奏

### 阶段一：环境搭建 & 模板初始化 🚀 （预计 1-2 小时）

- [ ] **1.1** 克隆 Firefly 仓库到工作区
  ```bash
  git clone https://github.com/CuteLeaf/Firefly.git firefly-temp
  ```
- [ ] **1.2** 将 Firefly 源码合并到 `meiluosi.github.io` 仓库
- [ ] **1.3** 安装依赖 `npm install`
- [ ] **1.4** 启动开发服务器 `npm run dev`，验证模板正常运行
- [ ] **1.5** 配置 GitHub Actions 自动部署（已有 GitHub Pages 经验）

### 阶段二：ACG 元素清理 🧹 （预计 1-2 小时）

- [ ] **2.1** 删除 21 项 ACG 组件/页面/配置（见第三节清单）
- [ ] **2.2** 清理 `Layout.astro`、`MainGridLayout.astro`、`SideBar.astro` 中的引用
- [ ] **2.3** 清理 `config/index.ts` 中的配置导出
- [ ] **2.4** 清理 npm 依赖（l2d-widget 等）
- [ ] **2.5** 简化 `backgroundWallpaper.ts` — 只保留静态 banner 模式
- [ ] **2.6** 修改 `coverImageConfig.ts` — 替换 ACG 图源
- [ ] **2.7** 开发服务器验证 — 确保无报错、无 ACG 元素残留

### 阶段三：个人品牌配置 🎨 （预计 2-3 小时）

- [ ] **3.1** 配置 `siteConfig.ts` — 标题、副标题、关键词、主题色
- [ ] **3.2** 配置 `profileConfig.ts` — 头像、姓名、简介、社交链接
- [ ] **3.3** 配置 `navBarConfig.ts` — 导航栏及系列下拉菜单
- [ ] **3.4** 配置 `sidebarConfig.ts` — 右侧栏布局（专业学术风格）
- [ ] **3.5** 配置 `commentConfig.ts` — Giscus 评论系统
- [ ] **3.6** 配置 `footerConfig.ts` — 页脚信息
- [ ] **3.7** 配置 `fontConfig.ts` — 中文字体（保留 Zen Maru Gothic + 思源黑体）
- [ ] **3.8** 迁移静态资源 — 头像、封面图、favicon
- [ ] **3.9** 整体 UI 审查 — 确保专业学术风格，无色情/二次元元素

### 阶段四：内容迁移 📝 （预计 3-4 小时）

- [ ] **4.1** 编写 Frontmatter 转换脚本（Jekyll → Firefly）
- [ ] **4.2** 批量迁移 30+ 篇文章到 `src/content/posts/`
- [ ] **4.3** 验证数学公式渲染（KaTeX `$$` 语法兼容性）
- [ ] **4.4** 验证代码块高亮
- [ ] **4.5** 迁移项目页面内容 → `src/content/spec/projects.md`
- [ ] **4.6** 迁移论文页面内容 → `src/content/spec/publications.md`
- [ ] **4.7** 创建关于页面 → `src/content/spec/about.md`
- [ ] **4.8** 配置标签/分类，确保系列文章正确归类

### 阶段五：功能优化 & 测试 🔧 （预计 2-3 小时）

- [ ] **5.1** 配置 Pagefind 搜索（中文分词测试）
- [ ] **5.2** 配置 Giscus 评论（创建 GitHub Discussion 分类）
- [ ] **5.3** 配置 OpenGraph 图片生成
- [ ] **5.4** 移动端响应式测试
- [ ] **5.5** 暗色模式测试
- [ ] **5.6** 性能测试（Lighthouse）
- [ ] **5.7** SEO 检查（meta、sitemap、robots.txt）
- [ ] **5.8** 404 页面自定义

### 阶段六：部署上线 🚢 （预计 1 小时）

- [ ] **6.1** 配置 `astro.config.ts` 中的 `site` URL
- [ ] **6.2** 构建生产版本 `npm run build`
- [ ] **6.3** 本地预览生产构建
- [ ] **6.4** 配置 GitHub Actions 部署脚本
- [ ] **6.5** 推送到 GitHub，触发自动部署
- [ ] **6.6** 验证 https://meiluosi.github.io 正常运行
- [ ] **6.7** 更新 GitHub Pages 设置（如需切换到 Actions 部署）

### 阶段七：持续优化 🔄 （长期）

- [ ] **7.1** 监控 Giscus 评论运行状态
- [ ] **7.2** 定期更新 Astro/Firefly 上游
- [ ] **7.3** 添加 RSS Feed
- [ ] **7.4** 添加站点分析（Umami 或 Microsoft Clarity）
- [ ] **7.5** 优化中文字体加载性能
- [ ] **7.6** 持续创作新内容

---

## 七、技术要点 & 注意事项

### 7.1 KaTeX 数学公式兼容性 ⚠️

当前博客大量使用 `$$...$$` 数学公式（如 DUCG、强化学习系列），Firefly 通过 `KatexManager.astro` 支持 KaTeX，但需注意：
- `$$` 块级公式 vs `$` 行内公式的转义
- Markdown 解析器对 `$$` 中特殊字符的处理（`_` 下划线等）
- 每个使用公式的文章可能需要微调

### 7.2 中文文件名

Firefly 使用 Astro 内容集合，文章文件名作为 URL slug。中文文件名可能需要：
- 硬编码 URL 路径，或
- 改为英文 slug + 中文标题 frontmatter

**建议**：保持中文文件名，在 frontmatter 中明确 `slug` 字段。

### 7.3 图片路径

Jekyll 图片存储在 `/img/`，使用相对路径。迁移到 Firefly 后需确保：
- 图片放在 `public/img/` 目录
- 文章内图片引用改为 `/img/xxx.jpg`

### 7.4 已有域名的 SEO 保持

- 保持文章 URL 路径不变或设置 301 重定向
- 保持已有页面的 meta description
- 确保 sitemap 自动生成

### 7.5 备份策略

⚠️ **在执行任何破坏性操作前**：
```bash
# 1. 创建当前站点的备份分支
git checkout -b backup-jekyll
git push origin backup-jekyll

# 2. 在新分支上进行 Firefly 迁移
git checkout -b firefly-migration
```

---

## 八、时间估算

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 一 | 环境搭建 & 初始化 | 1-2 小时 |
| 二 | ACG 元素清理 | 1-2 小时 |
| 三 | 个人品牌配置 | 2-3 小时 |
| 四 | 内容迁移 | 3-4 小时 |
| 五 | 功能优化 & 测试 | 2-3 小时 |
| 六 | 部署上线 | 1 小时 |
| **总计** | | **10-15 小时** |

---

## 九、检查清单 (Checklist)

### 上线前必须完成
- [ ] 所有 ACG 元素已移除（看板娘、樱花特效、音乐、番组/追番页面）
- [ ] 配置文件中无 ACG 相关设置
- [ ] comments 系统正常工作（Giscus）
- [ ] 搜索功能正常（Pagefind）
- [ ] 所有文章数学公式渲染正确
- [ ] 移动端响应式布局正常
- [ ] 暗色模式无样式问题
- [ ] 导航栏所有链接有效
- [ ] 项目页面内容完整
- [ ] 论文页面内容完整
- [ ] 关于页面已创建
- [ ] 404 页面正常显示
- [ ] GitHub Actions 部署成功
- [ ] HTTPS 证书正常

---

> 📌 **下一步**：确认本文档内容无误后，开始执行「阶段一：环境搭建 & 模板初始化」。
