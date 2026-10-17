# 📦 Components 组件目录

项目中所有可复用组件的集中管理。组件按照功能和职责进行分类，提供清晰的架构和易于维护的代码组织。

## 📁 目录结构

### 🏗️ layout/ - 页面布局组件

负责整体页面框架和布局结构的组件。

- `CategoryBar.astro` - 分类栏组件
- `ConfigCarrier.astro` - 配置载体组件
- `Footer.astro` - 页脚组件
- `Navbar.astro` - 导航栏组件
- `PostMeta.astro` - 文章元数据组件
- `SideBar.astro` - 侧边栏组件

### 🎮 controls/ - 导航和交互控件

页面导航和用户交互功能组件。

**导航控件**
- `ScrollDownIndicator.astro` - 向下滚动指示器

### 🔧 common/ - 公共可复用组件

通用的 UI 组件和工具组件，支持跨项目复用。

**基础 UI 组件**
- `ButtonLink.astro` - 链接按钮
- `ButtonTag.astro` - 标签按钮
- `Icon.svelte` - 图标组件（带加载状态和错误处理）
- `WidgetLayout.astro` - 小部件布局容器

**内容和展示组件**
- `CoverImage.astro` - 封面图组件（支持本地图片和随机图API）
- `ImageWrapper.astro` - 图片包装器（支持本地和远程图片）
- `Markdown.astro` - Markdown 内容样式包装器

### 🧩 widget/ - 小部件

侧边栏中使用的各种功能小部件。

- `Advertisement.astro` - 广告组件
- `Announcement.astro` - 公告组件
- `Calendar.astro` - 日历组件
- `Categories.astro` - 分类组件
- `Music.astro` - 音乐播放器小部件
- `Profile.astro` - 个人信息/社交链接小部件
- `SidebarTOC.astro` - 侧边栏目录组件
- `SiteInfo.astro` - 站点信息组件
- `SiteStats.astro` - 站点统计组件
- `Tags.astro` - 标签组件

### ✨ features/ - 全局功能特效组件

全局加载的功能增强和特效组件。

**管理器（初始化和管理功能）**
- `FancyboxManager.astro` - Fancybox 图片查看器管理
- `FontSetup.astro` - 字体加载和管理（基于 Astro Font API）
- `KatexManager.astro` - Katex 数学公式渲染管理

**功能组件**
- `EncryptedContent.astro` - 加密内容组件
- `EncryptedPost.astro` - 加密文章组件
- `MusicPlayer.astro` - 音乐播放器 UI 视图控制器
- `TypewriterText.astro` - 打字机动画效果

### 💬 comment/ - 评论系统组件

第三方评论系统集成组件。

- `index.astro` - 评论主组件
- `Artalk.astro` - Artalk 评论集成
- `Disqus.astro` - Disqus 评论集成
- `Giscus.astro` - Giscus 评论集成（GitHub 讨论）
- `Twikoo.astro` - Twikoo 评论集成
- `Waline.astro` - Waline 评论集成

### 📊 analytics/ - 数据统计组件

网站分析和统计集成组件。

- `GoogleAnalytics.astro` - Google Analytics
- `La51Analytics.astro` - 51la 统计
- `MicrosoftClarity.astro` - Microsoft Clarity
- `UmamiAnalytics.astro` - Umami 统计

### 🔧 misc/ - 杂项工具组件

其他辅助和工具类组件。

- `License.astro` - 许可证信息显示
- `SharePoster.svelte` - 分享海报生成

---

## 🗂️ 分类原则

| 分类 | 用途 | 特点 |
|------|------|------|
| **layout/** | 页面布局和结构 | 决定整体页面框架 |
| **controls/** | 导航和交互 | 用户交互功能 |
| **common/** | 通用可复用组件 | 跨多个页面/组件使用 |
| **widget/** | 侧边栏小部件 | 侧边栏特定组件 |
| **features/** | 全局功能特效 | 全局加载的增强功能 |
| **comment/** | 评论系统 | 第三方服务集成 |
| **analytics/** | 数据统计 | 分析和统计服务 |
| **misc/** | 工具和辅助 | 其他杂项功能 |
