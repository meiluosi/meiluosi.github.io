# 🎯 功能创意清单

> 按优先级排列的功能想法，含可行性评估。优先级 = 惊艳程度 × 实现难度 × 与博客定位契合度。

---

## 优先级说明

| 级别 | 含义 |
|------|------|
| 🥇 P0 | 强烈推荐立刻做 — 惊艳、低难度、契合定位 |
| 🥈 P1 | 值得做 — 需要一定投入但效果显著 |
| 🥉 P2 | 锦上添花 — 有时间再做 |
| 💡 Wildcard | 脑洞/实验性 — 技术探索向 |

---

## 🥇 P0 — 强烈推荐

### 1. AI 对话看板娘

**描述**：点击看板娘触发 LLM 对话，预设 System Prompt 为"枫语博客 AI 助手"，擅长回答 AI / RL / 因果推断 / LLM 相关问题。

**实现方式**：Serverless 代理（Cloudflare Workers）
- 前端：复用已有的 `SpineModel.astro` / `Live2DWidget.astro` + `pioConfig.ts` 点击交互
- 后端：Cloudflare Worker 代理 LLM API → 加 rate limit（每 IP 每小时 10 次）
- API：DeepSeek / 通义千问（国产、便宜、中文好）
- 成本：约 0.01 元/次对话，个人博客几乎免费

**已有基础**：
- ✅ `pioConfig.ts` — 点击交互、消息气泡、待机动画配置完整
- ✅ `SpineModel.astro` / `Live2DWidget.astro` — 看板娘渲染基础设施
- ✅ `public/pio/` — 模型资源文件

**待实现**：
- [ ] Cloudflare Worker 部署 LLM API 代理
- [ ] 前端对话 UI 组件（Svelte 5）
- [ ] 流式输出（SSE / WebSocket）
- [ ] System Prompt 设计
- [ ] 可选：RAG 索引博客文章，让看板娘能精准回答

**安全**：✅ Key 藏在 Worker 环境变量中，加单 IP 限流

---

### 2. AI 文章摘要（构建时静态生成）

**描述**：在 `pnpm build` 时对每篇文章调用 LLM 生成一句话摘要，写入 frontmatter，构建后是纯静态 HTML，零运行时风险。

**实现方式**：构建时预生成
- 新增 `scripts/generate-summaries.ts`
- 在 `pnpm build` 流程中插入该步骤
- 摘要存入 `aiSummary` frontmatter 字段
- 文章页顶部渲染摘要卡片

**已有基础**：
- ✅ 构建管线清晰（icons → LQIPs → Astro → fonts → Pagefind）
- ✅ 文章 frontmatter 结构完整

**待实现**：
- [ ] `scripts/generate-summaries.ts` 脚本
- [ ] LLM API 调用（仅构建时，Key 存 CI 环境变量）
- [ ] 摘要 UI 组件
- [ ] 可选：AI 生成关键词云

**安全**：✅ API Key 仅在 CI 环境，客户端不可见

---

### 3. 知识图谱可视化

**描述**：基于所有文章的标签和分类，用 D3.js / ECharts 生成交互式知识图谱。节点 = 文章/概念，连线 = 关联关系，点击跳转。

**实现方式**：构建时预生成 + 客户端渲染
- 构建时：从 content collection 提取标签、分类，生成图谱 JSON
- 客户端：Svelte 组件 + ECharts/D3.js 渲染力导向图

**待实现**：
- [ ] 图谱数据生成脚本（构建时）
- [ ] 力导向图 Svelte 组件
- [ ] 页面路由（如 `/graph/` 或在 `/about/` 页面内嵌）

---

## 🥈 P1 — 值得做

### 4. 粒子神经网络背景

**描述**：替代当前的樱花特效，粒子随机移动，距离近的连线，模拟神经元放电效果。鼠标移动产生引力/斥力。

**实现方式**：客户端 Canvas / Svelte
- 修改 `SakuraEffect.astro` 或新建组件
- 纯 Canvas 2D API，无需 Three.js

**待实现**：
- [ ] Canvas 粒子系统 Svelte 组件
- [ ] 鼠标交互（引力/斥力）
- [ ] 配色跟随主题色相

---

### 5. 阅读进度条 + 目录高亮

**描述**：页面顶部彩虹进度条（渐变色跟随主题色），右侧 TOC 自动高亮当前阅读位置。

**实现方式**：客户端 Svelte 组件
- 监听 `scroll` 事件计算进度
- `IntersectionObserver` 高亮 TOC 当前章节

**待实现**：
- [ ] 进度条组件
- [ ] TOC 高亮逻辑

---

### 6. 文章 AI 问答（RAG）

**描述**：在文章底部放置"问 AI 关于这篇文章"按钮，点击后弹出对话框，基于该文章内容 RAG 回答。

**实现方式**：Serverless 代理 + 构建时向量索引
- 构建时：用 Pagefind 的分词结果做简易检索（或生成 embeddings）
- 运行时：Cloudflare Worker 接收问题 → 检索相关段落 → LLM 回答

**注意**：比对话看板娘更复杂，涉及 RAG 管线。可先做看板娘，积累经验后再做此功能。

**待实现**：
- [ ] 构建时 embeddings 生成
- [ ] Worker 端 RAG 管线
- [ ] 前端问答 UI

---

## 🥉 P2 — 锦上添花

### 7. 打字机效果标题

**描述**：首页副标题用打字机效果轮播，展示技术 slogan。

**已有基础**：✅ `TypewriterText.astro` 已存在，只需改文字内容。

**待实现**：
- [ ] 修改 `backgroundWallpaper.ts` 中的打字机文字列表

---

### 8. 文章点赞/反应

**描述**：文章底部添加 emoji 反应按钮（👍 👏 🤔 🔥），数据存 Giscus Discussions。

**实现方式**：Giscus Reactions API（已有 Giscus 集成）

---

### 9. 暗色/亮色代码主题自动切换

**描述**：代码块跟随网站主题切换暗色/亮色高亮主题。

**实现方式**：Expressive Code 已支持 dual theme，配置即可。

---

## 💡 Wildcard — 脑洞/实验性

### 10. 浏览器内 RL 演示

**描述**：在文章中嵌入小规模 RL 环境（如 FrozenLake、CartPole），用户可调整参数观察效果。

**实现方式**：TensorFlow.js / ONNX Runtime Web 在浏览器运行小模型。

**可行性**：中低。需要将 RL 模型导出为 Web 兼容格式，适合作为技术展示文章。

---

### 11. 因果推断在线计算器

**描述**：嵌入简单的因果推断工具，如 do-calculus 可视化、ATE 计算器。

**实现方式**：纯前端 JS（如 `dagitty` 的 Web 版）或 WASM 编译的 R/Python 工具。

**可行性**：中。与博客因果推断主题高度契合，可作为互动教学工具。

---

### 12. 文章音频版（TTS）

**描述**：为每篇文章生成 AI 朗读音频，读者可"听文章"。

**实现方式**：构建时调用 TTS API（如 Edge TTS / OpenAI TTS），生成 mp3 文件。

**可行性**：中。构建时间会显著增加，但用户体验独特。

---

## 已完成功能

| 功能 | 状态 |
|------|------|
| Pagefind 全文搜索 | ✅ 已启用 |
| Giscus 评论系统 | ✅ 已启用 |
| KaTeX 数学公式 | ✅ 已启用 |
| 音乐播放器（Meting API） | ✅ 已启用 |
| 壁纸/背景切换 | ✅ 已启用 |
| 主题色切换（亮/暗/系统） | ✅ 已启用 |
| 加密文章 | ✅ 已启用 |
| RSS 订阅 | ✅ 已启用 |
| 站点地图 | ✅ 已启用 |
| 图片懒加载 + LQIP | ✅ 已启用 |
| Mermaid / PlantUML 图表 | ✅ 已启用 |

---

## 已删除功能

| 功能 | 原因 |
|------|------|
| Bangumi 番组计划 | 未使用，Bilibili API 不稳定 |
| Anime 追番页面 | 未使用，内容与博客定位不符 |
| ACG 本地音乐 | 版权风险，已改用 Meting API |
| Twikoo 评论 | 已改用 Giscus |
| Gallery 示例相册 | 清理模板示例 |