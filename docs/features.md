# 🎯 功能创意清单

> 按优先级排列的功能想法，含可行性评估。优先级 = 惊艳程度 × 实现难度 × 与博客定位契合度。

> **方向更新（2026-09）**：个人品牌长期主张为 RSI → AGI。当前执行顺序以 [roadmap.md](./roadmap.md) 为准：优先完善真实研究档案、案例叙事和内容维护流程。下方 AI 看板娘、粒子背景等旧点子保留作历史备选，不代表当前承诺，也不应覆盖新路线图。

---

## 当前执行方向（2026-10-25）

推进 Phase 29「智能问题演进史」。以现有 20 条智能探索路线为入口，梳理各路线在关键时间点提出的问题、方法、证据、局限和后续分叉。长期方案见 [intelligence-history-plan.md](./intelligence-history-plan.md)，当前成果见 [20路线统一研究入口](./intelligence-history/index.html)，第一轮保留为候选基线。

1. **第一轮候选全景**：20 条路线、106 个独立候选事件和 106 份来源；保留原作/译本/再版日期、阅读状态与资料缺口。
2. **可维护的图与文献队列**：同一份数据生成 SVG、Mermaid、离线交互图、路线底稿和文献队列；新增来源后可以同步更新。
3. **第二轮样板审校已完成**：学习效率与泛化9个节点、控制论与强化学习8个、具身与生态8个；每个节点有原文定位、阅读范围、机制和边界，配套25份笔记、16条关系、六张可编辑图和三个教学算例。逐篇状态为指定片段阅读，未计为全文完成或独立复现。
4. **第三轮A批已验收（2026-10-27）**：补三个样板的指定断层，展开四条路线；交付三张跨路线问题卡、两组版本对照、Baird六状态正式复现及可编辑图。见[A批工作台与验收记录](./intelligence-history/round-three-a/README.md)；B批于2026-10-27完成[指定范围验收](./intelligence-history/round-three-b/validation.md)：三条路线、25节点、三张问题卡、两项A批回访及一项合成方法核验；C—E批亦于2026-10-27完成[联合验收](./intelligence-history/cde-validation.md)：10条路线、66条节点记录、9张问题卡、9项回访和三项合成核验；全局已汇总144个去重节点，阅读与版本缺口另列。第四轮于2026-10-28启动，第一阶段[比较框架与首篇综合样板](./intelligence-history/round-four/index.html)已完成[验收](./intelligence-history/round-four/validation.md)：6条条件性结论、10份来源记录、两组可编辑图及精确方法核验。第二阶段[系统边界与发展](./intelligence-history/round-four/boundaries/index.html)亦已完成[验收](./intelligence-history/round-four/boundaries/validation.md)：9份选读、27条定位主张、6条结论、三组可编辑图、8项精确核验；3项新设计尚未执行。第三阶段[意义与意识解释](./intelligence-history/round-four/meaning/index.html)已完成[指定范围验收](./intelligence-history/round-four/meaning/validation.md)：12份选读、40条主张、6条结论、6项预测比较、3组可编辑图与11项精确检查；3项设计未执行。下一步综合修订与正式网站集成见[阶段计划](./intelligence-history/round-four/plan.md)。

## 历史执行方向（2026-10-22）

落地 Phase 28（见 [roadmap.md](./roadmap.md)）：
1. **学术工效与分发渠道**：编写论文级 `@media print` 样式；新增 `/en/rss.xml` 英文订阅源；修复 `/graph/` 的 Sitemap 排除与知识图谱 Hub 重定向缺陷；清理 3 个零引用废弃样式。
2. **解释性计算深化（方案 A）**：`ReactiveProseBlock` 扩展至 `bpe` 与 `dpo` 模式，深度嵌入预训练分词篇与偏好对齐篇。
3. **端侧轻量运行沙盒（方案 B）**：研发基于 Pyodide WebAssembly 的交互式 Python 代码沙盒 `CodeSandbox` 与自动挂载系统 `InteractiveProseEnhancer`，实现零首屏负载的端侧算例验证，并与 Web Audio 快门音效联动。

## 历史执行方向（2026-10-21）

落地全球顶级个人学术站四大流派设计（见 [design-archetypes.md](./design-archetypes.md)）：
1. **方向 A（解释性计算流）**：新增第七实验室 `/lab/transformer-forward/`，端侧执行极简 Micro-Transformer 单步推演、矩阵注意力热力图与神经回路消融。**已上线并深化**（新增逐层激活能量范数）。
2. **方向 B（活体知识花园流）**：启用思想生命周期（🌱/🌿/🌳/🔬/⚙️）分级；升级 `/graph/` 为全站力导向交互知识星系，支持 Andy Matuschak 式叠放抽屉与双向拓扑关联。**已上线**，本轮追加 3D 天体星图模式与「节点目录」纯文本索引（键盘、读屏与无 JavaScript 的路径，画布标 `aria-hidden`）。
3. **流派 3（微交互与物理工艺流）**：命令面板升级为全局控制台（all / action / lab / post 分页），与 Web Audio 机械微音效联动。**已上线**；分页语义由 `tablist` 改为 `group` + `aria-pressed`，与站内其它筛选按钮一致。
4. **流派 4（交互式叙事与长卷流）**：新增 `/essays/transformer-scrollytelling/`，Distill 风格双栏同步滚轮长卷，以五幕计算舞台解剖 Transformer 前向传播。**已上线**——这是本轮超出原 A/B 两向计划的新增方向；`/essays/` 索引页已把两篇长卷串成可浏览的入口。

后续推进顺序：第五卷（推理任务的强化学习）正文 → 英文概览随新正文同步回填 → `/graph/` 2D 拓扑视图的浏览器复验。

## 历史执行方向（2026-10-16）

进一步收紧页面顶部，仅保留标题；移除重复的随笔推荐、智能手记邀请及文章起读模块。文章索引默认采用与站点背景协调的透明卡片，并提供列表视图切换；文献架沿用相同的鼠尾草绿与暖白配色。

采用专业、明确的研究与功能标题，收敛全站字号。移除阅读收藏、阅读路线和文章末尾的反向链接／推荐区；文献架改为按主题筛选的交互资料卡片。后续内容结构等待用户另行设计，不自动恢复旧阅读路线。见 [本轮调整记录](./site-editorial-cleanup.md)。

## 历史执行方向（2026-10-09）

用户已确认以“人文＋科技”调整个人博客的表达：文章与作者声音前置，技术作为可探索的作品。当前优先完成首页编辑、舒适阅读、Strandbeest 视觉随笔、关于页和研究岛的物件叙事；见 [设计与验收记录](./humanities-and-technology.md) 和路线图 Phase 9。RSI 仍是长期研究方向，原始研究功能继续保留。下方记录与历史功能想法不覆盖这次已确认的方向。

### 智能手记与文献阅读（2026-10-09）

用户进一步确认将 RSI／AGI 改成有观察、有操作、有概念边界的互动随笔，并要求系统解读 dot 收集的文献，逐篇成文后形成综述。当前清单仍待来源线索，不能用现有文献架替代。执行见路线图 Phase 10 与 [阅读规范](./rsi-reading-notebook.md)。

### 上一轮执行方向（2026-10-04）

用户已确认推进研究工作室升级，核心内容为数据、分词、从零预训练、后训练、评测与部署。当前范围见 [研究工作室规划](./small-model-research-studio.md)，执行状态见路线图 Phase 8。首页入口、研究岛 2.0、全链路互动导读和独立训练项目优先于下方历史创意。

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

## 🥉 P2 — 锦上添花

### 6. 打字机效果标题

**描述**：首页副标题用打字机效果轮播，展示技术 slogan。

**已有基础**：✅ `TypewriterText.astro` 已存在，只需改文字内容。

**待实现**：
- [ ] 修改 `backgroundWallpaper.ts` 中的打字机文字列表

---

### 7. 文章点赞/反应

**描述**：文章底部添加 emoji 反应按钮（👍 👏 🤔 🔥），数据存 Giscus Discussions。

**实现方式**：Giscus Reactions API（已有 Giscus 集成）

---

### 8. 暗色/亮色代码主题自动切换

**描述**：代码块跟随网站主题切换暗色/亮色高亮主题。

**实现方式**：Expressive Code 已支持 dual theme，配置即可。

---

## 💡 Wildcard — 脑洞/实验性

### 9. 浏览器内 RL 演示

**描述**：在文章中嵌入小规模 RL 环境（如 FrozenLake、CartPole），用户可调整参数观察效果。

**实现方式**：TensorFlow.js / ONNX Runtime Web 在浏览器运行小模型。

**可行性**：中低。需要将 RL 模型导出为 Web 兼容格式，适合作为技术展示文章。

---

### 10. 因果推断在线计算器

**描述**：嵌入简单的因果推断工具，如 do-calculus 可视化、ATE 计算器。

**实现方式**：纯前端 JS（如 `dagitty` 的 Web 版）或 WASM 编译的 R/Python 工具。

**可行性**：中。与博客因果推断主题高度契合，可作为互动教学工具。

---

### 11. 文章音频版（TTS）

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

### 第四轮综合修订与网站联动（2026-10-28）

18条结论已共同复核，原句与修订分层保存；[来源台账](./intelligence-history/round-four/integration/source-registry.md)区分31条记录、29作品和26证据组。正式站点新增 /research/intelligence/ 及三篇综述、31个来源页、修订对照和研究队列，路线页、文献架、文章入口均可进入。图和原文定位支持直接追溯，SVG／Mermaid及研究数据可下载。三处关联认知透镜改用已核验的有限模型，明确不测量体验或判定认知构成。[验收](./intelligence-history/round-four/integration/validation.md)记录集成与部署边界。

### 第五轮：路线演进与来源追溯（2026-10-28）

`/research/intelligence/timelines/` 为20条路线提供年份图、节点问题／方法／限制、主张定位、来源版本与读段、既有关系和缺口。图、Mermaid和路线JSON可下载；原始批次数据和当前索引是维护入口。筛选需要脚本，节点与来源可在无脚本时阅读。见[执行计划](./intelligence-history/round-five/plan.md)。


第六轮[关键缺口与反方](./intelligence-history/round-six/README.md)：排序24项历史未映射候选，补读定义汇编、Marr视觉层次、Transformer和规模规律的指定原典片段，并加入Hoffmann与Pearce–Song的竞争解释；相关节点与结论回访进入网站。
