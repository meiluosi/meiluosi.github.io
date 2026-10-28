# 🗺️ 枫语个人品牌网站路线图

> 目标：把原博客改造成完整、易记、可持续更新的个人品牌网站。视觉方向结合沉浸式探索与清晰的个人叙事；内容保留真实的研究、项目和笔记。

## Phase 1：品牌体验重建

| 项目 | 状态 |
|------|------|
| 建立统一的暗色视觉、荧光绿强调色、字体和导航系统 | ✅ 已完成 |
| 首页改为个人介绍、探索方向、精选笔记和联系入口 | ✅ 已完成 |
| 创建独立的方向、笔记、关于和联系页面 | ✅ 已完成 |
| 为品牌页面和文章共用导航加入中英切换，并保留文章原文语言提示 | ✅ 31 篇已发布中文文章均有英文译本或概览，可双向跳转；另 2 篇仅有占位正文且未发布，因此暂不生成英文页 |
| 英文界面下本地化未翻译文章的统计与阅读辅助标签 | ✅ 已覆盖字数、阅读时长、引用、推荐和评论区域；正文原文语言提示保留 |
| 品牌页、浏览器标题与 404 页面双语切换 | ✅ 首页、About、Notes、Work、Contact 的标题与描述随语言更新；404 双向切换已实测 |
| 为中英界面同步翻译无障碍名称 | ✅ 导航、研究地图和头像的 aria-label / alt 随语言更新 |
| 为旧归档、分类、标签和过期页面建立新入口跳转 | ✅ 已完成 |
| 重做文章页，使长文阅读与新品牌一致 | ✅ 初步完成 |
| 构建交互式 3D 研究岛：拖拽旋转、键盘漫游、站点跳转 | ✅ 已实现，持续调整 |
| 完成移动端与桌面端的视觉、键盘和触控核验 | 🟡 已核验 390px 中文/英文与 1280px 桌面布局无横向溢出；研究地图键盘焦点及 Enter 导航已实测；取消手势不再触发站点跳转，真实触屏设备仍待测 |

## Phase 2：真实性与内容打磨

| 项目 | 状态 |
|------|------|
| 将 RSI → AGI 设为全站品牌主张，统一首页与关于页叙事 | ✅ 已完成首轮 |
| 将模型系统、反馈学习、因果评估与长期问题建立映射 | ✅ 已完成首轮 |
| 从现有文章中挑选研究档案，区分计划、实现记录与应用案例 | ✅ 已整理 4 条，并改为以研究问题、方法与下一步为主线 |
| 增加首页 Now 区块，将研究问题连到已有文章，并标明方案与模拟数据边界 | ✅ 已完成首轮 |
| 让首页 Now 与 3D 研究地图共用研究线索和双语站点名称 | ✅ 已完成 |
| 为笔记增加入门路径、内容类型标记与主题筛选 | ✅ 已完成首轮 |
| 为笔记目录展示英文译名，并标识仍为中文的正文 | ✅ 首轮完成；已配对文章可中英跳转 |
| 为笔记目录分类、标签、日期与筛选区辅助名称提供双语显示 | ✅ 英文模式下本地化，筛选仍按原始分类值工作 |
| 为首页与研究档案统一翻译项目类型、研究方向和状态标签 | ✅ 已完成 |
| 研究方向关联笔记在英文界面仅显示已有英文版本，避免回落到中文链接 | ✅ 已完成首轮 |
| 让英语文章进入独立的站内搜索索引 | ✅ 2 篇完整英译与 29 篇英文概览共 31 篇，均进入 Pagefind 英文索引；中英文索引共收录 62 个已发布文章页面 |
| 对照原文核验精选档案措辞，并暂缓发布两篇空正文文章 | ✅ 已完成 |
| 为四项研究建立来源型案例页，展示问题、方法、当前观察与 RSI 联系 | ✅ 已完成首轮；各页直链原文，并以自然的后续问题保留事实边界 |
| 补充个人具体职责、执行范围、运行记录和结果 | ⏳ 等待后续提供项目级事实与可公开证据 |
| 为案例页补充可核验的过程图、运行截图或结果图 | ⏳ 当前文章未提供可用运行产物，不制作推测性配图 |

## Phase 3：上线质量

| 项目 | 状态 |
|------|------|
| 检查所有旧链接、站内链接、邮件和社交链接 | ✅ 已完成静态站内链接审计；dist 中 39 个内部链接均有目标 |
| 优化 3D 包体、首屏加载与低性能设备体验 | ✅ Three.js 已改为按需加载；包体压缩与设备实测待继续 |
| 核验无 WebGL、减少动态效果、窄屏和键盘访问 | 🟡 加入 WebGL 上下文丢失降级、动态 reduced-motion 响应与静止时停绘；窄屏和键盘已实测，真实 WebGL 故障、系统偏好切换与触控设备仍待核验 |
| Astro 检查、TypeScript 检查、完整静态构建 | ✅ Astro 检查、isolatedDeclarations 类型检查与完整静态构建通过 |
| 发布前确认 GitHub Pages 路径、元信息、社交预览图和站点地图 | 🟡 根路径与站点 URL 已核对；页面 canonical、品牌分享图、robots 与 sitemap 已补齐并在静态构建产物中核验；真实部署可访问性待发布后确认 |

## Phase 4：持续增强

| 项目 | 方向 |
|------|------|
| 项目案例页 | 用可交互演示、过程图和结果讲清代表作品 |
| 个人知识地图 | 将研究方向、项目和笔记连成可探索的图谱 |
| 交互与动效细节 | 根据真实访客路径，精炼 3D 热点、滚动和页面过渡 |
| 内容维护流程 | 让新文章、项目和近期状态能低成本更新 |

## Phase 5：轻快配色与动效升级（2026-09-30 完成）

保留 RSI / AGI 主张、真实研究案例、双语内容、Now 与 3D 研究岛。参考 DeepSeek Harness 和 Moonshot 的内容展示节奏，先确定视觉方向，再实现动效。

| 项目 | 状态 |
|------|------|
| 真实品牌配色调研与概念预览 | ✅ 已筛选 Graza、Clay、Twelve Labs；用户明确选择 B「鼠尾草与桃色」 |
| 首页配色与研究岛日光材质原型 | ✅ 统一品牌页面、文章、分享图与 favicon；完成桌面与 320 / 390px 窄屏浏览器复核，修正研究岛取景、手机布局与导航换行 |
| 首屏与章节动效 | ✅ 完成标题分层入场、章节一次性渐入、Now 连线与研究岛首帧渐入；浏览器确认首屏内容正常显现、Now 动画触发、锚点滚动后卡片可见 |
| 内容页与交互反馈 | ✅ 完成卡片与箭头反馈、原生页面过渡、筛选重排与空状态；浏览器实测组合筛选、重置焦点、中英切换、研究方向与案例原文跳转。修正中文正文误套标题字号，手机正文保持 16px |
| 体验验证 | ✅ 静态检查、类型检查、完整构建与控制逻辑验证通过；补齐真实浏览器桌面/窄屏、双语阅读、筛选及导航检查，截图见下方。系统减少动态效果切换、实体触屏和真实 WebGL 故障仍属于后续设备兼容性抽查范围 |

配色参考：[Graza](https://www.graza.co/)、[Clay](https://www.clay.com/)、[Twelve Labs 品牌访谈](https://brandfetch.com/blog/twelvelabs-new-brand-and-logo)。概念配色为本站提案，不作为上述品牌的官方色值。

### 验证记录

早期浏览器工具读取失败，曾停在视觉验收处；本轮检查发现实际开发服务运行于 `http://localhost:4323/`，此前使用的 4322 已无监听。通过正常浏览器工具打开当前服务后恢复读取，没有绕过访问限制，用户无需提供截图。

2026-09-30 浏览器核验：

- 桌面 1440×960、手机 390×844 与 320×740；页面无横向溢出。研究岛手机构图改为独立下方区域，并增加三个始终可读的直达入口。
- 笔记组合筛选：实验计划 × 概率论产生 0 结果；清空恢复 33 条开发环境笔记并把焦点放回“全部”；英文计数为 `33 of 33 notes`。开发环境含两篇草稿，生产仍为 31 篇中文与 31 篇英文。
- 手机研究入口正确跳到 `/work/#llm-systems`；案例 → 中文原文 → 英文版本 → 中文原文的往返正常。
- 中文文章修复前正文错误继承 34px 标题字号；修复后正文 16px / 30.4px 行高、二级标题 24px。英文阅读页与中英文导航在 320px 下无横向溢出。
- 首页锚点滚动最终正确对齐项目区，三张精选卡片 opacity 为 1，Now 连接线已触发。画布键盘 Tab 可正常转到后续链接；本轮未声称实测了靠近站点后的 Enter 或实体触摸手势。
- 浏览器日志检查未发现错误或警告。减少动态效果与 WebGL 降级沿用已有实现和控制逻辑验证，本轮未模拟系统偏好切换或真实 GPU 故障。
- Astro 检查（169 文件，0 错误、0 警告、4 个已有提示）、TypeScript 检查与完整静态构建通过。

实际截图：[桌面首页](screenshots/sage-home-desktop.png)、[手机首页](screenshots/sage-home-mobile.png)。

页面过渡依据：[Chrome 官方跨文档 View Transitions 文档](https://developer.chrome.com/docs/web-platform/view-transitions/cross-document)。保持普通多页导航，无链接拦截或等待动画结束再跳转。

## Phase 6：个人经历与探索体验（2026-09-30 完成）

依据用户提供的简历与“从研究生阶段开始关注智能”的补充，推进以下内容；按用户要求不调整 Now。

- [x] About 增加双语叙事和三段经历：南京邮电大学本科、中国传媒大学硕士、2025 年起的大模型工程实践。自动化科研闭环按设计阶段表达。
- [x] 笔记目录提供中英文标题与正文搜索、正文片段、组合筛选、可分享查询地址和显式返回时的列表位置恢复。
- [x] 中英文文章共用目录：宽屏固定侧栏、窄屏折叠，当前章节高亮；按正文计算阅读进度。
- [x] 后续阅读按研究方向和共同主题选择，并说明阅读理由；替换中文文章原有随机推荐。
- [x] 研究岛增加站点预览，连接核心问题、代表案例和研究方向。键盘与手机按钮可打开相同预览，原生对话框支持 Escape 关闭与焦点返回。
- [x] 新增 `/lab/causal-playground/`，通过优惠券分配偏好和真实效果滑块展示选择偏差；链接研究方向、YLearn 案例与双语原文。明确教学模型假设，未使用个人实验结果或顾客数据。
- [x] 联系页增加邮箱复制与反馈、具体交流主题和完整中文文案，保留既有公开邮箱。

实现维护位置：`PersonalJourney.astro`、`NoteSearch.astro`、`note-filters.ts`、`api/note-search.json.ts`、`ArticleNavigation.astro`、`NextReading.astro`、`ExploreWorld.svelte`、`CausalPlayground.svelte`、`CopyEmail.svelte`。全文索引按需请求，只收录非草稿、非加密文章；开发和静态部署使用同一入口。

验收记录：

- `pnpm check`：175 个文件，0 错误、0 警告、4 个既有提示；`pnpm type-check`、完整 `pnpm build`、`git diff --check` 均通过。生产构建 84 页，Pagefind 保持 62 篇双语文章；新索引包含 31 对中英文正文，约 591 KB，首次搜索时加载。
- 桌面 1440×960、手机 390×844、窄屏 320×740：检查个人介绍、实验及中英文阅读页宽度，无横向溢出。手机目录点击后折叠并跳转章节，桌面目录未遮挡正文。
- 搜索 `uplift` 返回 YLearn 正文片段，叠加“应用拆解”后点击文章；显式返回恢复查询、筛选和位置。英文模式打开英文文章；无结果搜索的重置恢复列表并将焦点放回稳定控件。
- 因果实验默认显示直接差值 +68、组内效果 +20；随机分配均为 +20；反向偏好 -80 时直接差值 -44、真实效果仍为 +20；设定真实效果 -20、偏好 +60 时直接差值为 +28。已检查双语切换、键盘滑块和手机即时数值。
- 桌面及手机按钮打开研究站点预览，Escape 关闭并恢复按钮焦点；已检查远离站点时的键盘提示，未将此记作完整漫游验收。联系页复制显示成功反馈；未模拟剪贴板权限被拒绝。
- 控制台出现既有 Three.js Clock 弃用提示，无新增交互错误。构建保留既有 R 代码高亮、Markdown 配置弃用、robots 同名和重定向页面搜索提示。
- 截图：[个人经历](screenshots/personal-journey-desktop.png)、[桌面因果实验](screenshots/causal-playground-desktop.png)、[手机因果实验](screenshots/causal-playground-mobile.png)。

## Phase 7：研究导读、实验室与阅读连接（2026-10-04 完成）

按用户确定的三轮范围实施，延续鼠尾草绿与桃色、已有研究岛和文章体系。

### 第一轮：内容组织

- [x] `/rsi/`：解释长期主张、研究循环、三个阅读入口与开放问题。叙事依据用户已说明的方向和现有文章。
- [x] `/paths/` 及三个详情页：从搜索到自我对弈、怎样判断改进有效、从模型到智能系统。合计串联 10 篇既有笔记，标出基础、逐篇阅读目的和路线结束后的思考问题。
- [x] `/library/`：六份原始论文、教材或工具资料，附阅读切口、关联笔记、按路线筛选和可分享筛选地址。区分资料导读与个人读后评价。
- [x] 首页与公共导航增加“探索”；笔记目录增加导读、路线、实验室和收藏入口。中英文链接随语言指向相应文章。

### 第二轮：互动实验

- [x] `/lab/`：汇集 MCTS 与已有因果实验，分别连接阅读路线。
- [x] `/lab/mcts/`：实际执行 UCT 选择、单节点扩展、均匀随机 rollout 和路径回传；可单步、慢速演示、暂停、跑完预算和重置。
- [x] 预算、探索权重、随机种子可调；展示搜索路径、模拟路径、访问次数、均值、探索项和最终根动作。配置链接可复现相同序列。
- [x] 明示单智能体、固定叶子奖励、有限预算与无模型训练的教学规则。手机树图可横向滚动，统计表仍可直接阅读。

### 第三轮：阅读工具

- [x] 正文术语首次出现时可打开概念说明：RSI、AGI、MCTS、UCT、PUCT、ATE、CATE、RAG。扫描排除链接、标题、代码与公式；原生对话框支持键盘关闭和焦点返回。
- [x] 从已发布文章的 Markdown 链接与引用式链接构建反向链接，按中英文源文章去重；展示来源章节。代码块内容不会成为链接。
- [x] 文末显示所属路线、下一篇与关联资料；MCTS 和 AlphaZero 中英文正文补充相互延伸阅读入口。
- [x] `/reading/` 及文章保存按钮：中英文共用文章 ID、本地保存、跨标签页更新、空状态、移除与撤销、存储失败反馈。页面说明当前浏览器保存范围。

### 验收与证据

- 最新源码的 `pnpm check`（195 文件、0 错误、0 警告、4 个既有提示）、`pnpm type-check`、`pnpm build`、`git diff --check` 均通过。Pagefind 索引 65 篇文章；构建也包含工作区已有的三篇新增中文文章。
- 对九个新页面生成的 HTML 检查 274 个站内链接（含双语目标），无缺失目标。构建中的中英文 MCTS 文章包含收藏、概念说明、阅读路线和反向链接。
- 使用正式构建预览 `http://127.0.0.1:4324/` 核对。长期运行的 4323 开发服务保留旧样式，不能将其缓存画面当作本轮最终版本。
- 桌面 1440×960、手机 320×740 核对新栏目。七个导读、路线、文献和收藏页面宽度均为 320px；文章导航右缘为 304px，按钮均在视口内。MCTS 树在自己的滚动区横向浏览，未撑宽文档。
- MCTS：默认预算 50、c=1.4、seed=7，根动作访问次数 A/B/C 为 13/18/19；重置重跑一致。c=0 时为 1/1/48。预算 200、c=1.4、seed=42 时为 23/139/38，配置链接重新打开并重跑一致。首轮模拟奖励 0.40 正确回传；暂停后可继续单步。
- 文献按路线显示 2/6；语言切换后标题、筛选、导读与文章目标一致。收藏经过保存、刷新、移除、空状态、撤销和英文文章往返；概念对话框 Escape 关闭后焦点返回原术语。
- 构建保留已有 R 高亮、资源块大小、robots 同名等提示。快速连续跨页面操作时浏览器记录原生 View Transition 中止信息，实际导航和阅读交互正常；未模拟浏览器存储或剪贴板权限拒绝。
- 截图：[三条路线](screenshots/reading-paths-desktop.png)、[实验室](screenshots/lab-index-desktop.png)、[MCTS 桌面](screenshots/mcts-desktop.png)、[MCTS 手机](screenshots/mcts-mobile.png)、[概念预览](screenshots/concept-preview-desktop.png)。

## Phase 8：研究工作室与小模型全链路（2026-10-04）

本轮网站体验与可运行工程基线已完成。范围和一手依据见 [研究工作室规划](./small-model-research-studio.md)，网站检查见 [验收记录](./studio-validation.md)，实际训练见 [训练验证记录](./small-model-training-validation.md)。正式语言模型能力研究继续按下列未完成项推进。

- [x] 硬件实读：M4 MacBook Air、10核CPU/8核GPU、16GB；MPS在沙箱外可用；记录磁盘快照与预算边界。
- [x] 核对 MiniMind、nanochat、LLMs-from-scratch、MLX、PyTorch MPS 等一手资料，区分随机初始化训练与现成模型适配。
- [x] 中英原1B规划、案例、档案、Now和阅读路线同步更新，旧URL继续有效。
- [x] 轻量首页：精细SVG岛、显式3D启动、模型预算预览、三条首次访问路线。
- [x] 独立 `/world/`：装置/材质/光影、路径导航、缓动驾驶/轮动、相机跟随、到达反馈、内容面板、复位和44px触控控件。
- [x] 性能策略：动态导入、轻量/完整画质、DPR限制、离屏/后台暂停、减少动态响应与WebGL降级。
- [x] `/lab/model-pipeline/`：数据、简化BPE、next-token NLL、SFT mask、DPO、固定集评测与CLI部署导读；无JS静态阅读版本。
- [x] 三个项目在首页/档案/案例共用SVG视觉与独立PNG分享封面。
- [x] 可迁移 `projects/small-model-lab/`：自包含代码、配置、数据/分词、真实训练、检查点恢复、评测、本地CLI和发布文档。
- [x] 138,560参数tiny MPS基线：预训练12步、SFT8步、DPO8步，预训练另恢复至14步；保存真实报告，不作为模型能力或持续性能结论。
- [x] 网站诊断、类型检查、生产构建、289内部链接审计、六项Node测试（含957路线）、1440px与320px双语交互检查。
- [ ] 30M正式语料与完整训练预算；先短跑和持续资源测量，再进入完整预训练/后训练。
- [ ] 100M/约0.5B扩展决策；逐阶段测量内存、持续吞吐、检查点磁盘开销。
- [ ] 真实手机多指操作、GPU/热量/电量压测；线上性能指标与部署可访问性。
- [ ] 将有任务意义的真实模型回答、对照曲线与能力评测接入网站。tiny fixture损失不作为成果曲线展示。

交付目录可整体复制成独立GitHub项目，不依赖博客；先前的外部规划目录保留。本轮未创建远端、发布模型或部署网站。更大配置的参数量为29,676,544 / 91,740,672 / 514,859,520；30M/100M/500M是档位简称，不是已训练模型。

截图：[首页](screenshots/studio-home-desktop.jpg)、[最终研究岛](screenshots/studio-island-final.jpg)、[手机控制与面板](screenshots/studio-island-mobile.jpg)、[工作台](screenshots/studio-workbench-desktop.jpg)、[英语手机首页](screenshots/studio-home-mobile-en.jpg)。

## Phase 9：人文与科技（2026-10-09 完成）

用户已确认本轮方向，以个人作者、阅读与好奇心组织网站。当前设计和事实边界见 [人文与科技](./humanities-and-technology.md)。本轮优先级覆盖早期首页研究宣言与工程展示的安排，已有研究功能保留。

- [x] 首页重写为欢迎语、Strandbeest 选荐、文章、探索角落与简短近况，移除首页预算面板。
- [x] 关于页自然介绍作者与兴趣，公共导航突出文章、作品、关于我；中英文案一致。
- [x] 研究岛新增纸笔、木桌、铜灯与具体问题，保留已有驾驶、镜头、触控与降级能力。
- [x] 完成 Strandbeest 视觉随笔和几何互动，验证不变量、静态呈现与操作状态。
- [x] 完成书刊式阅读样式、构建、链接和双语浏览器验收。
- [x] 记录实际验证、截图和仍待验证的设备边界。

验收：`pnpm check`、`pnpm type-check`、生产构建和 10 项算法测试通过；3701 个内部链接属性无断链。已在真实浏览器检查桌面与 320/390px 窄视口、双语阅读、连杆操作、搜索筛选和研究岛站点到达。截图与设备边界见 [本轮记录](./humanities-and-technology.md#验收)。

## Phase 10：智能手记与文献阅读（2026-10-09 启动）

用户已确认推进 RSI／AGI 的人文互动叙事，并将 dot 曾收集的文献逐篇解读、汇成综述。

- [x] 将 RSI 导读组织为“改变了什么、能否迁移、如何评价”的互动随笔。
- [x] 增加可执行的规则泛化教学例，保留静态阅读、中英文与移动端体验。
- [x] 改善探索页阅读字号，接通实验、案例与文献的来回路径。
- [ ] 定位 dot 文献原始清单，逐篇核对一手全文、版本与证据范围。
- [ ] 完成逐篇解读与跨文献综述，连接文献架与文章目录。
- [x] 完成网站部分的内容核验、Astro、TypeScript、生产构建与浏览器交互验证；文献博客待原始清单确认后另行核验。

网站验收与文献剩余工作见 [智能手记记录](./rsi-reading-notebook.md)。该阶段尚未整体完成，不能将文献检索准备计为逐篇解读。

## Phase 11：大模型训练研究笔记 · 数据卷（2026-10-16 完成）

- [x] 完成中英文预训练语料调研，记录版本、规模口径与一手来源。
- [x] 写入六篇中文文章：语料地图、清洗、质量筛选、去重与污染、混合采样、可追溯语料构建。
- [x] 建立卷内目录及前后篇链接，沿用现有分类与标签入口。
- [x] 核验公式算例与代码示例，通过站点检查、生产构建和桌面／手机阅读验收。

来源与验证详见 [数据卷研究记录](./pretraining-data-research.md)。本阶段交付研究文章与教学示例；真实语料构建、消融训练和后训练各卷另行推进。

## Phase 12：文章概念网络与评论关闭（2026-10-16 完成）

- [x] 将标题拼接式导图替换为正文概念、章节层级和跨章节共同概念组成的 SVG 网络。
- [x] 接入中英文文章，支持分支、层级、搜索、原文定位与导出，并适配手机布局。
- [x] 根据用户后续决定关闭评论区；恢复排查期间临时开启的 Discussions。
- [x] 完成最终构建与浏览器交互复核。

实现与验收见 [文章概念网络记录](./article-concept-network.md)。自动提取的关系明确限定为层级与同一概念的再次出现。

## Phase 13：标题规范与探索模块精简（2026-10-16）

- [x] 统一标题字号，调整中英文界面中的口号式标题和主要导语。
- [x] 移除阅读收藏、阅读路线、文章反向链接和延伸推荐，处理旧地址。
- [x] 将文献架改成支持主题筛选、搜索和详情展开的紧凑卡片。
- [x] 完成构建、链接与桌面／手机交互验收。

详见 [标题与导航调整记录](./site-editorial-cleanup.md)。

## Phase 14：入口可用性与交付质量（2026-10-17 完成）

按「最值得先做的 5 件事」推进，全部是可用性与死负载清理，不改版式方向。

- [x] **分类与标签入口接回真实筛选**：`getCategoryUrl` 改指 `/notes/?topic=<原始分类>`，`getTagUrl` 改指 `/notes/?q=<标签>`；标签写入正文搜索索引；笔记页支持 `popstate`，浏览器前进/后退按地址栏恢复 q/format/topic。
- [x] **悬停预取 + 清掉 Swup 死层**：`astro.config.mjs` 开启 `prefetch`；`Layout.astro` 删除 `#progress-bar`、整套 `window.swup.hooks` 与 `setup()`（约 467 行）以及给不存在元素绑的 click-outside；`transition.css` 删除 Swup 与进度条规则，保留仍在使用的 `.onload-animation`。
- [x] **下线死负载**：`public/pio`（15MB，无任何页面引用）移至 `projects/live2d-assets/`；移除构建中的 `pagefind` 步骤并删除 `pagefind.yml`（索引 1.6MB，全站已改用 `/api/note-search.json`）；`rss.xml` 由 37 篇全文（4.1MB）改为最近 20 篇摘要（12KB）；sitemap 排除 9 个 301 跳转壳与分页页。
- [x] **顺带修掉一个隐性体积源**：`posts/[...slug].astro` 的封面图查找用 `import.meta.glob("../../**")` 把整个 `src/` 拖进模块图，导致每个页面产物里都带一个 665KB 的 Live2D 孤儿 chunk、并让文章页加载 22 个样式文件。收窄为图片扩展名后，文章页 CSS 从 22 个/455KB 降到 4 个/315KB，构建从 13.4s 降到 6.8s，`dist` 从 47MB 降到 25MB。
- [x] **文章页头部与目录**：元信息改用品牌 token（`--brand-muted` 对纸色 5.04:1，原 `black/30` 仅 2.10:1）；标签容器允许换行，320px 不再裁断；固定目录断点 1440 → 1180（1280 笔记本可用），宽度 `clamp(200px,16vw,240px)`，锚点偏移 88px 并修正此前被 `markdown.css` 压掉的 `scroll-margin`。
- [x] **双语与摘要**：为 68 个中英文章页输出 `hreflang`（zh-CN / en / x-default），31 篇有译文的中文文章在标题下新增服务端可爬的 `English overview →` 入口；30 篇文章的 description 从 10–17 字重写为 61–84 字（2 条空值补齐），保留原有证据边界表述。

验收：`pnpm check`（208 文件，0 error / 0 warning）、`pnpm type-check`、完整 `pnpm build`（102 页，6.8s）通过；构建产物核对 `dist` 无 `/pio`、无 `/pagefind`，文章页仅 4 个 CSS，sitemap 无 301 壳，68 页输出 `hreflang`，31 篇中文文章有静态英文入口。

仍未做：231KB 公共样式块（legacy Tailwind + Firemark 层）需要拆 CSS 入口才能进一步下降；`public/strandbeest`（2.6MB）与 6 篇未发布稿待定；Fancybox 样式仍为全局静态引入。

## Phase 15：英文文章页并入品牌阅读排版（2026-10-17 完成）

中文原文页有一套 `is:global` 覆盖把它重排成「鼠尾草 + 暖白」的阅读版式，英文概览页没有，导致同一篇文章的中英版本在留白、容器、标题色、元信息上不一致。

- [x] **抽出共用皮肤**：新增 `src/components/features/ArticleBrandStyle.astro`，承载 `#content-wrapper` 留白、`#post-container` 去卡片化、标题品牌墨绿、正文链接／引用／行内代码／图片配色；中文页与英文页各自 `import` 同一份。
- [x] **英文页结构对齐**：补上字数与阅读时长行、`PostMetadata`（日期／分类／标签）、版次标签、与中文页一致的标题与元信息层级；正文改用与中文页相同的 `Markdown` 包裹器（补回 `max-w-none!`，避免英文正文被 `prose` 收窄）。
- [x] **修掉英文页的公式破版**：31 篇英文页里有 9 篇含 `$…$`，此前只有 KaTeX 标记、没有 KaTeX 样式。英文页补 `KatexManager` 后与中文页一致。
- [x] **补英文页结构化数据**：新增 `BlogPosting` JSON-LD（`inLanguage: en`，含 `wordCount` 与 `translationOfWork` 指向中文原文）。
- [x] **顺带把英文页的 `.edition-label` 纳入统一排版**：等宽小字 + 大写 + 品牌强调色。

验收：同一篇文章的中英页 CSS 文件集合完全一致（`diff` 为空）；31 个英文页均有字数、阅读时长、版次标签与 `BlogPosting`；`pnpm check`（209 文件，0 error / 0 warning）、`pnpm type-check`、完整 `pnpm build`（102 页）通过；预览服务下 `/`、`/notes/`、`/notes/?topic=…`、中英文章页、`/rss.xml`、`/about/` 均返回 200。

仍未做：英文页的上一页／下一页导航与概念图数据仅覆盖中文原文；中英页各自的「英／中」切换仍是 JS 改写链接，静态入口已由 hreflang 与标题下链接覆盖。

## Phase 16：文章阅读链（2026-10-17 完成）

旧的上／下一篇由 `getSortedPosts()` 在构建时写入 frontmatter，顺序纯按发布时间：读完一篇强化学习可能接到因果推断；英文页则完全没有这个导航。37 篇里 31 篇没有任何站内文章互链。

- [x] **抽出可测试的阅读链**：新增 `src/utils/reading-chain.ts` 的 `buildReadingChain()`——同分类内按发布时间排序（同一天用 slug 稳定排序），分类邻居不存在时退回全局发布时间相邻，并标记为「按发布时间」。
- [x] **中文页替换旧链**：`posts/[...slug].astro` 不再依赖 frontmatter 里的 `prevSlug/nextSlug`，改用构建时计算的链；加密文章不进入链，避免把读者带到需要密码的页面。
- [x] **英文页新增同款导航**：`/en/posts/*` 按英文译文的分类与日期生成，仅包含已发布译文。
- [x] **统一视觉**：新增 `ReadingChain.astro`，品牌 token 卡片（等宽方向标签 + 两行标题 + 来源说明），移动端单列，补 `focus-visible` 与 `prefers-reduced-motion`；顺带把分类展示名抽到 `src/data/category-labels.ts`，供笔记目录与阅读链共用。
- [x] **边界处理**：最早一篇的“上一篇”、最新一篇的“下一篇”回落到「全部文章」。

验收：阅读链在 37 个中文页与 31 个英文页均渲染；数据卷 01→06 按系列顺序相连；MCTS ↔ CFR ↔ 多人博弈（游戏 AI）同主题相连；DUCG 建模（分类仅 1 篇）按时间兜底；无自链，仅最早／最新两处为「全部文章」；`pnpm check`（212 文件，0 error / 0 warning）、`pnpm type-check`、完整 `pnpm build`（102 页）与预览冒烟通过。

仍未做：英文概念图仍基于中文原文正文生成；阅读链目前只链到同分类的相邻一篇，跨分类的“相关阅读”仍保持移除状态。

## Phase 17：英文概念图、系列导航与相关阅读（2026-10-18 完成）

Phase 16 留下的三个口子：英文概念图层薄、跨分类没有入口、数据卷没有系列感。

- [x] **英文概念图补词表**：`article-concept-graph.ts` 的概念词表原本只有中文，英文文章只能靠加粗词与拉丁缩写撑起概念层。新增约 100 个英文概念词（exploration / reward / policy / value function / causal inference / deduplication …），中英各用各的语言命中，不做跨语言推断。同篇对比：MCTS 概览概念 **22 → 46**，Actor-Critic 概览 **17 → 44**；中文侧基本不变（114 → 116）。
- [x] **顺手修两处节点噪声**：`clean()` 去掉加粗片段的尾随标点（`Selection:` → `Selection`），并把 `Scope note` / `Further reading` 这类英文版块名排除出概念节点。
- [x] **系列元数据**：posts / enPosts 的 schema 与手写类型增加可选 `series` / `seriesOrder`；6 篇数据卷标注 `series: "大模型训练研究笔记 · 数据卷"`、`seriesOrder: 1..6`（正文与其他字段逐字未动）。
- [x] **系列导航**：新增 `SeriesNav.astro`，在标题下渲染一条紧凑横条「系列：… · 第 4/6 篇」+ 1–6 数字链接，当前篇用 `aria-current="page"`；只在声明了 `series` 的文章出现（当前 6 页），移动端数字行可横向滚动。
- [x] **相关阅读窄栏**：新增 `buildRelatedByTags()`（共同标签数 → 发布时间 → slug 排序，最多 3 篇）与 `RelatedArticles.astro`（一行小标题 + 可换行链接 + 共同标签原因），跨分类但不做大区块；与阅读链的上一篇／下一篇去重；`siteConfig.post.showRelatedByTags` 可一键关闭。构建后 **60 个文章页**显示，8 篇因标签唯一不显示。

验收：`pnpm check`（215 文件，0 error / 0 warning）、`pnpm type-check`、完整 `pnpm build`（102 页）通过；构建产物核对系列条只在 6 篇数据卷出现、相关栏与阅读链无重复项、英文 Mindmap 图数据从 30 个节点增长到 54；`astro preview` 下中文首页、数据卷首末篇、英文文章、`/notes/`、`/rss.xml` 均返回 200。

仍未做：英文译文的 `series` 字段留空（数据卷暂无英文译本，有译本时填上即可自动生效）；相关阅读只按标签精确匹配，没有同义词或主题轴归并。

## Phase 18：动效系统与交互一致性（2026-10-18 完成）

第一轮审计里「动效」与「交互」两个维度的第一批问题。

- [x] **motion token 收口**：`brand-theme.css` 的 `:root` 增加 `--motion-fast/base/slow` 与两条缓动，作为全站唯一动效刻度；`brand-motion.css` 删除局部 `--motion-*`，所有引用改指 token（filter 180ms、箭头 220ms、thread 线 900ms 都归位）。
- [x] **页面过渡修正**：此前 old 160ms 淡出与 new 220ms 淡入并行，中间帧两页文字叠加。现在 old 只做 140ms 淡出并留在上层（`z-index:2`），new 静态呈现（`animation:none`），`::view-transition` 铺 `--brand-paper` 底色。
- [x] **删掉死主题 VT**：`main.css` 里 0.5s 的 `::view-transition` 主题切换、`theme-fade-*` 与整段 `.is-theme-transitioning`（约 130 行）已无生产者（暗色不可达），删除后与新过渡不再打架。
- [x] **hover/press/reduce**：`.entry:hover` 的 padding 位移从全局移进 `(hover:hover) and (pointer:fine)` 查询，触屏点完不再粘住缩进；补 `:active` 按压反馈（纯合成属性，不重排）；`prefers-reduced-motion` 下把 padding/translate/scale 一并清零。
- [x] **入场节奏**：文章/内页入场从 620ms、`opacity:0`、18px 位移改为 320ms、`opacity:0.7`、10px，stagger 65ms → 50ms；一次性与 focus 保护不变。
- [x] **滚动管线**：`ReadingProgress` 改为 rAF 合并 + 几何缓存（`ResizeObserver` / resize / `fonts.ready` 重测），scroll 回调里没有任何布局读取，去掉 150ms 的 width transition；`ArticleNavigation` 只在固定目录或移动端展开时计算，rAF 内先读后写，卸载/bfcache 清理补齐。
- [x] **首页研究岛去水合**：compact 变体本来只渲染静态 SVG 与两行文案，改为 `ExploreWorldCompact.astro` 纯 Astro 输出——首页不再加载 `ExploreWorld` chunk 与 Svelte 岛运行时，仅剩语言切换一个岛；`/world/` 的完整 3D 不变。
- [x] **交互一致性**：笔记搜索补双语 placeholder、清除按钮空态禁用；文献架把 `topic`/`q` 写入 URL 并支持 `popstate` 还原（与 `/notes` 一致）；新增 `a11y.css` 零特异性焦点环兜底与 skip link（跳到 `#swup-container`，运行时补 `tabindex="-1"`）。

验收：`pnpm check`（216 文件，0 error / 0 warning）、`pnpm type-check`、完整 `pnpm build`（102 页）通过；产物核对首页无 `ExploreWorld` 引用（仅剩 `LanguageSwitcher` 岛）、`/world/` 仍加载 3D、`::view-transition-new(root)` 无动画、skip link 与 placeholder 出现在 HTML、文献架脚本含 `replaceState`/`popstate`；`astro preview` 下 `/`、`/notes/`、`/library/?topic=…`、中英文章页、`/world/` 均 200。

未验证：没有浏览器自动化，页面过渡的 `z-index` 层级、触屏 hover 粘滞修复、skip link 是否被 sticky 导航遮挡都只做了静态推演，需要在真实浏览器里看一眼。

## Phase 19：⌘K 命令面板与死交互层清理（2026-10-18 完成）

- [x] **命令面板数据**：新增 `/api/palette.json`，构建时生成 77 条（37 篇中文文章 + 31 篇英文译文 + 9 个页面），只保留标题/描述/标签/分类，产物 35KB，首次打开才请求。
- [x] **面板交互**：新增 `CommandPalette.svelte`，原生 `<dialog>` + `showModal()`；⌘K / Ctrl+K、`/`（非输入态）、导航按钮三个入口；懒加载 + 缓存 + loading/空/失败三态（失败可重试并给 `/notes/` 兜底）；按标题前缀 > 标题包含 > 标签/分类 > 描述打分，当前语言优先，最多 8 条；combobox + listbox + `aria-activedescendant`，↑↓/Home/End/Enter，IME 组合态不劫持 Enter，关闭后焦点返回触发元素。
- [x] **入口接线**：Navbar 增加「搜索 ⌘K」触发按钮（桌面 40px、移动端 44px、≤360px 收成图标），Layout 在 `<slot />` 后挂 `client:idle`。
- [x] **死交互层清理**：用可达性脚本（根 = `src/pages/**` + 配置入口，解析 import / export-from / 动态 import / `import.meta.glob`，支持路径别名）确认后删除 `src/components/` 下 **40 个零引用组件，约 272.5KB 源码**：搜索面板与高级搜索、悬浮目录与控件、两份显示设置、主题/壁纸/布局切换、下拉菜单系列、Live2D/Spine 看板娘、音乐管理、粒子/樱花特效、AI 聊天、知识图谱、归档与分页、旧文章卡片/页面、Hero、推荐阅读等；`src/components/README.md` 同步更新。
- [x] **连带清理**：`src/data/concepts.ts`（唯一消费者已删）、`scripts/generate-knowledge-graph.ts` 与 `src/constants/knowledge-graph.json`（消费组件已删，构建步骤一并移除）、`main.css` 的 `.search-panel mark` 死 CSS、`env.d.ts` 中已删组件的 window 全局类型。

验收：`pnpm check`（191 文件，0 error / 0 warning）、`pnpm type-check`、完整 `pnpm build`（102 页）通过；`/api/palette.json` 200、35KB、77 条；首页 2 个岛（语言切换 + 命令面板），文章页也带触发按钮；预览下 `/`、`/notes/`、`/library/`、中英文章页、`/world/`、`/about/` 与 `palette.json` 均 200。

未验证：命令面板未做浏览器/真机/屏幕阅读器/真实 IME 实测，⌘K 是否被浏览器抢占、`client:idle` 水合前的点击竞态、320–390px 换行只做了静态推演。

仍未做：配置层空壳与布局叶子组件已在 Phase 20 处理，SideBar 组件与 widget 目录已在 Phase 21 移除。

## Phase 20：内容时效信号、配置层空壳与布局叶子组件（2026-10-18 完成）

- [x] **内容时效自动推进**：首页「最近」日期改为 `max(now.updatedAt, 最新文章 published)`（按时区格式化，不会因 UTC 偏移错一天）；`now.ts` 的 `updatedAt` 更新到 10-18，mission 改写为当前真实在做的六篇数据卷与旧笔记重读。
- [x] **「翻开一篇旧笔记」数据驱动**：删除硬编码文章 id 与两段手写导语，规则改为 `pinned → 最早 reflection → 全站最早`，自动排除最近两篇；导语统一取 `description`（上一轮已重写为 60–90 字）。
- [x] **启用「最后更新」提示**：`showLastModified: true`，文章页原来的旧 `card-base` 卡片改为细窄提示条（左侧 2px 品牌强调色、13px 正文、无圆角无阴影），保留 `data-date`/`data-threshold` 与客户端阈值逻辑。
- [x] **配置层空壳清理**：删除 `pioConfig`、`aiChatConfig`、`effectsConfig`（樱花）、`musicConfig` 四组配置与对应类型，同步 config/types barrel、`setting-utils` 的樱花 getter、i18n 的 `sakuraEffect` 键（6 个文件对称删除）；`global.d.ts` 从 100 行缩到 16 行，只留仍有生产者的 `swup` 与 `table-of-contents`；连带删除零引用的 `sakura-manager.ts` 与 `backlinks.ts`。
- [x] **布局叶子组件精简**：`MainGridLayout` 移除 4 个在当前配置下恒不渲染的组件（`BackgroundPlayer`、`TypewriterText`、`ScrollDownIndicator`、`CategoryBar`）及其分支与变量；`SideBar` 移除 `Music` widget；删除 6 个组件文件。收尾时连带清掉 `sidebarConfig` 的 music 条目与类型、`backgroundWallpaper` 的 typewriter 配置与类型、`layout-styles.css` 的滚动指示器 CSS、`components/README.md` 与 `architecture.md` 的失效条目。
- [x] **依赖清理未落地**：`l2d-widget` / `pagefind` 已无代码引用，但离线环境无法重建 lockfile（需要 registry metadata），按「锁文件不得漂移」原则**保留在 package.json**，联网后执行 `pnpm remove l2d-widget pagefind` 即可。

验收：`pnpm check`（175 文件，0 error / 0 warning）、`pnpm type-check`、完整 `pnpm build`（102 页）通过；构建产物核对首页日期为 `2026.10.18`、旧笔记为 pinned 的 1B 全链路、导语来自 description、文章页提示条带 `hidden` 与阈值脚本；`scroll-down-indicator`/`typewriter`/`music-player`/`category-bar` 的 CSS 已不在产物中，配置层符号在产物 JS 中为 0；预览下首页、文章页、英文文章页、`/notes/`、`/api/palette.json`、`/about/` 均 200。

未验证：没有浏览器实测，提示条的实际显隐（依赖客户端按天数计算）与首页新导语的视觉节奏未肉眼确认。

仍未做：`docs/features.md` 保留历史功能设想，其中提到的 Live2D/樱花等已下线，属历史记录不再回溯。

## Phase 21：侧边栏组件、i18n 死键与依赖下线（2026-10-18 完成）

- [x] **侧边栏组件移除**：`MainGridLayout` 删除 `SideBar` import 与 5 处渲染块，替换为与构建产物逐字一致的静态占位（`#left-sidebar-dynamic` / `#right-sidebar-static` / `#right-sidebar-dynamic`），保留 `#main-grid` 的 class 计算、`data-sidebar-*` 与 `#grid-class-carrier`（Layout 脚本仍读这些）。删除 `SideBar.astro`、9 个 sidebar widget、`toc-utils.ts`，以及只被 Calendar 使用的 `/api/allPostMeta.json`。文章页 HTML 体积与改前一致，说明渲染没有变化。
- [x] **i18n 死键清理**：315 个键里有 232 个零引用（音乐、番组、追番、归档、标签页、搜索面板、赞助、壁纸/主题/特效设置、分页、旧版 404 文案等），从 `i18nKey.ts` 与 5 个语言文件对称删除，剩 83 个；`type-check` 通过即证明 5 份翻译表完整。
- [x] **依赖下线**：`l2d-widget`、`pagefind` 从 `package.json` 移除并用 `pnpm remove` 重建锁文件（此前离线环境只能保留），node_modules 中已不存在。
- [x] **收尾残留**：再删 3 个零引用组件（`WidgetLayout` / `ButtonTag` / `ButtonLink`）与 `widget-responsive.css`；`env.d.ts` 去掉已无生产者的 TOC 全局；`components/README.md`、`architecture.md` 同步。

验收：`pnpm check`（160 文件，0 error / 0 warning）、`pnpm type-check`、完整 `pnpm build`（102 页）通过；产物核对三处侧栏占位逐字保留、文章页 HTML 与改前同长、`/api/allPostMeta.json` 返回 404、widget/allPostMeta 引用为 0、`widget-title`/`sidebar-toc` 的 CSS 已消失；预览下首页、中英文章页、`/notes/`、`/api/palette.json`、`/api/note-search.json`、`/about/` 全 200。

仍未做：`sidebarConfig` / `responsive-utils` 的网格 class 计算仍保留（`#main-grid` 的 `data-sidebar-*` 与 `#grid-class-carrier` 仍被 Layout 脚本读取）；彻底去掉需要同时改 Layout 的脚本与首屏 class 生成，风险大于收益。`docs/features.md` 不回改。

## Phase 22：数据卷通俗化与后续栏目规划（2026-10-20）

- [x] 重写第一卷六篇，以具体问题、直觉、算例、设计边界与对照问题组织正文。
- [x] 保留数据集范围、版本口径、原始引用与核心推导，区分教学设定和真实结果。
- [x] 规划第二至第七卷，重点转向 SFT、偏好学习、推理强化学习和工具调用训练。
- [x] 完成站点构建、公式、概念网络与桌面／手机阅读检查。

详见 [栏目规划](./llm-training-series-plan.md) 和 [数据卷改写记录](./pretraining-data-research.md)。后续卷次为拟写内容，不新增网站导航模块。

## Phase 23：大模型训练研究笔记七卷写作（已完成，2026-10-23）

七卷、42 篇范围已确认，按卷推进来源核对、正文、技术审校和页面验证，不设 token 预算。详见[栏目规划](./llm-training-series-plan.md)与[执行记录](./llm-training-series-progress.md)。

- [x] 第一卷六篇读者友好改写与验收。
- [x] 第二卷前两篇：分词与训练目标；教学算例核对、构建及桌面/手机检查。
- [x] 第二卷后四篇：前向计算、batch、优化稳定性、小模型实验；整卷来源、算例和页面验收。
- [x] 第三卷前三篇：任务示范、消息与模板、监督位置；来源、算例和页面验收。
- [x] 第三卷后三篇：指令数据、LoRA/QLoRA、效果与退化评价；整卷验收。
- [x] 第四至第七卷正文与逐卷验收（2026-10-23 更新：42/42；此前 21/42 为历史进度）。
- [x] 全七卷最终交叉审校与稳定版本统一验收：42 篇中文与 42 份英文、168 个阅读状态通过，见最终验收记录。

## Phase 24：架构减脂、学术杂志视觉收口与五项互动实验扩展（2026-10-21 完成）

- [x] **布局与遗留债务清理**：`Layout.astro`（1,319 → 393 行）与 `MainGridLayout.astro`（857 → 121 行）精简，补首屏同步 `lang` 与 `[data-i18n-en]` 预解析消除英文闪烁；移除 12 个零引用遗留组件/工具/样式文件，合并字体栈 CSS 变量。
- [x] **视觉与文章页收口**：统一导航右侧控件簇触控尺寸与边框语言；重构 `License.astro` 与 `BibTeXCitation.astro` 为鼠尾草学术出版物尾注风格；去除 `posts/[...slug].astro` 重复 `render(entry)`。
- [x] **阅读与检索交互增强**：`CommandPalette.svelte` 增加搜索词 `<mark>` 高亮与水合前排队唤醒；`ArticleNavigation.astro` 新增移动端底部浮动目录胶囊与抽屉；`SeriesNav.astro` 增加本地已读进度标记；代码块与宽表格补充横向滚动阴影提示。
- [x] **五项实验室与互动展示落地**：
  1. `/lab/cfr-poker/`：Kuhn Poker 反事实遗憾最小化（CFR）实验台（12 信息集策略演化、1:3 诈唬价值比、人机实战对局）并为 `/work/` 与案例页补齐 `ResearchVisual kind="cfr"`。
  2. `/lab/preference-alignment/`：DPO 偏好对齐与 KL 锚定实验台（调节 $\beta$ 与迎合偏差 $\alpha_{\text{bias}}$，展示古德哈特前沿拐点与奖励黑客崩塌）。
  3. `/lab/rope-attention/`：RoPE 旋转位置编码与注意力星盘实验台（6 通道复平面日晷、平移不变性检验、线性插值 PI vs. NTK-Aware 缩放对比）。
  4. `MarginGadget3D.svelte`：交互式三维数学与机械小摆件（三维鞍点曲面与动量逃逸、多频相位浑天仪、UCT 探索与利用力矩天平），支持拖拽旋转与 `prefers-reduced-motion`。
  5. `InteractiveMathEnhancer.astro`：文章页 KaTeX 公式符号跨段落悬停/点击锁定联动高亮与内联温度微调尺。

## Phase 25：高密度学术阅读、机制探针与全感官交互系统（2026-10-21 完成）

- [x] **Web Audio 纯合成机械微音效（Haptics）**：实现零外部音频文件加载（0KB 网络负荷）的原生 Web Audio 振荡器合成系统（快门轻响、齿轮发条齿卡塔声、拨动滑块微阻尼声与机械开关音），全站导航、命令面板、博弈台与 3D 摆件联动，支持 `localStorage` 偏好记忆与减弱动画模式静音。
- [x] **Bret Victor 式动态响应段落（Explorable Reactive Prose）**：研发 `ReactiveProseBlock.svelte` 与 `reactive-prose.ts`，段落内嵌可直接水平拖动数值的桃色交互数字，拖动时正文描述、预测信息熵、困惑度、词元分布柱状图与 KV Cache 显存占用实时联动推演。
- [x] **Gwern & Tufte 学术工效系统**：
  1. `HoverPreviewCard.svelte`：全站内链与文献双向悬浮即时预览卡片，悬停时优雅浮现目标文章摘要、分类、标签与阅读耗时。
  2. `tufte-marginalia.css`：宽屏（$\ge 1280\text{px}$）下学术侧边旁注（Sidenote）与文末脚注右边栏浮动对齐，窄屏平滑回退为内联边框块。
- [x] **Transformer 机制可解释性探针（`/lab/attention-lens/`）**：
  1. 完整实现 Anthropic 论文提出的归纳回路（Induction Circuit L2H4）、前驱词头（L1H1）、重复词头（L1H6）、局部衰减头与注意力水槽（Attention Sink L3H0）。
  2. 交互式 $N \times N$ 因果下三角注意力热力图矩阵、Query-Key 词元聚焦路径与上下文学习复制现象验证。
- [x] **3D 研究岛天象光影周期系统（Celestial Ephemeris）**：
  1. `celestial.ts` 计算基于当地真实时间或手动微调的 24 小时太阳高度角、方位角、天空光色温与雾气投影。
  2. `ExploreWorld.svelte` 接入晨曦薄雾（Dawn）、明朗正午（Noon）、金桃晚霞（Sunset）与静夜星辰（Night）四大天象周期切换。

## Phase 26：四大流派落地与全站知识星系（2026-10-21 完成）

依据 [design-archetypes.md](./design-archetypes.md) 的流派坐标，本轮把方向 A（解释性计算流）与方向 B（活体知识花园流）从原型推进到可用状态，并首次落地流派 4（交互式叙事长卷）。

- [x] **方向 A 深化 · `/lab/transformer-forward/`**：`micro-transformer.ts` 的 `runForwardPass` 新增 `layerNorms`，返回 embedding → MHA → 残差 1 → FFN → 残差 2 → 最终归一化共 6 段的 L2 范数；`TransformerForwardLab.svelte` 以六段能量条显示信号流转，读者可把输出异常定位到具体残差支路。
- [x] **方向 B 深化 · `/graph/` 全站知识星系**：`KnowledgeGraph.svelte` 在 2D 力导向布局之外增加 3D 天体星图模式——80 颗背景星、按节点类型（hub / lab / evergreen / 其他）分层的轨道半径与倾角、Y/X 双轴旋转与透视投影，并提供 2D / 3D 切换与重置视图。
- [x] **方向 B 深化 · 思想生命周期与叠放抽屉**：`digital-garden.ts` 定义 🌱 萌芽 / 🌿 演进 / 🌳 常青 / 🔬 待证 / ⚙️ 实验五级元数据与启发式判定；`GardenMaturityBadge.astro` 渲染徽章；`ArticleLocalGraph.svelte` 提供 Andy Matuschak 式横向叠放抽屉，中英文文章页共用。
- [x] **流派 3 · 全局控制台**：`CommandPalette.svelte` 从纯文章搜索升级为带 all / action / lab / post 分页的控制台，内置全站音效开关、8 个实验室直达与 `> sound` 类快捷提示。
- [x] **流派 4 · Distill 式滚轮长卷（新增方向）**：新增 `ScrollytellingTransformer.svelte` 与 `/essays/transformer-scrollytelling/`，以双栏同步滚轮驱动五幕计算舞台——词元投影流形、RoPE 时空星盘、因果下三角注意力点积、多头归纳回路消融、SwiGLU 前馈门控与采样；`/lab/` 索引注册为第 8 个实验入口。
- [x] **验证**：`pnpm check`（179 文件，0 error / 0 warning / 0 hint）、`pnpm type-check`、`npx tsx --test src/utils/*.test.ts`（35 项全通过）、完整 `pnpm build`（137 页）通过。

## Phase 27：偏好卷收尾、双语回填与交互降级（2026-10-21）

本轮把 Phase 26 的实验方向落到内容与可用性上：第四卷补齐到 6/6，最新两波内容全部接入英文侧，同时修掉三处此前一直存在的真实缺陷（英文界面分类泄漏、图谱画布缺少非视觉路径、命令面板分页语义不完整）。

- [x] **偏好卷整卷完成**：补齐 04 DPO 为什么能直接使用偏好对、05 离线偏好优化与在线采样的区别、06 偏好分更高为什么未必任务更好。第四卷 6/6，累计 24/42。
- [x] **35 篇英文概览回填**：数据卷 6 + 预训练卷 6 + SFT 卷 6 + 偏好卷 6 + 果蝇连接组 11。体例沿用既有 31 篇——开头 Scope note 声明证据边界，3–5 个技术小节，末节指向中文原文与代码。英文概览与中文原文的配对靠 `sourceId` 自动接通，无需改动渲染代码。
- [x] **`/essays/` 索引与入口接线**：新增交互长卷索引页，列出两篇长卷（Transformer 五幕长卷、Strandbeest 机构仿真）并说明「为什么是长卷而不是幻灯片」；接入探索菜单、探索栏目栏与命令面板。
- [x] **英文界面分类泄漏修复**：`category-labels.ts` 增加 `categoryLabelsEn` 与 `getCategoryLabel()`；笔记目录的方向筛选、笔记卡片标签与文章页元信息在英文界面改用英文分类，命令面板的 `palette.json` 同步输出 `categoryEn` 并参与匹配打分。
- [x] **`/graph/` 降级与可达性**：
  1. `prefers-reduced-motion: reduce` 下默认进入 2D 图谱，停止自动旋转与轨道漂移，渲染循环只在交互后运行一段固定的沉降帧数；
  2. 画布离开视口或标签页转入后台时停止渲染循环，交互（缩放、平移、筛选、搜索、模式切换）时按需唤醒；
  3. 画布标记 `aria-hidden="true"`，新增服务端直出的「节点目录」纯文本链接列表，作为键盘、读屏与无 JavaScript 的路径；节点布局的随机量改为确定性伪随机，保证服务端与客户端一致。
- [x] **命令面板分页语义修正**：`role="tablist"` 改为 `role="group"` + `aria-pressed`，与站内其它筛选按钮保持一致（此前的 tablist 缺少 `role="tab"` 与 `aria-selected`，属于不完整的 ARIA 用法）。
- [x] **两个构建期工具**：`scripts/validate-frontmatter.js` 在 `astro build` 之前拦下未加引号却含冒号的 YAML 标量（批量写作英文概览时曾三次导致构建失败，报错只指向列号）；`scripts/audit-internal-links.js` 扫描 `dist/` 里的站内链接并报告失效目标。

### 浏览器验收中另外发现并修掉的两处真实缺陷

- [x] **「节点目录」整块不可见**：纯文本节点列表此前被放在 `.canvas-container` 内部，而该容器是固定 `height: 640px` 的定位上下文，外层 `.knowledge-cosmos-wrapper` 又有 `overflow: hidden`，于是这份列表被完全裁掉——服务端 HTML 里存在、读屏仍能读到，但页面上看不见。已把 `<details>` 移到画布容器之外成为同级节点，改为正常文档流。截图 `docs/screenshots/essays-graph-2026-10-22/graph-node-index-1440.png`。
- [x] **成熟度徽章在英文界面不跟随**：`data-i18n-zh` / `data-i18n-en` 两条翻译路径都以 `data-i18n` 键为选择器钩子，而笔记目录卡片与 `GardenMaturityBadge.astro` 的若干元素只有语言属性、没有键，因此客户端切换语言时这些文案保持中文（英文界面上仍显示「常青基石 / 萌芽想法」）。已补齐 `data-i18n` 键，数字花园抽屉内的阶段列表另用 `data-i18n-html="true"` 承载富文本。
- [x] **Strandbeest 系列 4 篇中文与 4 篇英文概览按上游仓库复核后发布**：这 8 篇此前一直是 draft，正文里的若干说法（后端是否公开托管、ArUco 追踪代码位置、是否存在独立求解器 K2 及其与 MuJoCo 的一致性）已与上游仓库 `meiluosi/strandbeest-evolution` 不符。复核后改写为仓库当前状态并解除 draft，其中「十二个演化时期」一文的 Pregluton 一节按现已上线的官方页面重写（此前该页为空，只能引 Wikipedia）。发布口径同时更正为「不含未运行实验的结论」。

### 关于渲染循环帧率的一条记录

验收时用独立探针页测得：`/graph/` 画布每帧重绘 78 个节点、80 颗背景星与全部标签，单帧绘制耗时 **3.2 ms**（2400×1280 后备缓冲，设备像素比 2）。据此判断该页并不「吃满一个核心」，先前把自动化交互超时归因于画布负载是不成立的——真实原因是自动化工具的可操作性检查与持续动画页面互相等待。为降低 CPU 曾尝试加 30 fps 帧率上限并按真实时间步长做运动补偿，但 2D 拓扑模式的力学路径在当时无法在浏览器里验收，于是按「不改没有验证过的分支」的原则整体回退，保留原有 60 fps 行为。/graph/ 的 2D 拓扑视图本轮未复验。

## Phase 28：学术打印规范、英文 RSS 与解释性计算深化（2026-10-22 完成）

本轮聚焦工效减脂、分发渠道完备与端侧可解释性计算深化（第一步与第二步全部落地）：

- [x] **修复 `/graph/` sitemap 排除**：`astro.config.mjs` 中的 `redirectOnlyPaths` 移除 `"/graph/"`，全站认知星系与知识图谱页面正式进入 `dist/sitemap-0.xml` 供搜索引擎抓取。
- [x] **修复知识图谱 Hub 链接跳转**：`src/pages/graph.astro` 中核心知识 Hub 从旧的 301 重定向地址 `"/categories/"` 改为 `getCategoryUrl(hub.category)`，点击直达 `/notes/?topic=...`，消除无谓网络重定向。
- [x] **删除 3 个零引用死样式文件**：删除 `src/styles/photoswipe.css`（859B）、`src/styles/scrollbar.css`（1.9KB）、`src/styles/toc.css`（2.9KB），保持样式源码清晰。
- [x] **编写论文级 `@media print` 学术打印样式**：新增 `src/styles/print.css` 并挂载至 `main.css`。剥离网页 UI、导航、抽屉、音频按钮与命令面板，纯白底墨黑字纸质对比；防止标题孤行与公式/代码块跨页断裂；Tufte 边注安全内联，外部超链接自动展开真实 URL。
- [x] **新增 `/en/rss.xml` 英文独立订阅源**：新增 `src/pages/en/rss.xml.ts`，自动匹配源文章已发布的英文译文，输出规范英文 RSS 2.0 摘要源（10KB）；并在 `Layout.astro` 补充 `<link rel="alternate" type="application/rss+xml" title="枫语 (English)">`。
- [x] **Bret Victor 风格可拖拽段落深度扩展（方案 A）**：`reactive-prose.ts` 与 `ReactiveProseBlock.svelte` 升级为支持 `sampling` / `bpe` / `dpo` 三大模式：
  1. `bpe` 模式：支持拖动词表规模 $V$ 与文本字符数，联动计算压缩比、切分 Token 数、Embedding 参数量与 FP16 显存；
  2. `dpo` 模式：支持拖动 $\beta$ 系数与成对对数概率差，联动计算隐式奖励差 $\Delta \hat{r}$、DPO 目标损失、反向传播梯度权重与 KL 漂移；
  3. 深度嵌入预训练分词篇（`2026-10-20-llm-training-pretrain-01-tokenization.md`）与偏好对齐篇（`2026-10-20-llm-training-preference-04-dpo.md`）。
- [x] **端侧轻量 Python / WebAssembly 运行沙盒（方案 B）**：研发 `CodeSandbox.svelte` 与 `InteractiveProseEnhancer.astro`：
  1. 零首屏负载（0KB 初始包体积）：仅在读者点击「▶ 运行算例」时通过单例模式按需拉取 Pyodide WebAssembly 运行时；
  2. 捕获 Python `sys.stdout` 与错误回溯，提供终端黑底风格输出框、运行耗时统计与代码重置；
  3. 联动 Web Audio 快门音效 `playShutterTick()` 与成功轻音 `playMechanicalTick()`；
  4. 深度嵌入分词手算篇（BPE 三文本频次统计与合并验证）与 DPO 篇（两词表损失与梯度手算验证）。
- [x] **验收**：`pnpm check`（182 文件，0 error / 0 warning / 0 hint）、`pnpm type-check`、37 项端侧单元测试全通过、完整 `pnpm build`（184 页）通过，4710 个内部链接审计 0 断链。

## Phase 29：智能问题演进史（2026-10-23 启动）

用户确认将研究方案维护到项目文档，第一、二轮已分别验收。执行依据：[研究与维护方案](./intelligence-history-plan.md)；当前成果：[20路线统一入口](./intelligence-history/index.html)及[C—D—E联合验收](./intelligence-history/cde-validation.md)，此前[第二轮样板](./intelligence-history/round-two/README.md)，[第一轮工作台](./intelligence-history/README.md)保留为候选基线。各轮独立验收，不把主张审校记为全文阅读或独立复现。

- [x] **第一轮：候选全景与资料盘点**。覆盖已有 20 条路线，建立 106 个独立候选事件、106 份来源及 9 个编辑性比较问题；逐路线记录核心问题、分支、已有材料、检索词和资料缺口。已打开并核对 17 份来源的书目、摘要或片段，全文深读为 0。
- [x] **可编辑时间线与文献工作台**。以 `atlas.json` 为统一维护入口，生成 SVG 等比例时间线、Mermaid 顺序图、可筛选的离线交互图、逐路线底稿和阅读队列；古典文本另列，译本年不冒充原作年，时间先后不宣称历史影响。
- [x] **第二轮：三条样板审校（2026-10-25）**。学习效率与泛化9个节点，控制论、强化学习与主动推断8个，具身、生态与动力系统8个。完成25份指定片段阅读笔记、25条定位主张、16条分类型关系、六张可编辑图和三个可运行教学算例；图、台账与文献夹统一重生成。版本差异、反方材料与历史断层见[审校记录](./intelligence-history/round-two/review.md)。
- [x] **第三轮：全路线扩展与交叉审校（指定范围已验收，2026-10-27）**。补各分支关键问题、失败与争论；核实继承、反驳、并行和融合关系，形成跨路线比较。
  - [x] **A批指定范围已验收（2026-10-27）**：34个新增或补读节点、41份来源记录、109条定位主张、33条关系、三张问题卡和两组版本对照；符号9节点、连接主义10节点、概率因果与世界模型10节点、认知架构6节点。完成三个样板的指定补链、Baird六状态独立复现和统一审校；14张路线图、笔记与工作台从数据同步生成。范围、未读部分和证据见[A批计划](./intelligence-history/round-three-a/plan.md)及[验收记录](./intelligence-history/round-three-a/validation.md)。A—E五批均已完成指定范围，原文深度与版本缺口继续保留。
  - [x] **B批能力与评价已验收（2026-10-27）**：三条路线，25节点、29来源、73定位主张、20关系；6张路线SVG、可编辑Mermaid、三张问题卡、两项A批回访和一项精确合成方法核验。详见[B批计划](./intelligence-history/round-three-b/plan.md)与[验收记录](./intelligence-history/round-three-b/validation.md)。选读、版本缺口与独立验证分别记录。
  - [x] **C批生物与发展**：三条路线，19节点记录、19来源、40主张、15关系；三张问题卡、三项回访和观察／干预的精确合成核验。见[C批验收](./intelligence-history/round-three-c/validation.md)。
  - [x] **D批社会与文化**：三条路线，22节点记录、23来源、39主张、16关系；三张问题卡、三项回访和共享误差的精确合成核验。古典文本另列年代与译本。见[D批验收](./intelligence-history/round-three-d/validation.md)。
  - [x] **E批解释边界**：四条路线，25节点记录、25来源、50主张、20关系；三张问题卡、三项回访和顺序效应的精确合成核验。见[E批验收](./intelligence-history/round-three-e/validation.md)。
  - [x] **全局覆盖与共同问题索引**：20条路线已有审校序列，第二轮至E批150条节点记录去重为144个节点，15张问题卡、11项回访；25个首轮候选无同ID匹配，第四轮复核后其中1个有章节部分覆盖、24个无对应作品阅读映射，另保留各来源的未读部分。建立[可筛选入口](./intelligence-history/index.html)、[覆盖与缺口表](./intelligence-history/coverage.md)、[问题索引](./intelligence-history/questions-index.md)和可编辑总图。C—E三项方法核验、交互与项目检查均通过。
- [x] **第四轮：综合研究与网站联动（2026-10-28指定范围完成）**。围绕共同问题形成跨路线综述、开放问题和实验或论证方案；逐阶段推进，见[执行计划](./intelligence-history/round-four/plan.md)。
  - [x] **阶段1：比较框架与证据审计**。建立八维比较框架；完成《从表现到机制的证据边界》：6条结论、6个案例、10份来源记录、21条定位主张、两组SVG／Mermaid、6项精确检查和3项未执行研究设计。当前索引修订与历史快照分开，桌面／手机交互及工程检查通过。见[工作台](./intelligence-history/round-four/index.html)和[验收](./intelligence-history/round-four/validation.md)。
  - [x] **阶段2：系统边界与发展**。完成[第二篇综合工作台](./intelligence-history/round-four/boundaries/index.html)：9份原典选读、27条定位主张、6条条件性结论、5个比较案例、3组SVG／Mermaid、8项精确检查和3项未执行研究设计。历史快照保留，当前阅读增量单列；见[验收](./intelligence-history/round-four/boundaries/validation.md)。
  - [x] **阶段3：意义与意识解释**。完成[第三篇综合工作台](./intelligence-history/round-four/meaning/index.html)：12份选读、40条定位主张、6条条件性结论、5类案例、6项预测比较、3组SVG／Mermaid与11项精确检查；3项设计未执行。版本、原分析／探索分析和理论方回应分别记录；见[验收](./intelligence-history/round-four/meaning/validation.md)。
  - [x] **阶段4：综合修订与网站联动（指定范围完成）**。完成[18条结论共同复核](./intelligence-history/round-four/integration/review.md)，13条收紧措辞、5条保留；31条来源版本与范围统一，9项设计排序。正式网站原生研究入口、三篇综述、31个来源页与可编辑图已接入路线、文献架及文章入口；三处关联演示已纠偏。见[验收](./intelligence-history/round-four/integration/validation.md)。

- [x] **第五轮：路线演进时间线网站化（指定范围完成，2026-10-28）**。以学习效率、心理测量、心灵哲学为样板，将20条路线的169条路线节点展示接入正式网站；同ID去重为144个节点，原批次阅读状态不变。节点展示问题、方法、前提、限制、原文定位和下一问，图中年份不暗示历史影响；图与数据可编辑。见[执行计划](./intelligence-history/round-five/plan.md)与[验收](./intelligence-history/round-five/validation.md)。

- [x] **第六轮：关键缺口与反方补读**。24项第五轮基线候选按核心问题排序；4项候选获得指定片段阅读映射，另选2项竞争解释。新节点、来源、历史关系与第四轮结论增量均分层保存，时间线与文献入口同步。见[第六轮](./intelligence-history/round-six/README.md)及[验收记录](./intelligence-history/round-six/validation.md)。

验证范围与结果见[第一轮验收](./intelligence-history/validation.md)、[第二轮验收](./intelligence-history/round-two/validation.md)、[A批验收](./intelligence-history/round-three-a/validation.md)、[B批验收](./intelligence-history/round-three-b/validation.md)、[CDE联合验收](./intelligence-history/cde-validation.md)、[第四轮第一阶段验收](./intelligence-history/round-four/validation.md)、[第二阶段验收](./intelligence-history/round-four/boundaries/validation.md)及[第三阶段验收](./intelligence-history/round-four/meaning/validation.md)。第四阶段已接入正式网站的本地项目；保留研究文档与离线工作台，未执行公网部署。

## 版本记录

### 2026-10-18：侧边栏组件、i18n 死键与依赖下线

- [x] MainGridLayout 移除 SideBar 与 5 处渲染块，替换为逐字一致的静态占位；删除 SideBar + 9 个 widget + toc-utils + allPostMeta 接口。
- [x] i18n 从 315 个键精简到 83 个（删除 232 个零引用键，6 个文件对称）。
- [x] `l2d-widget`、`pagefind` 从依赖中移除并重建锁文件。
- [x] 再删 3 个零引用组件与 widget-responsive.css，清理 env.d.ts 与 README/architecture。

### 2026-10-18：时效信号、配置空壳与布局叶子组件

- [x] 首页「最近」日期随最新文章自动推进，旧笔记入口改为 pinned/reflection 数据驱动，删除硬编码。
- [x] 启用「最后更新」提示并改成品牌细窄提示条（阈值 30 天，客户端计算）。
- [x] 删除 pio/aiChat/effects/music 四组配置与类型、樱花运行时与反链工具，global.d.ts 100→16 行。
- [x] MainGridLayout 移除 4 个恒不渲染的叶子组件与 6 个组件文件，连带清掉 sidebar/typewriter/scroll-down 残留。

### 2026-10-18：⌘K 命令面板与死交互层清理

- [x] 新增 `/api/palette.json`（77 条、35KB）与 `CommandPalette.svelte`：⌘K / `/` / 导航按钮三入口，键盘与 ARIA 完整，懒加载三态兜底。
- [x] 清理 40 个零引用交互组件（约 272.5KB 源码）并同步 README。
- [x] 连带删除 concepts.ts、知识图谱生成脚本与产物、死 CSS 与 window 全局类型。

### 2026-10-18：动效系统与交互一致性

- [x] motion token 上移 `:root`，全站动效刻度统一；页面过渡改为 old 淡出 + new 静态。
- [x] 删除 main.css 里约 130 行已死的主题 View Transition 与 `.is-theme-transitioning`。
- [x] 触屏 hover 粘滞修复、补 `:active` 按压反馈、reduce 下清零位移；入场 620ms→320ms。
- [x] 阅读进度与目录高亮改成 rAF + 几何缓存，滚动回调内无布局读取。
- [x] 首页研究岛 compact 版去水合，首页不再加载 ExploreWorld chunk 与 Svelte 岛运行时。
- [x] 笔记搜索补 placeholder 与清除空态；文献架筛选写入 URL 并支持后退还原。
- [x] 新增零特异性焦点环与 skip link。

### 2026-10-18：英文概念图、系列导航与相关阅读

- [x] 概念词表补英文：英文文章概念层从 22/17 提升到 46/44（MCTS、Actor-Critic 概览），中文侧几乎不变。
- [x] 清掉加粗节点名的尾随标点，并把 Scope note 之类版块名排除出概念图。
- [x] 新增 `series` / `seriesOrder` 元数据，6 篇数据卷用 `SeriesNav` 串成「第 n/6 篇」。
- [x] 新增按共同标签的相关阅读窄栏（可开关、与阅读链去重），60 个文章页命中。

### 2026-10-17：文章阅读链

- [x] 新增 `buildReadingChain()`：同主题优先、发布时间兜底，替换掉只在 frontmatter 里生效的旧上／下一篇。
- [x] 中英文章页共用 `ReadingChain.astro`，英文页首次获得文章间导航。
- [x] 分类展示名抽到 `src/data/category-labels.ts`，笔记目录与阅读链共用。
- [x] 加密文章不进入链；最早／最新一篇回落到「全部文章」。

### 2026-10-17：英文文章页并入品牌阅读排版

- [x] 抽出 `ArticleBrandStyle.astro`，中英文章页共用同一份阅读页全局皮肤。
- [x] 英文页补字数／阅读时长、`PostMetadata`、版次标签与统一标题层级。
- [x] 修复英文页只有 KaTeX 标记、没有 KaTeX 样式的问题（9 篇含公式）。
- [x] 英文页补 `BlogPosting` JSON-LD（`inLanguage: en` + `translationOfWork`）。

### 2026-10-17：入口修复、死负载清理与文章页头部

- [x] 分类/标签死链改指 `/notes` 真实筛选参数，补 `popstate` 与移动端 44px 触控尺寸。
- [x] 开启悬停预取，删除 Layout 中的 Swup 死层与进度条。
- [x] 下线 15MB Live2D 资源与 1.6MB Pagefind，RSS 改摘要，sitemap 清 301 壳。
- [x] 收窄封面图 glob，移除孤儿 chunk，文章页 CSS 22→4 个。
- [x] 文章页元信息对比度与标签换行、目录断点 1180、锚点偏移修正。
- [x] 补 hreflang 与中文文章页的静态英文入口，重写 30 条 description。

### 2026-10-16：紧凑页头与文章双视图

- [x] 标题区仅保留标题，移除文章索引、首页与作品页的重复推荐。
- [x] 文献架与文章卡片统一为鼠尾草绿、暖白透明配色。
- [x] 文章默认卡片展示，支持列表切换与显示偏好记忆，共用搜索及筛选。
- [x] 完成静态检查、类型检查、生产构建和桌面／手机交互验证。

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2025-07 | v0.1 | 初始博客功能路线图 |
| 2026-09 | v1.0 | 将路线图切换为个人品牌网站重建计划 |
| 2026-09-26 | v1.1 | 补齐研究档案状态层、3D 无 WebGL 直达入口，并完成站内链接审计 |
| 2026-09-26 | v1.2 | 品牌页面加入中英切换；文章保留原文并提示正文语言；复核 Astro 与生产构建 |
| 2026-09-26 | v1.3 | AlphaZero 五子棋文章新增完整英文版本和双向切换；笔记目录支持英文标题与原文语言标记 |
| 2026-09-26 | v1.4 | 两篇英文正文进入 Pagefind 英文索引；新增 Kuhn Poker CFR 英文版本 |
| 2026-09-26 | v1.5 | 新增 YLearn 因果推断英文概览，明确模拟数据、结果和原文代码边界 |
| 2026-09-27 | v1.6 | 新增 1B 本地模型全链路英文概览；目录和档案入口随语言直达对应文章，并调整英文窄屏首页排版 |
| 2026-09-27 | v1.7 | 修复英文文章受旧语言偏好影响的问题；中文原文入口显式切换中文；研究方向说明及关联笔记在英文模式下同步本地化 |
| 2026-09-27 | v1.8 | 新增 LLM 推理优化英文概览及双语原文入口；英文文章数量增至 5 篇，Pagefind 英文索引同步更新 |
| 2026-09-27 | v1.9 | 英文界面下未翻译中文文章的字数、阅读时长、引用、推荐和评论标签同步本地化 |
| 2026-09-27 | v1.10 | 品牌页面切换时同步更新浏览器标题与页面描述；404 页面支持双语 |
| 2026-09-27 | v1.11 | 补齐品牌导航、RSI/AGI 互动地图与头像的无障碍英文名称；案例继续严格以现有文章为事实边界 |
| 2026-09-27 | v1.12 | 新增 Actor–Critic 算法家族英文概览与中英原文跳转；英文文章增至 6 篇，并纳入 Pagefind 英文索引 |
| 2026-09-28 | v1.13 | 笔记索引的分类、标签、日期和筛选区无障碍名称支持中英切换 |
| 2026-09-28 | v1.14 | 首页 Now 按研究问题列出来源文章；3D 研究地图与降级入口共享方向数据并支持双语标签 |
| 2026-09-28 | v1.15 | 首页精选项目与研究档案的项目类型、研究方向和待补充状态同步支持中英切换 |
| 2026-09-28 | v1.16 | 研究档案按研究方向展示的关联笔记在英文模式仅保留已有译文的入口，避免跳回中文原文 |
| 2026-09-28 | v1.17 | 新增 LLM Agent 英文技术概览，覆盖工具、记忆、规划、多智能体与评估，并明确不构成个人实现或 RSI 结果证明 |
| 2026-09-29 | v1.18 | 新增 MARL 与 DQN CartPole 英文概览；区分合作任务伪代码和文章所报告的 CartPole 指标，补明各自证据边界 |
| 2026-09-29 | v1.19 | 新增 Transformer 架构与 LLM 训练英文概览；把架构示例、训练配方和实测结果分开表述 |
| 2026-09-29 | v1.20 | 新增 LoRA/PEFT 英文概览；保留原文 GSM8K 对比并标明缺少复现材料，不将报告值写成已验证结果 |
| 2026-09-29 | v1.21 | 新增 RAG 与提示工程英文概览；区分教学示例和实测证据，并说明检索/提示评估的边界 |
| 2026-09-29 | v1.22 | 新增多模态大模型英文概览；将 CLIP、BLIP 与视觉语言模型代码标为架构示例，补充数据与评估边界 |
| 2026-09-29 | v1.23 | 新增 MCTS 英文概览，并把搜索基础接入强化学习研究方向及首页 Now 内容映射 |
| 2026-09-29 | v1.24 | 收紧强化学习笔记匹配规则，修复 3D 研究地图键盘焦点提示并在画布失焦时清除移动键 |
| 2026-09-29 | v1.25 | 新增统计学与假设检验英文概览，并将实验设计和效果量记录接入因果评估方向 |
| 2026-09-29 | v1.26 | 新增 DUCG 建模流程英文概览；因果评估方向的笔记入口扩展为三篇，分别覆盖建模、模拟数据与统计检验 |
| 2026-09-29 | v1.27 | 新增模型基础强化学习英文概览，并接入反馈学习与自我改进方向的内容入口 |
| 2026-09-29 | v1.28 | 修复 isolatedDeclarations 导出类型声明，Astro 检查、TypeScript 检查与完整静态构建均通过 |
| 2026-09-29 | v1.29 | 修正 3D 首页桌面与手机标题换行、压低研究岛过曝亮度，实测中英移动导航及键盘进入研究档案 |
| 2026-09-29 | v1.30 | 修复触控取消被误判为点击；减少动态效果时按需重绘，并为 WebGL 上下文丢失提供直达链接降级 |
| 2026-09-29 | v1.31 | 为 12 篇仍有实质正文的中文文章新增英文概览；31 篇已发布中文文章均验证了英文回链并进入双语搜索索引，两个未发布占位正文继续不翻译 |
| 2026-09-29 | v1.32 | 为 1B 模型、AlphaZero 五子棋、Kuhn Poker CFR 和 YLearn 建立双语来源型案例详情页；首页与研究档案连接案例页，原文入口保留；未有记录的职责、运行结果和配图继续标为待补证据 |
| 2026-09-29 | v1.33 | 全站英文笔记目录核对发现卡片摘要未随标题切换；补上英文摘要更新、入门卡片类型标签双语化，并为中文原文提示补上视觉标记 |
| 2026-09-29 | v1.34 | 预览核对英文首页时，修复研究地图初次载入的状态提示语言，并让首页精选笔记日期跟随中英文界面格式 |
| 2026-09-29 | v1.35 | 更新架构文档中的品牌首页、Now、3D 研究线、案例数据入口与实际构建管线，注明每类内容的维护位置 |
| 2026-09-29 | v1.36 | WebGL 不可用或上下文丢失时，将研究画布移出键盘与指针交互，并阻止后续渲染调度，让直达链接成为可用降级入口 |
| 2026-09-29 | v1.37 | 将 Work 方向页的锚点、双语标题说明、标签和笔记检索词并入研究线索数据，统一 Now、3D 地图与方向页的维护入口 |
| 2026-09-30 | v1.38 | 启动轻快配色探索；建立颜色无关的品牌动效控制，接入分层入场、章节渐入和稳定的卡片反馈，完成静态检查、控制逻辑核验与完整构建 |
| 2026-09-30 | v1.39 | 按用户选择落实 B「鼠尾草与桃色」：集中管理配色变量，统一品牌与文章浅色主题，更新研究岛材质、光照和标签，以及分享预览图与 favicon；主要文字组合对比度均超过 4.5:1 |
| 2026-09-30 | v1.40 | 补齐笔记筛选反馈与空状态、原生页面过渡、Now 研究线动画和研究岛首帧渐入；代码与构建核验通过，浏览器视觉验证仍待解除工具阻碍 |
| 2026-09-30 | v1.41 | reduced-motion 覆盖文章页并恢复旧入场元素可见性，语言按钮增加焦点框；再次通过检查与构建，完成品牌与案例保留审计，目标待视觉验收 |
| 2026-09-30 | v1.42 | 恢复真实浏览器验收；修复移动研究岛裁切、窄屏双语导航与中文正文误用标题字号；实测筛选/焦点/双语文章往返并保存桌面与手机截图，完成 Phase 5 |
| 2026-09-30 | v1.43 | 全站研究叙事去除“真实做过”“个人实现”等证明式措辞；首页、研究档案与四个案例页改为问题、方法、观察、下一步和 RSI 连接的自然表达 |
| 2026-09-30 | v1.44 | 根据简历完善双语个人经历；新增正文搜索与阅读返回、文章目录及延伸阅读理由、研究岛预览、因果互动实验和联系页复制邮箱；Now 不变 |
| 2026-10-04 | v1.45 | 完成 RSI 导读、三条阅读路线、六项文献架、实验室和 MCTS 互动实验，以及双语概念预览、正文反向链接与本地阅读收藏；完成正式构建与桌面、手机验证 |
| 2026-10-21 | v1.46 | 沉淀四大顶级流派设计哲学；上线第七实验室 `/lab/transformer-forward/`（端侧微模型 Forward Pass 单步推演与神经回路消融）；启用全站数字花园思想生命周期（🌱/🌿/🌳/🔬）与 `/graph/` 全站力导向交互知识星系 |
| 2026-10-21 | v1.47 | 流派 4 落地：`/essays/transformer-scrollytelling/` Distill 式五幕滚轮长卷；`/graph/` 增加 3D 天体星图模式；命令面板升级为 all / action / lab / post 全局控制台；前向推演实验新增逐层激活能量范数 |


### 2026-10-23：第五卷首篇

- [x] 强化学习问题建模中文正文与英文概览、算例及页面验收。
- [x] 修复 320px 中英文导航溢出，保留所有功能按钮。
- [x] 第五卷后五篇与第六、七卷（后续已完成，见下方逐篇记录）。

- [x] 第五卷第二篇：奖励检查器中英正文、正文代码反例及穷举验证、公式与页面验收。第五卷尚余四篇。

- [x] 第五卷第三篇：策略梯度与 baseline 中英正文、有限差分和采样枚举、公式与页面验收。第五卷尚余三篇。

- [x] 第五卷第四篇：PPO-Clip、GAE 与边界处理中英正文，数值及页面验收完成。第五卷尚余两篇。

- [x] 第五卷第五篇：GRPO 组内信号、标准差与长度权重中英正文及验收。第五卷尚余效果分析一篇。

- [x] 第五卷六篇及英文概览完成，整卷教学计算、链接、公式与 24 个阅读状态验收通过。
- [x] 第六卷工具调用与多步交互、第七卷训练评测与工程复现正文与逐卷验收（后续完成）。

- [x] 第六卷第一篇：工具轨迹中英文、固定环境反例与页面验收。其余五篇待推进。

- [x] 第六卷第二篇：工具接口 SFT、重复监督算例与中英文页面验收。其余四篇待推进。

- [x] 第六卷第三篇：多步奖励、成本与势函数边界算例及中英文页面验收。其余三篇待推进。

- [x] 第六卷第四篇：环境重置、缓存污染与重放契约验证，中英文页面验收完成。其余两篇待推进。

- [x] 第六卷第五篇：失败轨迹、修正监督与选择偏差，中英文和算例验收完成。尚余综合评测。

- [x] 第六卷六篇及英文概览完成，六份教学检查、整卷链接与 24 个阅读状态验收通过。
- [x] 第七卷训练评测与工程复现六篇及逐卷验收。七卷最终核验见末项。

- [x] 第七卷第一篇：训练对照、重复实验与混合偏差算例及中英文页面验收。

- [x] 第七卷第二篇：基准、人评与模型评审的证据边界及校准算例，中英文验收完成。

- [x] 第七卷第三篇：训练成本核算、假设账本与中英文页面验收完成。

- [x] 第七卷第四篇：训练环境、状态恢复与标量执行反例，中英文页面验收完成。

- [x] 第七卷第五篇：模型交付、分层行为回归与算例，中英文页面验收完成。

- [x] 第七卷第六篇：实验结论管理与教学检查；整卷中英文 24 个阅读状态验收通过。
- [x] 七卷最终交叉审校与本地统一验收完成：逐篇矩阵齐全，168 个阅读状态与 7577 个文章链接/锚点通过；未提交推送。
- [x] **3D 数字花园与探索世界 2.0 升级**：融合自然箱庭（梯田草甸/自由跑跳/微风花瓣粒子）、古遗迹机关（悬空浮石踏板联动）与须弥虚空神树（神经脉络/3枚灵感神瞳收集）；复用旧岛台灯/手稿/棋盘/因果拓扑精致道具；双模导航（WASD跑跳 + 957条路线花径自动导引无缝接管）；HUD 集成原神风成就拾取横幅与 `[E]` 交互气泡；清理过渡代码，243页全站构建与测试全部通过。

- [x] **第七轮：关键结论复核**。锁定群体研究代码、数据和更正；完成描述性核对及三／八任务模型尝试，记录三任务与更正值四舍五入一致、八任务差异及其尚未知原因。见[第七轮](./intelligence-history/round-seven/README.md)。
