# 枫语：人文与科技（2026-10-09）

用户在参考调研后确认推进：像一份有作者声音的个人杂志，文章、观察与可以动手的技术作品共同呈现好奇心。保留鼠尾草与桃色、研究岛、实验、阅读路线和长期 RSI 问题；首页不再以训练预算、进度或反复出现的研究宣言组织。

## 内容与设计原则

- 首屏以本人和欢迎语开始，随后是精选随笔、旧笔记与近期文章、可探索作品和简短近况。
- 文字先于操作；章节标题采用宋体/衬线字体，正文留出可读字号和行距，控制英文大写标签与编号。
- 图来自作品和计算，图注交代意义。原始实验记录留在相关研究内容中，不构造新的成果数据。
- 关于页只使用既有个人经历及已公开的研究兴趣；不编造名字来历、生活故事或阅读经历。
- 四份本轮开始前已有修改/新增的 Strandbeest 中英草稿保持原样；新视觉随笔独立编写，不批量发布草稿。

## 参考与取舍

| 本人站点 | 借鉴 |
| --- | --- |
| [Maggie Appleton](https://maggieappleton.com/) | 技术、文化与人的行为相遇；长文、笔记、书籍和近况共存 |
| [Robin Sloan](https://www.robinsloan.com/notes/home-cooked-app/) | 为真实兴趣与身边的人制作软件的个人尺度 |
| [Craig Mod](https://craigmod.com/) | 图片、文字和具体观察共同叙事 |
| [Anthony Fu 的照片页说明](https://antfu.me/posts/photos-page) | 技术实现与记忆、兴趣出现在同一篇文章 |
| [见字如面](https://hiwannz.com/archives/1277.html) | 从生活细节出发，保留疑问与观点变化 |

以上是参考内容组织与表达的依据，本站配色、文案与代码独立实现。

## 实现维护入口

| 内容 | 维护位置 |
| --- | --- |
| 首页欢迎语、编辑选荐与近期文章 | `src/pages/index.astro` |
| 简短近况 | `src/data/now.ts` |
| 作者介绍、全站导航与阅读排版 | `about.astro`、`Navbar.astro`、`navBarConfig.ts`、`markdown.css` |
| 视觉随笔 | `src/pages/essays/strandbeest.astro` |
| 连杆图与几何计算 | `src/components/features/strandbeest/`、`src/utils/strandbeest-geometry.ts` |
| 岛上纸笔、木桌与站点叙述 | `ExploreWorld.svelte`、`world/WorldPoster.svelte`、`world/create-world.ts` |

首页明确排除草稿与加密文章，近期文章按发表日期选择，作者选荐独立于置顶顺序。首页连杆图在服务端计算；`ExploreWorld compact` 只显示静态场景和入口，不创建 WebGL 画布或加载 Three.js。

## 验收

- `pnpm check`：208 个文件，0 error、0 warning；保留 4 个既有 hint。
- `pnpm type-check`：通过。
- `env AI_SUMMARY_API_KEY= pnpm build`：96 页构建成功，Pagefind 索引 63 页。禁用可选 AI 摘要生成，避免改写原内容。
- `node --import tsx --test src/utils/model-pipeline.test.ts src/components/features/world/navigation.test.ts src/utils/strandbeest-geometry.test.ts`：10 项通过，覆盖 BPE、预算、评测规则、957 条导航路径和连杆几何不变量。
- 构建产物审计：68 页、3701 个内部链接属性，无缺失文件或锚点；含 `data-i18n-href-en`。中文与英文公开文章各 31 篇。
- 草稿保护：四份已有 Strandbeest 草稿的 SHA-1 与本轮起点完全一致；公开路由、搜索 API、知识图谱、站点地图和 RSS 均未收录这些草稿。
- 随笔静态产物：完整四节正文与来源、两幅可见 SVG、八条非空计算路径；不依赖 JavaScript 才能开始阅读。
- 浏览器已验证：中文／英文切换，首页 320px 无横向溢出且无 canvas；连杆播放暂停与方向键，三步讲解、曲柄 40 时不可达提示与 137/360 个闭合相位、重置恢复 360/360；目录筛选与正文搜索组合；研究岛降级详情、Escape 关闭与焦点返回。

- 研究岛三维复验：实际初始化、点选文字桌自动到达、收起纸条、暂停动效、完整／轻量画质切换、跟随开关、回起点和总览均通过；320px 英文无横向溢出，控制栏与展开按钮无重叠。
- 390px 的作者页、作品页、中英文章均无横向溢出。实际计算样式确认中英文正文 16px、行高 31.2px；长文标题采用 30px 衬线字体。
- 本轮诊断并解决一次开发预览故障：并行运行 dev 与构建导致 Astro/Vite 虚拟模块缓存失效；停止 dev，完成最终构建，再启动 dev 后恢复。没有将静态降级画面当作三维验收结果。

响应式验证使用桌面浏览器的窄视口，不等于真实触屏、移动 GPU 或长时间帧率测试。更大模型训练与 Strandbeest 实体制作不属于本轮成果。

构建保留既有的 R 语言高亮回退、大 chunk、重复 robots.txt 和旧重定向页没有 `<html>` 的提示；本轮没有宣称这些历史问题已消除。

## 浏览器截图

以下截图来自本地开发预览，底部可见 Astro 开发工具栏；正式构建不会包含该工具栏。

- [桌面首页](./screenshots/humanities-2026-10-09/home-desktop.jpg)
- [手机首页](./screenshots/humanities-2026-10-09/home-mobile.jpg)
- [手机作者介绍](./screenshots/humanities-2026-10-09/about-mobile.jpg)
- [手机旧文阅读](./screenshots/humanities-2026-10-09/article-mobile.jpg)
- [桌面连杆随笔](./screenshots/humanities-2026-10-09/essay-desktop.jpg)
- [手机英文分步互动](./screenshots/humanities-2026-10-09/essay-study-mobile.jpg)
- [桌面三维研究岛](./screenshots/humanities-2026-10-09/world-desktop.jpg)
