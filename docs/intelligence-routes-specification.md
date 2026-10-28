# 「智能是什么」20 大探索路线家族与认知透镜架构规范 (Intelligence Paradigms & Cognitive Lenses Specification)

> **定位声明**：本文档是博客围绕核心母命题 **「智能是什么 (What Is Intelligence?)」** 进行长期内容组织、交互组件开发与路线维护的架构参考依据。

---

## 一、 核心设计哲学：从单一主线到跨学科全景星图

在早期探索中，我们曾以「学习、世界模型、自主探索、持续发展」四层递进实验安排作为组织线索；然而跨学科视角下，这只是理解智能的一组路线，而非学界公认的唯一充要条件。

为了避免“把某一条工程路线当作智能全部本质”的狭隘偏见，全站内容架构升级为 **三维坐标认知地图**：

1. **6 个问题层次 (Question Levels)**：区分不同学派究竟在回答什么问题（判定与定义、测量与结构、机制与实现、发展与起源、边界与归属、意义与价值）。定义、测量、机制与价值判断不能直接互相替代。
2. **5 大路线聚落 (5 Route Clusters) 与 20 个路线家族 (20 Paradigm Families)**：覆盖从心理学、计算机科学、控制论、神经科学、演化生物学到心灵哲学与东方“知/智/慧”传统的 20 个主要解释家族。
3. **8 条核心分歧轴 (8 Divergence Axes)**：每条路线都不在真空中孤立存在，而是在 8 组根本张力（如「外显行为表现 vs 内部生成机制」、「离散符号操作 vs 连续向量动力学」、「颅内孤立计算 vs 具身-环境-文化耦合」等）中占据独特的坐标位置。

---

## 二、 全站页面与组件拓扑

| 模块 / 路径 | 核心职责 | 关联核心文件 |
| :--- | :--- | :--- |
| **首页星轨引力场** (`/`) | 以开普勒椭圆星轨展示围绕「智能是什么」公转的 6 大母轨道行星，悬停/点击星体可直接跳转对应探索路线与专题实验 | `src/components/features/OrbitHero.astro` |
| **20 路线全景地图** (`/routes/`) | 展示 6 个问题层次、5 大聚落筛选器、20 个路线家族卡片及 8 条根本分歧矩阵 | `src/pages/routes.astro`<br>`src/data/intelligenceRoutes.ts` |
| **单路线深度专题页** (`/routes/[id]/`) | 20 个独立预渲染专题页，包含：**00 专属交互认知透镜**、**01 重要分支体系**、**02 解释价值与盲区**、**03 理论分歧天平**、**04 关联实验室**、**05 收录研究笔记** | `src/pages/routes/[id].astro` |
| **研究笔记与双重过滤** (`/notes/`) | 支持按「5 大智能探索聚落 + 20 路线坐标徽章」实时过滤文章，每篇笔记自动解析并挂载所属路线胶囊 | `src/pages/notes.astro`<br>`src/utils/route-matcher.ts` |
| **三维/二维知识星图** (`/graph/`) | 知识星系节点注入 20 路线元数据，点击文章节点可在侧边抽屉直接跳转至其所属的智能路线专题页 | `src/pages/graph.astro`<br>`src/components/features/KnowledgeGraph.svelte` |

---

## 三、 20 路线全量「交互认知透镜 (Cognitive Lenses)」清单

为了让读者在进入任意一条路线专题页 (`/routes/[id]/`) 的前 30 秒内，不仅通过文字阅读，更能通过**动手调节参数**直觉理解该学派的核心机制与理论边界，我们为全部 20 条路线一对一构建了专属 Svelte 5 交互沙盒（位于 `src/components/features/route-lenses/`）：

### 聚落 I：测量、表现与学习效率 (`measurement`)
| 序号 | 路线 ID (`route.id`) | 路线名称 | 专属交互透镜组件 | 核心可交互原理 |
| :--- | :--- | :--- | :--- | :--- |
| **I** | `behavioral` | 行为表现与操作性路线 | `Lens01Behavioral.svelte` | **图灵操作性黑盒测试器**：对比相同外部回答背后的「查表匹配」与「因果推演」，直观揭示行为不可区分性与内部机制的张力。 |
| **II** | `psychometric` | 心理测量与能力结构路线 | `Lens02Psychometric.svelte` | **$g$ 因素与过程重叠模拟器**：调节底层认知过程共享率，观察流体/晶体/空间任务的正相关流形 ($g$-factor) 如何无需单一物理引擎而涌现。 |
| **III** | `learning-efficiency` | 学习效率与泛化路线 | `Lens03LearningEfficiency.svelte` | **Chollet ARC 少样本先验归纳沙盒**：在 $3\times3$ 网格变换中对比「1 样本核心先验归纳（对称/重力/拓扑）」与「海量样本死记硬背」。 |
| **IV** | `rational-resource` | 理性、适应与资源约束路线 | `Lens04RationalResource.svelte` | **有限理性与最优停止期界沙盒**：调节候选观察窗口阈值 $r$ 与每步思考能耗成本，演示西蒙「满意化 (Satisficing)」与净生态效用曲线。 |

### 聚落 II：表征、推理与控制架构 (`representation`)
| 序号 | 路线 ID (`route.id`) | 路线名称 | 专属交互透镜组件 | 核心可交互原理 |
| :--- | :--- | :--- | :--- | :--- |
| **V** | `symbolic-logic` | 符号、逻辑、搜索与组合路线 | `Lens05SymbolicLogic.svelte` | **物理符号系统与启发式搜索剪枝树**：对比 BFS 蛮力遍历、DFS 盲目深探与 $A^*$ 启发式估值 $f(n)=g(n)+h(n)$ 对指数爆炸分支的剪除。 |
| **VI** | `connectionist` | 连接主义、表征学习与连续优化 | `Lens06Connectionist.svelte` | **深层流形解耦与向量变换沙盒**：调节网络层深 $L$ 与激活函数（非线性 ReLU vs 线性坍缩），观察纠缠样本流形如何被逐步展平并由超平面切分。 |
| **VII** | `causal-world-models` | 概率、因果与世界模型路线 | `Lens07CausalModels.svelte` | **珀尔因果三层阶梯与图切断手术**：在混杂因子 DAG 中切换「观察 $P(Y|X)$」、「干预 $P(Y|do(X))$ 斩断后门路径」与「反事实回溯」。 |
| **VIII** | `cybernetics-rl` | 控制论、强化学习与自主行动 | `Lens08CyberneticsRL.svelte` | **阿什比必要多样性定律稳态舱**：注入外部环境熵增扰动 $D$，调节控制器策略库多样性 $R$，验证 $V(E) \ge V(D) - V(R)$ 负反馈闭环。 |
| **IX** | `cognitive-architectures` | 认知架构与心智模块整合路线 | `Lens09CognitiveArch.svelte` | **ACT-R / Soar 50ms 认知总线循环**：单步推进「感觉缓冲 → 工作记忆瓶颈总线 → 陈述性记忆激活 → 产生式 IF-THEN 规则触发」。 |

### 聚落 III：生命、脑网络与具身演化 (`embodiment`)
| 序号 | 路线 ID (`route.id`) | 路线名称 | 专属交互透镜组件 | 核心可交互原理 |
| :--- | :--- | :--- | :--- | :--- |
| **X** | `neuroscience` | 神经科学、脑网络与生物基质 | `Lens10Neuroscience.svelte` | **P-FIT 顶额整合与神经同调沙盒**：切换静息态与高阶流体推理负荷，观察 dlPFC-IPL 伽马波段 (40Hz) 相位同调与神经代谢效率比。 |
| **XI** | `developmental-openended` | 发展、建构、好奇心与开放式演进 | `Lens11Developmental.svelte` | **皮亚杰平衡化与内在好奇心沙盒**：注入可同化样本与诱发失衡的反例（顺应重构基式），对比平庸区、黄金最近发展区与纯噪声区的学习进展导数。 |
| **XII** | `embodied-dynamical` | 具身、生成（Enactive）与动力系统 | `Lens12EmbodiedDynamics.svelte` | **麦吉尔被动动态行走与形态学计算**：调节斜坡倾角 $\gamma$ 与髋关节被动刚度 $k$，展示零算力（0 FLOPs）下由重力-身体耦合涌现的极限环步态。 |
| **XIII** | `autopoiesis-evolution` | 生命自创生、生物适应与演化路线 | `Lens13Autopoiesis.svelte` | **马图拉纳自创生胞膜与马尔可夫毯**：调节跨膜自由能摄取与环境热力学冲刷，直观呈现内部催化网络与脂质膜边界相互生成的环形因果。 |

### 聚落 IV：群体、文化与物理极限 (`collective`)
| 序号 | 路线 ID (`route.id`) | 路线名称 | 专属交互透镜组件 | 核心可交互原理 |
| :--- | :--- | :--- | :--- | :--- |
| **XIV** | `sociocultural-language` | 社会文化、语言、心智理论与教育 | `Lens14Sociocultural.svelte` | **托马塞洛文化棘轮效应模拟器**：调节代际符号传授保真度与社群连接规模，观察跨代知识如何突破「文化滑脱阈值」实现指数累积。 |
| **XV** | `collective-extended` | 群体智能、延展心智与人机共生 | `Lens15ExtendedMind.svelte` | **历史与当前支持的有限模型**：交叉改变四步输入、当前支持和状态重置；检验同样移除效应与不同历史。读数不表示人类准确率、延迟或认知构成。 |
| **XVI** | `information-physics` | 信息论、压缩、热力学与物理限制 | `Lens16InformationPhysics.svelte` | **蒂什比信息瓶颈与兰道尔能耗边界**：拖动拉格朗日乘子 $\beta$，在信息平面上权衡输入压缩代价 $I(X;T)$、预测有效互信息 $I(T;Y)$ 与热力学擦除耗散。 |

### 聚落 V：心灵、意识与价值智慧 (`mind`)
| 序号 | 路线 ID (`route.id`) | 路线名称 | 专属交互透镜组件 | 核心可交互原理 |
| :--- | :--- | :--- | :--- | :--- |
| **XVII** | `philosophy-of-mind` | 心灵哲学：功能主义、意向性与接地 | `Lens17PhilosophyOfMind.svelte` | **指称映射与锚点**：枚举三符号双射，在无锚点、一个、两个及矛盾约束下保留6／2／1／0个候选；不把映射识别当作理解或意识指标。 |
| **XVIII** | `consciousness-self` | 意识、主观体验、自我模型与反身性 | `Lens18Consciousness.svelte` | **报告与候选信号依赖**：交叉刺激S和报告要求Q，比较N_A=S、N_B=R=S∧Q；均不是体验真值。GNW与IIT分开，不使用人为Φ或普适点火阈值。 |
| **XIX** | `wisdom-values` | 智慧、价值理性与跨文明心性传统 | `Lens19WisdomValues.svelte` | **斯腾伯格智慧平衡 vs 回形针最大化器**：独立调节「工具理性算力」与「跨主体长远价值反思」，揭示高智商低智慧引发的古德哈特目标异化风险。 |
| **XX** | `metaphysics-quantum` | 量子认知、微管量子假说与形而上学 | `Lens20QuantumCognition.svelte` | **量子认知非对易投影次序效应**：在希尔伯特子空间中切换提问测量算符顺序 $P_B P_A |\psi\rangle$ 与 $P_A P_B |\psi\rangle$，演示语境叠加态坍缩与方法论边界。 |

---

## 四、 日常写作与后续维护指南 (Maintenance Workflow)

### 1. 新增研究笔记时如何归入 20 路线？
当使用 `pnpm new-post` 或新建 Markdown 笔记 (`src/content/posts/*.md`) 时：
- **自动映射**：`src/utils/route-matcher.ts` 中的 `resolvePostRoute(post.data)` 会根据文章的 `category`、`tags` 或标题关键词，自动将其归入最匹配的路线家族（并在 `/notes/` 卡片、`/routes/[id]/` 专题页底部以及 `/graph/` 星图中自动建立双向链接）。
- **显式指定标签**：若希望文章精准落入特定路线，可在 Frontmatter 的 `tags` 中包含该路线的代表性关键词（参见 `src/utils/route-matcher.ts` 中的匹配规则表）。

### 2. 如何扩展或升级某一路由的交互实验？
本博客采用 **双层交互体系 (Two-Tier Interactive Architecture)**：
- **Tier 1 — 概念认知透镜 (`src/components/features/route-lenses/LensXX*.svelte`)**：内嵌于每个 `/routes/[id]/` 页面顶部（区块 `00`），通过 `RouteLensDispatcher.svelte` 按 `route.id` 分发，要求轻量、免配置、30 秒内直击理论要害。
- **Tier 2 — 深度算法工作台 (`src/pages/lab/*`、`/world/`、`/rsi/`)**：独立全屏实验台（如因果推演工作台、MCTS 树搜索、RoPE 注意力流形、CFR 博弈、生命演化沙盒）。当新建了一个全屏实验室后，只需在 `src/data/intelligenceRoutes.ts` 对应路线的 `linkedLabs` 数组中追加 `{ nameZh, nameEn, href }`，即可在路线专题页的「区块 04：动手验证」中自动展示入口卡片。

### 3. 校验与构建清单
每次修改路线数据 (`src/data/intelligenceRoutes.ts`) 或透镜组件 (`src/components/features/route-lenses/*.svelte`) 后，提交前请依次执行：
```bash
pnpm check        # Astro & Svelte 组件诊断
pnpm type-check   # TypeScript --isolatedDeclarations 严格类型检查
pnpm build        # 全量静态预渲染构建（验证 20 个 /routes/[id]/ 专题页及 Pagefind 索引生成）
```

### 第四轮审校与原生研究入口（2026-10-28）

XV、XVII、XVIII三项现由 `ReviewedInferenceLens.svelte` 复用先前核验的有限模型，入口在构建期仍按路线选择独立组件。`/research/intelligence/` 提供三篇综述、18条修订对照、31个来源页与后续队列；其他认知透镜仍保留原状态，本次没有为它们追加证据认证。详情见[综合修订验收](./intelligence-history/round-four/integration/validation.md)。
