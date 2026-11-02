/**
 * 在线交互专著与书稿数据定义 (Living Books & Interactive Monographs)
 * 将分散的单篇笔记、20 条智能路线与可交互透镜统筹为体系化、分卷连载的在线专著
 */

export interface BookChapter {
	num: string;
	titleZh: string;
	titleEn: string;
	summaryZh: string;
	summaryEn: string;
	href: string;
	routeBadge?: string;
	hasInteractive?: boolean;
	status?: "published" | "upcoming";
}

export interface BookVolume {
	volNum: string;
	titleZh: string;
	titleEn: string;
	epigraphZh: string;
	epigraphEn: string;
	chapters: BookChapter[];
}

export interface BookSidenote {
	marker: string;
	titleZh: string;
	titleEn: string;
	bodyZh: string;
	bodyEn: string;
	linkHref?: string;
	linkTextZh?: string;
	linkTextEn?: string;
}

export interface LivingBook {
	id: string;
	roman: string;
	titleZh: string;
	titleEn: string;
	subtitleZh: string;
	subtitleEn: string;
	editionBadgeZh: string;
	editionBadgeEn: string;
	status: "completed" | "living";
	completionPercent: number;
	accentColor: string;
	spineColor: string;
	epigraphZh: string;
	epigraphEn: string;
	abstractZh: string;
	abstractEn: string;
	featuredLensId: "embodied" | "connectionist" | "neuroscience" | "world-model";
	volumes: BookVolume[];
	sidenotes: BookSidenote[];
}

export const LIVING_BOOKS: LivingBook[] = [
	{
		id: "what-is-intelligence",
		roman: "MONOGRAPH I",
		titleZh: "何谓智能：20 条路线与心智的本质",
		titleEn: "What Is Intelligence? Twenty Paradigms & The Nature of Mind",
		subtitleZh: "从机械连杆、因果阶梯、自由能瓶颈到自创生心智的跨学科勘测",
		subtitleEn:
			"An Interdisciplinary Survey from Mechanical Linkages and Causal Ladders to Autopoietic Mind",
		editionBadgeZh: "常青连载专著 · v0.9 Living Edition",
		editionBadgeEn: "Living Monograph · v0.9 Edition",
		status: "living",
		completionPercent: 85,
		accentColor: "#2f6b56",
		spineColor: "#1d4436",
		epigraphZh:
			"“定义、测量、内部机制、演化起源与价值智慧，回答的是六个不同层次的追问。任何试图用单一工程路线包办‘智能全部本质’的尝试，都会在它遗漏的盲区里遭遇反噬。”",
		epigraphEn:
			"“Definition, measurement, mechanism, evolutionary origin, and practical wisdom address six distinct layers of inquiry. Any attempt to reduce all of intelligence to a single engineering pipeline will be blindsided by what it leaves out.”",
		abstractZh:
			"本书围绕母命题「智能是什么」展开体系化勘测。全书分为五卷二十章，每一章对应一个解释智能的经典学派家族，并配套一个可动手调节参数的物理/数学认知透镜（Cognitive Lens），引导读者在操作与反驳中建立立体心智地图。",
		abstractEn:
			"Organized into five volumes and twenty interactive chapters, this living monograph surveys the major explanatory traditions of intelligence—each paired with a hands-on mathematical or physical Cognitive Lens.",
		featuredLensId: "embodied",
		volumes: [
			{
				volNum: "第一卷 · VOLUME I",
				titleZh: "表现、心理测量与学习转化效率",
				titleEn: "Performance, Psychometrics & Acquisition Efficiency",
				epigraphZh:
					"从可观测的外显行为出发，追问能力结构的统计共变、少样本先验泛化与有限计算资源下的最优停止。",
				epigraphEn:
					"From observable behavior to latent factor manifolds, few-shot induction, and bounded resource rationality.",
				chapters: [
					{
						num: "Ch.01",
						titleZh: "行为表现与操作性路线：先看它能做什么",
						titleEn: "Behavioral & Operational Criteria",
						summaryZh:
							"图灵测试、操作主义与现代能力维度分级：为何外显行为不可区分性既是工程标尺，又是遮蔽内部机制的黑盒？",
						summaryEn:
							"Turing operationalism and capability evaluation: why behavioral equivalence is both a benchmark and a black box.",
						href: "/routes/behavioral/",
						routeBadge: "Route 01",
						hasInteractive: true,
					},
					{
						num: "Ch.02",
						titleZh: "心理测量与能力结构路线：一般智力 g 与过程重叠",
						titleEn: "Psychometrics, g-Factor & Process Overlap",
						summaryZh:
							"斯皮尔曼 g 因素、流体/晶体智力层级与过程重叠理论：统计上的共同方差为何不等于单一生物学引擎？",
						summaryEn:
							"Spearman's g, CHC hierarchy, and Process Overlap Theory: statistical positive manifold without a monolithic engine.",
						href: "/routes/psychometric/",
						routeBadge: "Route 02",
						hasInteractive: true,
					},
					{
						num: "Ch.03",
						titleZh: "学习效率与泛化路线：Chollet ARC 与核心先验",
						titleEn: "Skill Acquisition Efficiency & Core Priors",
						summaryZh:
							"智能不是静态题库储备，而是借助对称、拓扑、物体性等核心先验在极少样本下合成新程序的转化效率。",
						summaryEn:
							"François Chollet's ARC thesis: intelligence as the conversion efficiency from priors + experience into novel skill.",
						href: "/routes/learning-efficiency/",
						routeBadge: "Route 03",
						hasInteractive: true,
					},
					{
						num: "Ch.04",
						titleZh: "理性、适应与资源约束路线：满意化与生态理性",
						titleEn: "Bounded, Resource & Ecological Rationality",
						summaryZh:
							"西蒙有限理性、Gigerenzer 快速节俭启发式与王培 NARS：当思考本身消耗时间与能量时，何时停止计算才是真聪明。",
						summaryEn:
							"Simon's satisficing, ecological heuristics, and adaptation under insufficient knowledge and resources.",
						href: "/routes/rational-resource/",
						routeBadge: "Route 04",
						hasInteractive: true,
					},
				],
			},
			{
				volNum: "第二卷 · VOLUME II",
				titleZh: "符号逻辑、连续流形、因果阶梯与闭环控制",
				titleEn: "Symbols, Manifolds, Causality & Cybernetic Control",
				epigraphZh:
					"深入认知系统的内部算法机制：离散形式推演与高维向量几何如何对峙？统计相关如何跨越至因果干预与闭环调节？",
				epigraphEn:
					"Inside the algorithmic engine: discrete symbol search vs. continuous manifolds, causal interventions, and homeostasis.",
				chapters: [
					{
						num: "Ch.05",
						titleZh: "符号、逻辑、搜索与组合路线：启发式剪枝艺术",
						titleEn: "Physical Symbol Systems & Heuristic Search",
						summaryZh:
							"纽厄尔与西蒙物理符号系统假说、A* 启发式估值与 MCTS 蒙特卡洛树搜索如何驯服组合爆炸。",
						summaryEn:
							"Newell & Simon's symbol hypothesis and heuristic tree pruning against exponential combinatorial explosion.",
						href: "/routes/symbolic-logic/",
						routeBadge: "Route 05",
						hasInteractive: true,
					},
					{
						num: "Ch.06",
						titleZh: "连接主义、表征学习与连续优化路线：流形解缠",
						titleEn: "Connectionism & Deep Manifold Unfolding",
						summaryZh:
							"分布式向量表征、逐层非线性几何变换与注意力回路如何将纠缠感官空间展平为线性可分超平面。",
						summaryEn:
							"Distributed vector spaces and layered nonlinearities untangling complex sensory manifolds.",
						href: "/routes/connectionist/",
						routeBadge: "Route 06",
						hasInteractive: true,
					},
					{
						num: "Ch.07",
						titleZh: "概率、因果与世界模型路线：珀尔因果三层阶梯",
						titleEn: "Probabilistic World Models & Pearl's Causal Ladder",
						summaryZh:
							"从观察关联 P(Y|X) 跃迁至干预图手术 P(Y|do(X)) 与反事实想象，剖析动态不确定性因果图（DUCG）。",
						summaryEn:
							"Climbing from Rung 1 association to Rung 2 do-calculus intervention and Rung 3 counterfactual imagination.",
						href: "/routes/causal-world-models/",
						routeBadge: "Route 07",
						hasInteractive: true,
					},
					{
						num: "Ch.08",
						titleZh: "控制论、强化学习与自主行动路线：必要多样性定律",
						titleEn: "Cybernetics, Reinforcement Learning & Ashby's Law",
						summaryZh:
							"维纳负反馈闭环与阿什比定律：‘只有多样性才能消解多样性’，智能在环境扰动中守护稳态。",
						summaryEn:
							"Wiener's feedback loops and Ashby's Law of Requisite Variety in sequential agency.",
						href: "/routes/cybernetics-rl/",
						routeBadge: "Route 08",
						hasInteractive: true,
					},
				],
			},
			{
				volNum: "第三卷 · VOLUME III",
				titleZh: "认知架构、脑连接组、发生学与具身自创生",
				titleEn: "Architectures, Connectomes, Development & Embodiment",
				epigraphZh:
					"跳出纯软件算法幻觉，重返系统级认知总线、生物脑网络同调、儿童发生认识论与无芯片的被动具身步态。",
				epigraphEn:
					"Beyond disembodied software: cognitive buses, brain phase synchrony, ontogenetic curiosity, and morphological mechanics.",
				chapters: [
					{
						num: "Ch.09",
						titleZh: "认知架构与心智模块整合路线：ACT-R 与 50ms 认知节拍",
						titleEn: "Integrated Cognitive Architectures (ACT-R / Soar)",
						summaryZh:
							"感知缓冲、工作记忆瓶颈总线、陈述性知识块与程序性产生式规则如何协同运转。",
						summaryEn:
							"Coordinating working memory bottlenecks, declarative chunks, and procedural production cycles.",
						href: "/routes/cognitive-architectures/",
						routeBadge: "Route 09",
						hasInteractive: true,
					},
					{
						num: "Ch.10",
						titleZh: "神经科学、脑网络与生物基质路线：P-FIT 与神经效率",
						titleEn: "Network Neuroscience & P-FIT Synchrony",
						summaryZh:
							"顶额叶整合理论（P-FIT）、伽马波段振荡同调与全脑连接组拓扑如何支撑流体推理。",
						summaryEn:
							"Parieto-frontal phase synchronization and metabolic neural efficiency across connectome hubs.",
						href: "/routes/neuroscience/",
						routeBadge: "Route 10",
						hasInteractive: true,
					},
					{
						num: "Ch.11",
						titleZh: "发展、建构与开放式演进路线：皮亚杰平衡化与好奇心",
						titleEn: "Developmental Epistemology & Intrinsic Curiosity",
						summaryZh:
							"同化、顺应与欧迪耶‘学习进展导数最大化’：自主智能体如何在最近发展区自我驱动成长。",
						summaryEn:
							"Piagetian schema accommodation and Oudeyer's learning-progress curiosity drive.",
						href: "/routes/developmental-openended/",
						routeBadge: "Route 11",
						hasInteractive: true,
					},
					{
						num: "Ch.12",
						titleZh: "具身、生成与动力系统路线：被动行走与形态学计算",
						titleEn: "Embodied, Enactive & Dynamical Systems",
						summaryZh:
							"麦吉尔被动双足步行机与 Strandbeest 仿生连杆：身体材料与重力耦合如何以 0 FLOPs 卸载控制计算。",
						summaryEn:
							"McGeer's passive dynamic walker and morphological computation offloading central control.",
						href: "/routes/embodied-dynamical/",
						routeBadge: "Route 12",
						hasInteractive: true,
					},
					{
						num: "Ch.13",
						titleZh: "生命、自维持与演化路线：自创生胞膜与内生目的",
						titleEn: "Autopoiesis, Basal Cognition & Evolution",
						summaryZh:
							"马图拉纳与瓦雷拉自创生（Autopoiesis）及莱文多尺度基底认知：为什么唯有存在本身脆弱的生命体才拥有真实目的。",
						summaryEn:
							"Maturana & Varela's autopoietic boundary maintenance and the biological origin of intrinsic teleology.",
						href: "/routes/autopoiesis-evolution/",
						routeBadge: "Route 13",
						hasInteractive: true,
					},
				],
			},
			{
				volNum: "第四卷 · VOLUME IV",
				titleZh: "文化棘轮、延展心智与热力学信息物理",
				titleEn: "Cultural Ratchets, Extended Mind & Information Physics",
				epigraphZh:
					"将视野推向个体颅骨之外与物理学底层：语言棘轮如何跨代累积？外部笔记本如何成为心智的一部分？压缩为何等于理解？",
				epigraphEn:
					"Beyond the individual skull and down to statistical physics: cumulative culture, cognitive scaffolding, and Landauer bounds.",
				chapters: [
					{
						num: "Ch.14",
						titleZh: "社会文化与语言路线：托马塞洛文化棘轮效应",
						titleEn: "Sociocultural Ratchet & Shared Intentionality",
						summaryZh:
							"共同注意、高保真教学与符号传统如何防止代际知识滑脱，让人类心智实现指数级累积起飞。",
						summaryEn:
							"How high-fidelity social transmission and linguistic symbols lock in cumulative cultural evolution.",
						href: "/routes/sociocultural-language/",
						routeBadge: "Route 14",
						hasInteractive: true,
					},
					{
						num: "Ch.15",
						titleZh: "群体智能、延展心智与人机共生路线：奥托的笔记本",
						titleEn: "Extended Mind, Cognitive Scaffolding & Symbiosis",
						summaryZh:
							"克拉克与查尔默斯对等原则：当纸笔、算盘与代码解释器同生物脑形成紧密双向回路，认知边界便已跨越颅骨。",
						summaryEn:
							"Clark & Chalmers' Parity Principle: external symbolic artifacts as literal components of mind.",
						href: "/routes/collective-extended/",
						routeBadge: "Route 15",
						hasInteractive: true,
					},
					{
						num: "Ch.16",
						titleZh: "信息论、压缩、热力学与物理限制路线：信息瓶颈",
						titleEn: "Information Bottleneck, Free Energy & Thermodynamics",
						summaryZh:
							"蒂什比信息瓶颈、弗里斯顿自由能原理与兰道尔擦除极限：智能是在热力学能耗约束下提取最小充分统计量的艺术。",
						summaryEn:
							"Tishby's Information Bottleneck, Friston's Free Energy, and Landauer thermodynamic efficiency.",
						href: "/routes/information-physics/",
						routeBadge: "Route 16",
						hasInteractive: true,
					},
				],
			},
			{
				volNum: "第五卷 · VOLUME V",
				titleZh: "心灵哲学、意识全局点火、实践智慧与量子认知",
				titleEn: "Intentionality, Consciousness, Wisdom & Quantum Models",
				epigraphZh:
					"直面智能探索最深邃的终极疆域：语法何以通达语义？无意识模块何以涌现主观体验？高算力如何避免沦为愚蠢的工具异化？",
				epigraphEn:
					"The ultimate philosophical horizon: symbol grounding, conscious ignition, normative wisdom, and contextual projection.",
				chapters: [
					{
						num: "Ch.17",
						titleZh: "心灵哲学：功能主义、意向性与中文房间接地难题",
						titleEn: "Philosophy of Mind, Intentionality & Symbol Grounding",
						summaryZh:
							"塞尔中文房间与哈纳德符号接地问题：走出‘用词典解释词典’的形式语法死循环。",
						summaryEn:
							"Searle's Chinese Room and Harnad's symbol grounding problem: bridging syntax and semantics.",
						href: "/routes/philosophy-of-mind/",
						routeBadge: "Route 17",
						hasInteractive: true,
					},
					{
						num: "Ch.18",
						titleZh: "意识、主观体验与全局工作空间路线：非线性点火",
						titleEn: "Consciousness, Global Workspace Ignition & IIT",
						summaryZh:
							"巴尔斯/迪昂全局神经工作空间（GNW）与托诺尼整合信息度 Φ：跨模块广播与‘意识难问题’。",
						summaryEn:
							"Global Neuronal Workspace phase ignition, Integrated Information Phi, and Chalmers' Hard Problem.",
						href: "/routes/consciousness-self/",
						routeBadge: "Route 18",
						hasInteractive: true,
					},
					{
						num: "Ch.19",
						titleZh: "智慧、价值与跨文化路线：斯腾伯格平衡与回形针警示",
						titleEn: "Practical Wisdom (Phronesis), Values & Alignment",
						summaryZh:
							"亚里士多德实践智慧、儒道佛心性修养与古德哈特定律：为什么高智商绝不自动等于高智慧。",
						summaryEn:
							"Aristotelian Phronesis, Eastern contemplative traditions, and Sternberg's balance theory against instrumental runaway.",
						href: "/routes/wisdom-values/",
						routeBadge: "Route 19",
						hasInteractive: true,
					},
					{
						num: "Ch.20",
						titleZh: "形而上学、非经典计算与量子认知路线：语境非对易投影",
						titleEn: "Quantum Cognition & Non-Classical Foundations",
						summaryZh:
							"严格区分‘类量子希尔伯特空间决策概率模型’与形而上学微管猜想，解析人类判断中的提问次序干涉效应。",
						summaryEn:
							"Separating rigorous Hilbert-space quantum probability models of human judgment from speculative metaphysics.",
						href: "/routes/metaphysics-quantum/",
						routeBadge: "Route 20",
						hasInteractive: true,
					},
				],
			},
		],
		sidenotes: [
			{
				marker: "边注 §1",
				titleZh: "为什么需要 20 条路线而非单一定义？",
				titleEn: "Why 20 Paradigms Instead of One Definition?",
				bodyZh:
					"Legg 与 Hutter (2007) 曾收集了 70 余种智能定义。当我们把‘测量标尺’误认为‘内部机制’、或把‘工具解题’等同于‘实践智慧’时，就会陷入范畴错误。",
				bodyEn:
					"Conflating operational benchmarks with internal causal mechanisms or normative wisdom leads to severe category errors.",
				linkHref: "/routes/",
				linkTextZh: "打开 2D 思想引力象限仪 ↗",
				linkTextEn: "Open 2D Paradigm Compass ↗",
			},
			{
				marker: "边注 §2",
				titleZh: "风吹动力兽 (Strandbeest) 的具身启示",
				titleEn: "Theo Jansen's Strandbeest & Morphology",
				bodyZh:
					"Theo Jansen 用 13 个神圣数字比例构成的塑料管连杆，在海风推动下无需一行代码即可走出防陷沙步态，与本书第 12 章被动行走遥相呼应。",
				bodyEn:
					"Jansen's linkage ratios convert wind into lifelike locomotion with zero microprocessors, echoing Chapter 12.",
				linkHref: "/world/",
				linkTextZh: "进入研究岛操控连杆步态 ↗",
				linkTextEn: "Manipulate Strandbeest in /world/ ↗",
			},
		],
	},
	{
		id: "forging-language-models",
		roman: "MONOGRAPH II",
		titleZh: "从零锻造语言模型：七卷全链路实证手记",
		titleEn:
			"Forging Language Models from Scratch: A Seven-Volume Empirical Monograph",
		subtitleZh:
			"涵盖语料工程、预训练动力学、指令微调、偏好对齐、强化学习推理、工具交互与可复现评测的 42 章系统专著",
		subtitleEn:
			"42 Empirical Chapters Spanning Corpus Engineering, Pretraining Dynamics, SFT, Preference Alignment, RL Reasoning, Tool Use, and Reproducible Evaluation",
		editionBadgeZh: "七卷本全集完结 · v1.0 Complete Edition",
		editionBadgeEn: "7 Volumes Complete · v1.0 Edition",
		status: "completed",
		completionPercent: 100,
		accentColor: "#0284c7",
		spineColor: "#0c4a6e",
		epigraphZh:
			"“不要把大模型当作靠魔法咒语召唤的黑盒神谕。当你亲手切分每一个字节对、计算每一层激活显存、屏蔽每一处监督损失并审计每一条强化学习轨迹时，神话就会还原为精密而诚实的工程因果。”",
		epigraphEn:
			"“Do not treat language models as black-box oracles summoned by incantation. When you trace every BPE merge, activation tensor, loss mask, and RL rollout by hand, mythology dissolves into honest engineering causality.”",
		abstractZh:
			"本专著共七卷四十二章，完整记录了从原始互联网网页清洗到可验证推理智能体交付的全链路研究与工程实证。每一卷均坚持一手论文溯源、精确教学算例与诚实的失败边界分析。",
		abstractEn:
			"Across 7 volumes and 42 chapters, this monograph documents the end-to-end science and engineering of language models—from raw web corpus deduplication to verifiable RL reasoning and tool-use trajectories.",
		featuredLensId: "connectionist",
		volumes: [
			{
				volNum: "第一卷 · VOLUME I",
				titleZh: "预训练数据与可追溯语料工程",
				titleEn: "Pretraining Corpora & Traceable Data Engineering",
				epigraphZh:
					"模型能力的上限首先铸造于语料分布、清洗算子、MinHash 去重与退火配比之中。",
				epigraphEn:
					"Model ceilings are forged in corpus distributions, filtering cascades, deduplication, and annealing mixtures.",
				chapters: [
					{
						num: "Vol.1-01",
						titleZh: "预训练语料全景图：来源、规模与演进谱系",
						titleEn: "The Pretraining Corpus Map",
						summaryZh:
							"剖析 Common Crawl、FineWeb、Dolma 等主流语料的构成比例与质量退化边界。",
						summaryEn:
							"Mapping web crawls, curated domains, and synthetic mixtures in modern pretraining.",
						href: "/posts/2026-10-15-llm-training-data-01-map/",
					},
					{
						num: "Vol.1-02",
						titleZh: "网页抽取、HTML 净化与文本归一化算子",
						titleEn: "Text Extraction & Normalization Pipelines",
						summaryZh:
							"从 WET/WARC 原始流中剥离导航噪声、修复编码坍缩并保留数学与代码结构。",
						summaryEn:
							"Stripping boilerplate from WARC archives while preserving code and mathematical notation.",
						href: "/posts/2026-10-15-llm-training-data-02-cleaning/",
					},
					{
						num: "Vol.1-03",
						titleZh: "启发式规则过滤与轻量模型质量打分器",
						titleEn: "Heuristic & Model-Based Quality Filtering",
						summaryZh:
							"对比 Gopher/C4 规则集与 FastText/KenLM 质量分类器的精度-召回率权衡。",
						summaryEn:
							"Balancing rule cascades against classifier-driven quality filtering.",
						href: "/posts/2026-10-15-llm-training-data-03-quality/",
					},
					{
						num: "Vol.1-04",
						titleZh: "模糊去重（MinHash LSH）、子串去重与评测集防污染",
						titleEn: "Fuzzy Deduplication & Benchmark Decontamination",
						summaryZh:
							"精确推导 Jaccard 相似度下界、局部敏感哈希桶配置与 N-gram 评测集净化。",
						summaryEn:
							"MinHash LSH banding math, suffix-array substring dedup, and test-set decontamination.",
						href: "/posts/2026-10-15-llm-training-data-04-dedup/",
					},
					{
						num: "Vol.1-05",
						titleZh: "多源领域配比优化与训练末期高质退火",
						titleEn: "Domain Mixture Optimization & Mid-Training Annealing",
						summaryZh:
							"DoReMi/RegMix 代理模型权重搜索与退火阶段注入高密度数学/代码语料的损耗曲线。",
						summaryEn:
							"Proxy-model domain weighting and high-quality curriculum annealing.",
						href: "/posts/2026-10-15-llm-training-data-05-mixture/",
					},
					{
						num: "Vol.1-06",
						titleZh: "构建可溯源、可复现的端到端语料流水线",
						titleEn: "Building a Traceable & Reproducible Corpus Pipeline",
						summaryZh: "数据指纹、血缘审计台账与流式分片打包的工程闭环。",
						summaryEn:
							"Data lineage manifests, cryptographic hashing, and deterministic tokenized shards.",
						href: "/posts/2026-10-15-llm-training-data-06-build/",
					},
				],
			},
			{
				volNum: "第二卷 · VOLUME II",
				titleZh: "预训练基座、前向力学与优化器动力学",
				titleEn: "Pretraining Foundations, Forward Mechanics & Optimization",
				epigraphZh:
					"从字节级分词器信息密度到 RoPE 旋转位置编码、RMSNorm 残差流与余弦学习率退火。",
				epigraphEn:
					"From byte-level BPE compression to RoPE attention geometry, residual scaling, and AdamW dynamics.",
				chapters: [
					{
						num: "Vol.2-01",
						titleZh: "分词器设计：BPE 合并算法、词表规模与压缩率审计",
						titleEn: "Tokenization: BPE Merges, Vocabulary Size & Fertility",
						summaryZh:
							"字节级 BPE 算例推导、多语言及代码数字切分策略与嵌入层参数开销。",
						summaryEn:
							"Byte-level BPE mechanics, digit splitting, and compression fertility metrics.",
						href: "/posts/2026-10-20-llm-training-pretrain-01-tokenization/",
					},
					{
						num: "Vol.2-02",
						titleZh: "自回归目标函数、交叉熵损失与困惑度物理意义",
						titleEn: "Autoregressive Objectives, Cross-Entropy & Perplexity",
						summaryZh: "因果掩码下最大似然估计与信息论交叉熵比特率的精确对应。",
						summaryEn:
							"Causal language modeling, next-token likelihood, and bits-per-byte evaluation.",
						href: "/posts/2026-10-20-llm-training-pretrain-02-objective/",
					},
					{
						num: "Vol.2-03",
						titleZh: "现代 Transformer 前向解剖：Pre-RMSNorm、RoPE 与 SwiGLU",
						titleEn: "Modern Transformer Anatomy: RMSNorm, RoPE & SwiGLU",
						summaryZh:
							"逐张量追踪 GQA 分组查询注意力、旋转位置编码与门控前馈层的形状及显存占用。",
						summaryEn:
							"Tensor-by-tensor walkthrough of Grouped-Query Attention, RoPE, and SwiGLU.",
						href: "/posts/2026-10-20-llm-training-pretrain-03-forward/",
						hasInteractive: true,
					},
					{
						num: "Vol.2-04",
						titleZh: "序列拼接打包（Packing）、跨文档掩码与吞吐量优化",
						titleEn: "Sequence Packing, Document Masking & Throughput",
						summaryZh:
							"消除 Padding 算力浪费并防止跨文档注意力串扰的块对角掩码实现。",
						summaryEn:
							"Eliminating padding waste via sequence packing and intra-document attention masks.",
						href: "/posts/2026-10-20-llm-training-pretrain-04-batching/",
					},
					{
						num: "Vol.2-05",
						titleZh: "AdamW 二阶矩、梯度裁剪、Warmup 与 WSD 学习率调度",
						titleEn: "AdamW Moments, Gradient Clipping & WSD Schedules",
						summaryZh:
							"权重衰减解耦、梯度范数尖峰诊断以及 Warmup-Stable-Decay 调度的随时分支优势。",
						summaryEn:
							"Decoupled weight decay, loss spike mitigation, and Warmup-Stable-Decay schedules.",
						href: "/posts/2026-10-20-llm-training-pretrain-05-optimization/",
					},
					{
						num: "Vol.2-06",
						titleZh: "小规模基座预训练完整实验台账与缩放定律核验",
						titleEn: "Small-Scale Pretraining Ledger & Scaling Law Audit",
						summaryZh:
							"在受控算力预算下完成端到端预训练并核验验证集损失收敛轨迹。",
						summaryEn:
							"End-to-end pretraining ledger verifying Chinchilla token-to-parameter trade-offs.",
						href: "/posts/2026-10-20-llm-training-pretrain-06-experiment/",
					},
				],
			},
			{
				volNum: "第三卷 · VOLUME III",
				titleZh: "监督微调（SFT）、对话模板契约与参数高效适配",
				titleEn: "Supervised Fine-Tuning, Chat Templates & Adaptation",
				epigraphZh:
					"预训练赋予模型世界知识分布，监督微调则教会模型以受控角色与对话边界激发这些潜能。",
				epigraphEn:
					"Pretraining builds world representations; SFT establishes conversational framing and instruction following.",
				chapters: [
					{
						num: "Vol.3-01",
						titleZh: "示范数据的本质：对齐表面格式假说（LIMA）与知识边界",
						titleEn:
							"Nature of Demonstrations & The Superficial Alignment Hypothesis",
						summaryZh:
							"剖析少量极高质量示范如何激活基座能力，以及强塞基座未知事实为何诱发幻觉。",
						summaryEn:
							"Why high-quality demonstrations unlock capabilities while unfamiliar facts induce hallucination.",
						href: "/posts/2026-10-20-llm-training-sft-01-demonstrations/",
					},
					{
						num: "Vol.3-02",
						titleZh: "多轮消息结构、特殊控制符（Special Tokens）与对话状态机",
						titleEn: "Multi-Turn Message Framing & Special Control Tokens",
						summaryZh:
							"ChatML 边界标记、System/User/Assistant 角色隔离与停止符截断契约。",
						summaryEn:
							"ChatML role delimiters, turn-taking state machines, and EOS termination.",
						href: "/posts/2026-10-20-llm-training-sft-02-messages/",
					},
					{
						num: "Vol.3-03",
						titleZh: "提示词损失掩码（Prompt Loss Masking）与多轮监督区间",
						titleEn: "Prompt Loss Masking & Multi-Turn Supervision Spans",
						summaryZh:
							"手算 `-100` 忽略索引张量，验证仅对 Assistant 回复计算交叉熵的梯度差异。",
						summaryEn:
							"Computing loss masks over assistant spans to prevent memorizing user prompts.",
						href: "/posts/2026-10-20-llm-training-sft-03-loss-mask/",
					},
					{
						num: "Vol.3-04",
						titleZh: "指令数据合成、复杂度进化（Evol-Instruct）与多样性去重",
						titleEn: "Instruction Synthesis, Evol-Instruct & Diversity",
						summaryZh: "平衡指令广度、推理深度与拒答边界的数据集配比工程。",
						summaryEn:
							"Curating instruction complexity, task diversity, and safety refusal boundaries.",
						href: "/posts/2026-10-20-llm-training-sft-04-data/",
					},
					{
						num: "Vol.3-05",
						titleZh: "全参微调 vs LoRA 低秩适配：秩空间特征与灾难性遗忘",
						titleEn: "Full Fine-Tuning vs. LoRA Low-Rank Adaptation",
						summaryZh:
							"对比全参数更新与低秩矩阵增量在学习容量、显存占用与旧知识保留上的实测表现。",
						summaryEn:
							"Comparing full-weight updates against low-rank adapters on capacity and forgetting.",
						href: "/posts/2026-10-20-llm-training-sft-05-adaptation/",
					},
					{
						num: "Vol.3-06",
						titleZh: "SFT 阶段评测陷阱：格式迎合偏见与指令遵循精准度",
						titleEn: "Evaluating SFT: Format Bias vs. Instruction Fidelity",
						summaryZh: "区分‘学会拉长排版讨好裁判’与‘真正满足多重约束指令’。",
						summaryEn:
							"Disentangling verbose formatting bias from genuine constraint satisfaction.",
						href: "/posts/2026-10-20-llm-training-sft-06-evaluation/",
					},
				],
			},
			{
				volNum: "第四卷 · VOLUME IV",
				titleZh: "人类偏好对齐、奖励建模与直接偏好优化（DPO）",
				titleEn: "Preference Alignment, Reward Modeling & DPO",
				epigraphZh:
					"当‘写出一篇好回答’比‘判别哪篇回答更好’更难时，比较偏好看似捷径，却暗藏奖励黑客与分数异化。",
				epigraphEn:
					"Comparative preference unlocks alignment where generation is hard, yet courts reward hacking and length bias.",
				chapters: [
					{
						num: "Vol.4-01",
						titleZh: "为什么监督微调不够？从示范模仿转向比较偏好信号",
						titleEn: "Beyond Imitation: Comparative Preference Signals",
						summaryZh:
							"分析 SFT 在开放生成与安全边界上的局限，以及成对比较偏好的信息优势。",
						summaryEn:
							"Why pairwise comparison succeeds where likelihood imitation plateaus.",
						href: "/posts/2026-10-20-llm-training-preference-01-feedback/",
					},
					{
						num: "Vol.4-02",
						titleZh: "偏好对构造（Chosen / Rejected）、标注信度与长度偏差净化",
						titleEn: "Constructing Preference Pairs & Debiasing Length",
						summaryZh: "识别偏好数据中的阿谀奉承（Sycophancy）与冗长偏见。",
						summaryEn:
							"Controlling length bias, sycophancy, and annotator disagreement in preference pairs.",
						href: "/posts/2026-10-20-llm-training-preference-02-pairs/",
					},
					{
						num: "Vol.4-03",
						titleZh: "Bradley-Terry 奖励模型推导、标量打分头与校准审计",
						titleEn: "Bradley-Terry Reward Modeling & Calibration",
						summaryZh:
							"从成对胜率对数似然推导奖励模型损失，检验跨分布泛化能力。",
						summaryEn:
							"Deriving the Bradley-Terry pairwise ranking objective and auditing OOD calibration.",
						href: "/posts/2026-10-20-llm-training-preference-03-reward-model/",
					},
					{
						num: "Vol.4-04",
						titleZh: "DPO 直接偏好优化：将奖励函数解析代入 KL 约束最优策略",
						titleEn:
							"Direct Preference Optimization (DPO) Mathematical Derivation",
						summaryZh:
							"一步步完成从 RLHF 目标函数到 DPO 隐式奖励对数比损失的闭式数学变换。",
						summaryEn:
							"Closed-form derivation mapping KL-constrained RLHF into implicit reward log-ratios.",
						href: "/posts/2026-10-20-llm-training-preference-04-dpo/",
					},
					{
						num: "Vol.4-05",
						titleZh: "离线偏好（Offline）vs 在线采样对齐（Online Iterative）",
						titleEn: "Offline vs. Online Iterative Preference Alignment",
						summaryZh:
							"剖析策略分布偏移如何导致离线 DPO 退化，以及在线采样如何修补分布外漏洞。",
						summaryEn:
							"How policy distribution shift degrades offline DPO and why online rollouts matter.",
						href: "/posts/2026-10-20-llm-training-preference-05-offline-vs-online/",
					},
					{
						num: "Vol.4-06",
						titleZh:
							"高偏好分≠高正确率：奖励黑客（Reward Hacking）与古德哈特边界",
						titleEn: "Preference Score vs. Truth: Auditing Reward Hacking",
						summaryZh:
							"实证展示模型如何通过语气自信与排版华丽骗取高分却丧失事实与数学严谨性。",
						summaryEn:
							"Empirical audit of Goodhart's Law when proxy reward diverges from factual correctness.",
						href: "/posts/2026-10-20-llm-training-preference-06-score-vs-correctness/",
					},
				],
			},
			{
				volNum: "第五卷 · VOLUME V",
				titleZh: "强化学习推理、可验证奖励（RLVR）与 PPO/GRPO",
				titleEn: "Reinforcement Learning for Reasoning, Verifiers & GRPO",
				epigraphZh:
					"在数学证明与代码合成等具备客观验证器的领域，强化学习让模型从‘模仿人类笔迹’跃迁至‘自主试错搜索’。",
				epigraphEn:
					"Anchored by deterministic verifiers, RL shifts models from imitating human transcripts to autonomous search.",
				chapters: [
					{
						num: "Vol.5-01",
						titleZh: "将语言生成建模为马尔可夫决策过程（Token-Level MDP）",
						titleEn: "Language Generation as a Token-Level MDP",
						summaryZh:
							"状态、动作、轨迹回报与 KL 散度正则项在自回归生成中的严格形式化。",
						summaryEn:
							"Formalizing states, token actions, trajectory returns, and KL penalties.",
						href: "/posts/2026-10-23-llm-training-rl-01-formulation/",
					},
					{
						num: "Vol.5-02",
						titleZh: "可验证奖励（RLVR）：数学解析验证器与代码沙盒单元测试",
						titleEn: "Reinforcement Learning with Verifiable Rewards (RLVR)",
						summaryZh:
							"为什么确定性规则/编译器验证器能彻底免疫神经网络奖励模型的伪造漏洞？",
						summaryEn:
							"Why deterministic math parsers and unit-test sandboxes resist reward hacking.",
						href: "/posts/2026-10-23-llm-training-rl-02-verifiers/",
					},
					{
						num: "Vol.5-03",
						titleZh: "REINFORCE 策略梯度、高方差困境与基线减除（Baseline）",
						titleEn: "Policy Gradients, Variance Reduction & Baselines",
						summaryZh:
							"通过具体数值算例演示对数概率梯度如何放大高回报轨迹并抑制坏采样。",
						summaryEn:
							"Worked numerical examples of score-function gradients and baseline subtraction.",
						href: "/posts/2026-10-23-llm-training-rl-03-policy-gradient/",
					},
					{
						num: "Vol.5-04",
						titleZh: "PPO 近端策略优化：广义优势估计（GAE）与裁剪代理目标",
						titleEn:
							"PPO: Generalized Advantage Estimation & Clipped Surrogates",
						summaryZh:
							"剖析 Actor-Critic 四模型显存开销与重要性采样比率裁剪区间 [1-ε, 1+ε]。",
						summaryEn:
							"Actor-Critic memory footprint, GAE lambda returns, and importance ratio clipping.",
						href: "/posts/2026-10-23-llm-training-rl-04-ppo/",
					},
					{
						num: "Vol.5-05",
						titleZh: "GRPO 组相对策略优化：省去价值网络的组内归一化优势",
						titleEn:
							"GRPO: Group Relative Policy Optimization Without a Critic",
						summaryZh:
							"同一提示采样 G 条轨迹、以组内均值标准差归一化优势函数的数学推导与显存红利。",
						summaryEn:
							"Eliminating the Critic network via group-normalized relative advantages.",
						href: "/posts/2026-10-23-llm-training-rl-05-grpo/",
					},
					{
						num: "Vol.5-06",
						titleZh: "推理强化学习评测：Pass@1 vs Pass@k 与思维链长度膨胀审计",
						titleEn: "Evaluating Reasoning RL: Pass@k & CoT Length Inflation",
						summaryZh:
							"区分强化学习究竟是涌现了全新解题路径，还是锐化了基座已有的采样分布。",
						summaryEn:
							"Auditing whether RL discovers novel reasoning paths or sharpens existing Pass@k support.",
						href: "/posts/2026-10-23-llm-training-rl-06-evaluation/",
					},
				],
			},
			{
				volNum: "第六卷 · VOLUME VI",
				titleZh: "工具调用、环境交互契约与多步智能体训练",
				titleEn: "Tool Use, Environment Contracts & Multi-Step Agent Training",
				epigraphZh:
					"连接延展心智路线：让语言模型走出封闭权重，在编译器、搜索引擎与外部工具的真实反馈中闭环修正。",
				epigraphEn:
					"Connecting to Extended Mind: coupling neural intuition with compilers, search engines, and external sandboxes.",
				chapters: [
					{
						num: "Vol.6-01",
						titleZh: "工具交互轨迹结构：Thought / Action / Observation 状态流",
						titleEn:
							"Tool-Use Trajectories: Thought, Action & Observation Loops",
						summaryZh:
							"ReAct 交错轨迹协议、结构化函数签名 Schema 与环境返回观察帧。",
						summaryEn:
							"Formalizing interleaved reasoning, JSON tool calls, and environment observations.",
						href: "/posts/2026-10-23-llm-training-tool-01-trajectories/",
					},
					{
						num: "Vol.6-02",
						titleZh:
							"多步轨迹监督：为什么必须屏蔽外部环境 Observation 的损失？",
						titleEn: "Trajectory Supervision & Observation Loss Masking",
						summaryZh:
							"证明若不掩码工具返回文本，模型会退化为死记硬背造假执行结果的幻觉器。",
						summaryEn:
							"Why failing to mask environment observations teaches models to hallucinate fake tool outputs.",
						href: "/posts/2026-10-23-llm-training-tool-02-supervision/",
					},
					{
						num: "Vol.6-03",
						titleZh:
							"结果奖励 vs 过程步骤信用分配（Step-Level Credit Assignment）",
						titleEn: "Outcome Rewards vs. Step-Level Credit Assignment",
						summaryZh:
							"长程工具调用链中的稀疏奖励归因、无效冗余调用惩罚与过程验证。",
						summaryEn:
							"Resolving credit assignment across multi-hop tool invocations.",
						href: "/posts/2026-10-23-llm-training-tool-03-rewards/",
					},
					{
						num: "Vol.6-04",
						titleZh: "可复现工具沙盒契约：幂等性、超时熔断与状态快照",
						titleEn: "Deterministic Tool Sandboxes, Idempotency & Snapshots",
						summaryZh: "构建支持高并发 RL 采样回放的无副作用容器化环境。",
						summaryEn:
							"Engineering side-effect-free, replayable execution sandboxes for RL rollouts.",
						href: "/posts/2026-10-23-llm-training-tool-04-environments/",
					},
					{
						num: "Vol.6-05",
						titleZh: "工具调用典型失效模式：参数幻觉、死循环重试与错误恢复",
						titleEn:
							"Failure Taxonomy: Schema Hallucination, Retry Loops & Recovery",
						summaryZh:
							"构造注入报错反例的训练轨迹，教会智能体根据异常堆栈自主修正调用。",
						summaryEn:
							"Training agents to recover from compiler tracebacks and API errors.",
						href: "/posts/2026-10-23-llm-training-tool-05-failures/",
					},
					{
						num: "Vol.6-06",
						titleZh: "工具智能体端到端评测：端到端解决率、调用开销与安全越界",
						titleEn: "Evaluating Tool Agents: Task Success, Latency & Safety",
						summaryZh: "综合评估任务完成率、工具调用次数效率与沙盒逃逸风险。",
						summaryEn:
							"Measuring end-to-end resolution, tool-call parsimony, and sandbox safety.",
						href: "/posts/2026-10-23-llm-training-tool-06-evaluation/",
					},
				],
			},
			{
				volNum: "第七卷 · VOLUME VII",
				titleZh: "实验设计、统计显著性、成本核算与可复现交付",
				titleEn:
					"Experimental Design, Statistical Rigor & Reproducible Delivery",
				epigraphZh:
					"没有控制变量、方差区间与开源复现脚本的跑分榜单只是营销噪音。科学始于诚实的实验台账。",
				epigraphEn:
					"Leaderboards without controlled variables, confidence intervals, and reproduction scripts are marketing noise.",
				chapters: [
					{
						num: "Vol.7-01",
						titleZh: "单变量消融实验设计、随机种子方差与最小有效基线",
						titleEn: "Controlled Ablations, Seed Variance & Strong Baselines",
						summaryZh:
							"规避同时改动多个超参的归因混乱，建立严格的消融对照矩阵。",
						summaryEn:
							"Isolating causal training variables and quantifying multi-seed variance.",
						href: "/posts/2026-10-23-llm-training-eval-01-experiments/",
					},
					{
						num: "Vol.7-02",
						titleZh:
							"评测证据等级：McNemar 配对检验、Bootstrap 置信区间与污染排查",
						titleEn:
							"Statistical Evidence: McNemar Tests & Bootstrap Intervals",
						summaryZh:
							"为什么提升 0.8% 往往只是采样噪声？如何计算配对题目级别显著性。",
						summaryEn:
							"Item-level paired significance testing and confidence bounds for benchmark deltas.",
						href: "/posts/2026-10-23-llm-training-eval-02-evidence/",
					},
					{
						num: "Vol.7-03",
						titleZh: "全生命周期算力与显存成本核算（FLOPs、VRAM 与吞吐帕累托）",
						titleEn: "Compute, VRAM & Latency Cost Accounting",
						summaryZh:
							"精确核算预训练、SFT 与 RL 采样阶段的算力账单与推理延迟比。",
						summaryEn:
							"Full-lifecycle FLOPs accounting across pretraining, SFT, and RL rollouts.",
						href: "/posts/2026-10-23-llm-training-eval-03-costs/",
					},
					{
						num: "Vol.7-04",
						titleZh: "确定性复现工程：锁定算子内核、数据顺序与权重校验和",
						titleEn: "Deterministic Reproduction: Kernels, Shards & Checksums",
						summaryZh: "排查浮点数非结合律引起的跨卡漂移，交付一键复现脚本。",
						summaryEn:
							"Taming floating-point non-associativity and locking training reproducibility.",
						href: "/posts/2026-10-23-llm-training-eval-04-reproduction/",
					},
					{
						num: "Vol.7-05",
						titleZh: "模型卡（Model Card）、安全边界声明与开源制品交付标准",
						titleEn: "Model Cards, Failure Boundaries & Artifact Delivery",
						summaryZh:
							"诚实披露模型不擅长领域、分布外退化条件与复现实验包结构。",
						summaryEn:
							"Transparently documenting out-of-distribution failure modes and release artifacts.",
						href: "/posts/2026-10-23-llm-training-eval-05-delivery/",
					},
					{
						num: "Vol.7-06",
						titleZh: "七卷本总跋：当语言模型训练回归「智能是什么」的母问题",
						titleEn:
							"Epilogue: Returning from LLM Engineering to 'What Is Intelligence?'",
						summaryZh:
							"总结七卷工程全链路能解决什么，又有哪些根本问题必须交还给因果、具身与自创生路线。",
						summaryEn:
							"Synthesizing what scaling and RLVR achieve—and what remains for causality and embodiment.",
						href: "/posts/2026-10-23-llm-training-eval-06-conclusions/",
					},
				],
			},
		],
		sidenotes: [
			{
				marker: "边注 §1",
				titleZh: "旗舰全链路交互实验室",
				titleEn: "Interactive Model Pipeline Studio",
				bodyZh:
					"本书配套开发了全屏交互工作台 `/lab/model-pipeline/`，支持实时调节词表大小、RoPE 旋转角、SFT 损失掩码与 GRPO 组内归一化优势。",
				bodyEn:
					"Manipulate BPE merges, RoPE rotation angles, SFT loss masks, and GRPO group normalization live in the companion studio.",
				linkHref: "/lab/model-pipeline/",
				linkTextZh: "启动全链路模型实验台 ↗",
				linkTextEn: "Launch Model Pipeline Lab ↗",
			},
			{
				marker: "边注 §2",
				titleZh: "独立开源训练代码库",
				titleEn: "Standalone Training Repository",
				bodyZh:
					"位于 `projects/small-model-lab/` 的 PyTorch 独立训练包，与本书七卷内容逐一对应，包含完整单元测试与可复现脚本。",
				bodyEn:
					"The companion PyTorch package in `projects/small-model-lab/` mirrors all seven volumes with runnable scripts.",
				linkHref: "/work/local-1b-model-workflow/",
				linkTextZh: "查看小模型实证课题档案 ↗",
				linkTextEn: "View Small-Model Case Study ↗",
			},
		],
	},
	{
		id: "drosophila-connectome",
		roman: "MONOGRAPH III",
		titleZh: "果蝇全脑连接组：从生物突触拓扑到具身智能",
		titleEn:
			"The Drosophila Connectome: From Synaptic Topology to Embodied Agency",
		subtitleZh:
			"基于全脑突触级连接组（FlyWire）的预注册计算神经科学实验、蘑菇体多巴胺舱室审计与阴性结果诚实报告",
		subtitleEn:
			"Preregistered Computational Neuroscience Audits of the Whole-Brain FlyWire Connectome, Mushroom Body Compartments, and Honest Negative Results",
		editionBadgeZh: "11 章预注册专著 · v1.0 Preregistered Edition",
		editionBadgeEn: "11 Chapters Complete · v1.0 Preregistered",
		status: "completed",
		completionPercent: 100,
		accentColor: "#d97706",
		spineColor: "#78350f",
		epigraphZh:
			"“只有约 14 万个神经元、5000 万个突触的果蝇大脑，如何在毫瓦级功耗下完成气味觅食、空间导航与实时避障？当我们拒绝用事后调参粉饰曲线、严格执行预注册检验时，生物连接组给出了最冷峻也最迷人的答案。”",
		epigraphEn:
			"“How does a fruit fly brain of ~140,000 neurons and 50 million synapses navigate, forage, and adapt on microwatts of power? Under strict preregistered controls and honest negative-result reporting, the biological connectome speaks.”",
		abstractZh:
			"本专著共三卷十一章，围绕果蝇全脑连接组（FlyWire / 蘑菇体回路）展开严格的计算神经科学实证。全书建立在预注册协议、随机拓扑打乱基线（Shuffled Controls）与多巴胺舱室调制之上，并专门以整章篇幅诚实报告阴性结果与过度宣称审计。",
		abstractEn:
			"Across 3 volumes and 11 chapters, this monograph audits computational capabilities of the fruit fly connectome against degree-preserving shuffled controls—culminating in a rigorous negative-result summary.",
		featuredLensId: "neuroscience",
		volumes: [
			{
				volNum: "第一卷 · VOLUME I",
				titleZh: "生物连接组环路、预注册协议与觅食群体涌现",
				titleEn: "Connectome Circuits, Preregistration & Foraging Swarms",
				epigraphZh:
					"在动手跑任何花哨任务前，先锁死拓扑度分布零模型与预注册证伪判据。",
				epigraphEn:
					"Before running task benchmarks, lock in degree-preserving null models and falsifiable preregistration criteria.",
				chapters: [
					{
						num: "Ch.01",
						titleZh: "果蝇连接组解剖框架：投射神经元、肯尼恩细胞与蘑菇体输出",
						titleEn:
							"Circuit Anatomy: PNs, Kenyon Cells & Mushroom Body Output",
						summaryZh:
							"梳理 PN → KC 稀疏高维扩张编码与 MBON/DAN 多巴胺价值调制环路。",
						summaryEn:
							"Mapping sparse high-dimensional Kenyon cell expansion and dopaminergic compartment modulation.",
						href: "/posts/2026-10-21-fly-connectome-01-circuit-and-framework/",
						routeBadge: "Route 10",
					},
					{
						num: "Ch.02",
						titleZh: "预注册实验协议与防自欺审校清单",
						titleEn:
							"Preregistered Experimental Protocol & Anti-HARKing Controls",
						summaryZh:
							"确立度保持随机重连基线（Degree-Preserving Rewiring）与多重比较校正标准。",
						summaryEn:
							"Establishing degree-preserving null topologies and strict statistical thresholds.",
						href: "/posts/2026-10-21-fly-connectome-02-protocol-and-review/",
					},
					{
						num: "Ch.03",
						titleZh: "气味梯度觅食与多智能体群体聚集涌现实验",
						titleEn: "Odor-Gradient Foraging & Multi-Agent Swarm Emergence",
						summaryZh:
							"测试生物突触拓扑在闭环感官运动觅食与群体交互中的鲁棒性。",
						summaryEn:
							"Evaluating sensorimotor chemotaxis and collective aggregation driven by connectome weights.",
						href: "/posts/2026-10-21-fly-connectome-03-foraging-and-swarm/",
						routeBadge: "Route 12",
					},
				],
			},
			{
				volNum: "第二卷 · VOLUME II",
				titleZh: "持续学习、元学习、组合优化与自然气味阳性对照",
				titleEn:
					"Continual Learning, Meta-Learning, TSP & Natural Odor Controls",
				epigraphZh:
					"生物稀疏连接究竟在哪类生态任务上真正击败随机网络，又在哪类人工抽象任务上被过度神化？",
				epigraphEn:
					"Where does biological sparsity genuinely outperform random graphs, and where have claims been overstated?",
				chapters: [
					{
						num: "Ch.04",
						titleZh: "持续学习与抗灾难性遗忘审计：稀疏激活的真实边界",
						titleEn:
							"Continual Learning Audit & Catastrophic Forgetting Bounds",
						summaryZh:
							"检验肯尼恩细胞稀疏度对任务序列遗忘率的抑制作用及其容量上限。",
						summaryEn:
							"Auditing Kenyon cell sparsity against sequential task interference.",
						href: "/posts/2026-10-21-fly-connectome-04-continual-learning-audit/",
					},
					{
						num: "Ch.05",
						titleZh: "少样本元学习与跨环境奖励反转适应",
						titleEn: "Few-Shot Meta-Learning & Reward Reversal Adaptation",
						summaryZh: "测试气味-效价反转时的突触可塑性快速重映射效率。",
						summaryEn:
							"Rapid valence remapping under reward-reversal schedules.",
						href: "/posts/2026-10-21-fly-connectome-05-meta-learning/",
						routeBadge: "Route 03",
					},
					{
						num: "Ch.06",
						titleZh: "旅行商问题（TSP）三轮严苛复核：生物拓扑能解 NP-Hard 吗？",
						titleEn:
							"Three-Round TSP Audit: Can Connectomes Solve Combinatorial Search?",
						summaryZh:
							"剥离后处理启发式水分后，诚实检验连接组在组合优化上的真实贡献。",
						summaryEn:
							"Stripping away post-hoc heuristics to test whether connectome topology aids TSP search.",
						href: "/posts/2026-10-21-fly-connectome-06-tsp-three-rounds/",
						routeBadge: "Route 05",
					},
					{
						num: "Ch.07",
						titleZh: "阳性对照：自然生态气味统计流形上的结构匹配优势",
						titleEn: "Positive Controls on Natural Ecological Odor Manifolds",
						summaryZh:
							"证明演化雕琢的非均匀投射连接专门匹配自然界真实化学气味的协方差结构。",
						summaryEn:
							"Demonstrating evolutionary tuning to the covariance structure of natural chemical odors.",
						href: "/posts/2026-10-21-fly-connectome-07-positive-controls-natural-odours/",
					},
				],
			},
			{
				volNum: "第三卷 · VOLUME III",
				titleZh: "个体差异、多巴胺舱室动力学与阴性结果总报告",
				titleEn: "Individuality, Dopaminergic Compartments & Negative Results",
				epigraphZh:
					"科学的尊严不仅在于发现阳性显著性，更在于敢于发表干净、彻底、可复现的阴性证伪结论。",
				epigraphEn:
					"Scientific integrity rests on publishing clean, falsifiable, and reproducible negative results.",
				chapters: [
					{
						num: "Ch.08",
						titleZh: "跨半球与个体突触连接差异：随机发育噪声还是功能冗余？",
						titleEn: "Inter-Hemispheric & Individual Connectome Variability",
						summaryZh: "比较左右半球连接不对称性对行为输出稳定性的影响。",
						summaryEn:
							"Quantifying left-right hemispheric variability versus functional invariance.",
						href: "/posts/2026-10-21-fly-connectome-08-individual-differences/",
					},
					{
						num: "Ch.09",
						titleZh:
							"蘑菇体 15 个多巴胺舱室（Compartments）的异构遗忘与冲突仲裁",
						titleEn:
							"Heterogeneous Decay & Conflict Arbitration in 15 MB Compartments",
						summaryZh: "快慢记忆通道并行写入与厌恶/奖赏冲突仲裁的舱室动力学。",
						summaryEn:
							"Multi-timescale memory traces and valence arbitration across mushroom body compartments.",
						href: "/posts/2026-10-21-fly-connectome-09-compartments-and-dopamine/",
					},
					{
						num: "Ch.10",
						titleZh: "连接组计算神经科学 12 项方法论审校清单",
						titleEn:
							"Twelve-Point Methodological Review Checklist for Connectomics",
						summaryZh: "防范度分布混淆、读出层过拟合与基线不公的标准化自查表。",
						summaryEn:
							"Preventing readout overfitting, degree-distribution confounding, and unfair baselines.",
						href: "/posts/2026-10-21-fly-connectome-10-review-checklist/",
					},
					{
						num: "Ch.11",
						titleZh: "全书总结与阴性结果诚实汇总结论：解剖图谱到智能还差什么？",
						titleEn:
							"Summary of Negative Results: What Static Connectomes Cannot Explain",
						summaryZh:
							"为什么仅凭静态电子显微镜突触权重矩阵不足以复现完整生物智能？指向神经调质、具身动力学与发育可塑性的必然性。",
						summaryEn:
							"Why static EM synaptic matrices alone cannot replicate living intelligence without neuromodulation and embodiment.",
						href: "/posts/2026-10-21-fly-connectome-11-summary/",
					},
				],
			},
		],
		sidenotes: [
			{
				marker: "边注 §1",
				titleZh: "为什么阴性结果（Ch.11）是全书最重要的部分？",
				titleEn: "Why Negative Results Are the Crown Jewel",
				bodyZh:
					"当前 AI 与计算神经科学界充斥着‘把静态连接组矩阵塞进任意任务宣称超越基准’的幸存者偏差。本书通过严格控制度分布与读出层容量，明确划出了静态解剖连接能做与不能做的真实分界线。",
				bodyEn:
					"By enforcing degree-matched null models and equal readout capacity, this study demarcates what static anatomy truly contributes versus readout illusion.",
				linkHref: "/routes/neuroscience/",
				linkTextZh: "对照第 10 路线：神经科学与脑网络 ↗",
				linkTextEn: "Compare with Route 10: Neuroscience ↗",
			},
		],
	},
	{
		id: "frontier-foundation-models",
		roman: "MONOGRAPH IV",
		titleZh: "前沿基础模型技术研究室：从 DeepSeek 演进到系统协同",
		titleEn:
			"Frontier Foundation Model Research: From DeepSeek Evolution to Systems Co-Design",
		subtitleZh:
			"以模型演进为主线、核心研究问题为索引，深度还原 MLA、DeepSeekMoE、GRPO、R1、mHC、Engram、V4.1 CED 与底层基础设施的协同设计",
		subtitleEn:
			"Problem-Indexed Dissection of MLA, DeepSeekMoE, GRPO, R1, mHC, Engram, V4.1 CED, and AI Infrastructure Co-Design",
		editionBadgeZh: "常青研究专著 · v0.1 Skeleton Established",
		editionBadgeEn: "Living Research Monograph · v0.1 Skeleton",
		status: "living",
		completionPercent: 10,
		accentColor: "#4338ca",
		spineColor: "#1e1b4b",
		epigraphZh:
			"“真正理解一个前沿模型体系，不能止步于知道它用了 MLA、MoE 或 GRPO，而必须回答三个更深的问题：当时遇到了什么瓶颈而放弃了其他路线？架构、训练算法与底层硬件如何协同？脱离千卡集群后哪些规律依然成立？”",
		epigraphEn:
			"“To truly understand a frontier model family, we must ask: Why this design over rejected alternatives? How do architecture, RL, and kernels co-evolve? And which insights generalize beyond cluster scale?”",
		abstractZh:
			"本专著是「前沿基础模型技术研究室」的核心载体。第一阶段以 DeepSeek 从 V1、DeepSeekMoE、V2、V3、R1 直至 V3.2、mHC、Engram、V4.1-Flash 与 DualPath 的完整技术演进为主线，围绕六大研究视角与三维知识库索引（问题索引、技术索引、论文与四级证据索引）展开；每章严格交付机制推导、设计决策还原、四级证据审查与 MacBook Air M4 可复现实验，并为后续横向对标 OpenAI、Anthropic、DeepMind、Qwen 建立统一标尺。",
		abstractEn:
			"Organized around six research perspectives and a three-index knowledge base (Problems, Techniques, and Four-Tier Evidence Audits), this monograph reconstructs the full technical evolution of DeepSeek from V1–V3 and R1 to mHC, Engram, V4.1-Flash, and DualPath.",
		featuredLensId: "connectionist",
		volumes: [
			{
				volNum: "第一卷 · VOLUME I",
				titleZh: "高效基础模型基座：全景、MoE、MLA 与 V3 全链路（2023–2024）",
				titleEn:
					"Efficient Foundation Models: Panorama, DeepSeekMoE, MLA & V3 Pipeline",
				epigraphZh:
					"从标准 Dense Transformer 的显存与通信墙出发，追踪细粒度稀疏激活、低秩潜向量 KV 压缩与 FP8 无辅助损失训练如何合流。",
				epigraphEn:
					"Tracing how fine-grained sparsity, low-rank KV compression, and aux-loss-free FP8 training broke the memory and communication walls.",
				chapters: [
					{
						num: "Unit 01",
						titleZh:
							"DeepSeek 技术演进全景：从标准 Transformer 到 V4.1，究竟改变了什么？",
						titleEn:
							"DeepSeek Technical Evolution Panorama: From Vanilla Transformer to V4.1",
						summaryZh:
							"梳理 V1 → V2 → V3 → R1 → V3.2 → V4 → V4.1 的架构与训练路线，还原每一代解决的扩展瓶颈、引入的新问题与关键设计决策表。",
						summaryEn:
							"Mapping the full trajectory from V1 to V4.1: bottlenecks solved, new trade-offs introduced, and a master design decision ledger.",
						href: "/posts/2026-10-20-llm-training-pretrain-03-forward/",
						routeBadge: "P-00 · Q01-Q06",
						hasInteractive: true,
						status: "upcoming",
					},
					{
						num: "Unit 02",
						titleZh:
							"深入 DeepSeekMoE：细粒度专家分割、共享专家隔离与无辅助损失负载均衡",
						titleEn:
							"Inside DeepSeekMoE: Fine-Grained Experts, Shared Isolation & Aux-Loss-Free Balancing",
						summaryZh:
							"从 Dense FFN 与传统 GShard/Switch MoE 出发，推导专家分工机制、V3 动态偏置负载均衡，并开展小规模路由实验。",
						summaryEn:
							"From Dense FFN to fine-grained expert segmentation, shared expert isolation, and V3 dynamic bias load balancing.",
						href: "/posts/2026-10-20-llm-training-pretrain-03-forward/",
						routeBadge: "P-01 · Q-02",
						hasInteractive: true,
						status: "upcoming",
					},
					{
						num: "Unit 03",
						titleZh:
							"深入 MLA（多头潜在注意力）：为什么可以只保存低秩压缩表示？",
						titleEn:
							"Inside Multi-Head Latent Attention (MLA): Low-Rank KV Compression & Decoupled RoPE",
						summaryZh:
							"从 MHA、MQA、GQA 进入 MLA，详解低秩上下投影、推理期矩阵吸收、解耦 RoPE 位置编码及其与 KV Cache 带宽的定量关系。",
						summaryEn:
							"Tensor-level derivation of low-rank KV joint compression, matrix absorption during inference, and decoupled RoPE.",
						href: "/lab/rope-attention/",
						routeBadge: "P-03 · Q-01",
						hasInteractive: true,
						status: "upcoming",
					},
					{
						num: "Unit 04",
						titleZh:
							"还原 DeepSeek-V3 完整训练链路：MTP 多 Token 预测、FP8 混合精度与系统权衡",
						titleEn:
							"Reconstructing DeepSeek-V3: Multi-Token Prediction, Fine-Grained FP8 & Co-Design",
						summaryZh:
							"把 MLA、无辅助损失 MoE、MTP、细粒度 Tile/Block FP8 量化与 DualPipe 组合起来，交付全链路架构图与成本审计报告。",
						summaryEn:
							"Synthesizing MLA, aux-loss-free MoE, MTP, tile/block-wise FP8 quantization, and DualPipe into an end-to-end systems audit.",
						href: "/lab/model-pipeline/",
						routeBadge: "P-04 · Q-03/Q-05",
						hasInteractive: true,
						status: "upcoming",
					},
				],
			},
			{
				volNum: "第二卷 · VOLUME II",
				titleZh:
					"复杂推理与强化学习演进：GRPO、R1、过程验证与 Agent RL（2024–2025）",
				titleEn:
					"Complex Reasoning & RL: GRPO, R1-Zero/R1, Process Verification & Agent RL",
				epigraphZh:
					"当强化学习甩掉昂贵的同规模 Critic 网络，长思维链反思、定理自验证与多步工具交互是如何在规则奖励下涌现的？",
				epigraphEn:
					"How long-CoT self-reflection, theorem verification, and multi-turn tool agency emerge under critic-free reinforcement learning.",
				chapters: [
					{
						num: "Unit 05",
						titleZh:
							"DeepSeekMath 与 GRPO 起源：组内相对优势如何消去 Critic 显存负担？",
						titleEn:
							"DeepSeekMath & GRPO: Eliminating the Value Critic via Group-Relative Advantage",
						summaryZh:
							"对比 PPO 与 GRPO 的方差、显存与长度偏置，复现小模型在可验证数学任务上的组内采样消融实验。",
						summaryEn:
							"Comparing PPO and GRPO across variance, memory footprint, and length bias with small-model ablations.",
						href: "/posts/2026-10-23-llm-training-rl-05-grpo/",
						routeBadge: "P-02 · Q-04",
						hasInteractive: true,
						status: "upcoming",
					},
					{
						num: "Unit 06",
						titleZh:
							"DeepSeek-R1 与 R1-Zero：纯 RL 推理涌现、冷启动 SFT 必要性与蒸馏边界",
						titleEn:
							"DeepSeek-R1 & R1-Zero: Pure-RL Emergence, Cold-Start SFT & Distillation Boundaries",
						summaryZh:
							"对比 R1-Zero 与 R1 两条训练路径：可读性崩塌、语言混杂、多阶段奖励设计，以及蒸馏究竟是迁移模式还是激发上限。",
						summaryEn:
							"Dissecting R1-Zero vs. R1: readability collapse, multi-stage rule rewards, and what distillation transfers vs. limits.",
						href: "/posts/2026-10-23-llm-training-rl-06-evaluation/",
						routeBadge: "P-05 · Q-04",
						status: "upcoming",
					},
					{
						num: "Unit 07",
						titleZh:
							"DeepSeekMath-V2：如何从最终答案奖励转向可自验证的推理过程？",
						titleEn:
							"DeepSeekMath-V2: From Outcome Rewards to Self-Verifiable Reasoning Processes",
						summaryZh:
							"当问题缺乏简单标量核对器或存在伪证正确率时，如何联合训练生成器与证明验证器闭环。",
						summaryEn:
							"Moving beyond outcome-only rewards to generator-verifier loops for rigorous mathematical proof checking.",
						href: "/posts/2026-10-23-llm-training-rl-02-verifiers/",
						routeBadge: "P-07 · Q-04",
						status: "upcoming",
					},
					{
						num: "Unit 08",
						titleZh:
							"DeepSeek-V3.2：动态稀疏注意力（DSA）与大规模 Agent 环境强化学习",
						titleEn:
							"DeepSeek-V3.2: Dynamic Sparse Attention & Large-Scale Agentic RL",
						summaryZh:
							"剖析闪电索引器（Lightning Indexer）如何实现细粒度稀疏注意力，并与合成多步工具调用环境协同扩展。",
						summaryEn:
							"Combining Lightning Indexer fine-grained sparse attention with synthetic multi-turn tool-use reinforcement learning.",
						href: "/posts/2026-10-23-llm-training-tool-01-trajectories/",
						routeBadge: "P-06 · Q-01/Q-06",
						status: "upcoming",
					},
				],
			},
			{
				volNum: "第三卷 · VOLUME III",
				titleZh:
					"架构重塑与不对称计算：mHC、Engram、V4 与 V4.1-Flash（2025–2026）",
				titleEn:
					"Architectural Redesign: mHC, Engram, Million-Token V4 & Asymmetric V4.1-Flash",
				epigraphZh:
					"跳出标准残差流、同质 FFN 参数存储与输入输出对称计算的隐含假设，重新审视基础模型的连接、记忆与编码边界。",
				epigraphEn:
					"Challenging standard residual connections, monolithic FFN memory, and symmetric input-output computation.",
				chapters: [
					{
						num: "Unit 09",
						titleZh:
							"mHC（流形约束超连接）：能否拓宽残差连接容量又保持深层训练稳定？",
						titleEn:
							"mHC (Manifold-Constrained Hyper-Connections): Expanding Residual Streams Safely",
						summaryZh:
							"分析无约束超连接（HC）为何破坏恒等映射导致信号爆炸，以及 Sinkhorn-Knopp 双随机矩阵流形投影如何恢复稳定性。",
						summaryEn:
							"How projecting multi-stream hyper-connection matrices onto the Birkhoff polytope restores identity-mapping stability.",
						href: "/posts/2026-10-20-llm-training-pretrain-05-optimization/",
						routeBadge: "P-08 · Q-03",
						status: "upcoming",
					},
					{
						num: "Unit 10",
						titleZh:
							"Engram 条件记忆：静态世界知识查找（O(1)）与动态神经计算能否分开？",
						titleEn:
							"Engram Conditional Memory: Decoupling O(1) Knowledge Lookup from Neural Computation",
						summaryZh:
							"剖析基于现代哈希 N-gram 嵌入的 Engram 模块如何与 MoE 形成 U 型缩放互补，并将大容量词表卸载至主机内存。",
						summaryEn:
							"Separating static knowledge lookup via hashed N-gram tables from dynamic MoE expert computation.",
						href: "/posts/2026-10-20-llm-training-pretrain-01-tokenization/",
						routeBadge: "P-09 · Q-02",
						status: "upcoming",
					},
					{
						num: "Unit 11",
						titleZh:
							"DeepSeek-V4 与 V4.1-Flash：百万上下文、CED 不对称编解码与跨层 KV 复用",
						titleEn:
							"DeepSeek-V4 & V4.1-Flash: Million-Token Context, CED & Cross-Layer KV Reuse",
						summaryZh:
							"研究为什么输入阶段与输出阶段应当采用不同计算结构（Causal Encoder–Decoder、CSA2 与跨层 KV 共享），并严格执行四级证据边界审查。",
						summaryEn:
							"Examining asymmetric prompt encoding vs. decoding (CED), CSA2, cross-layer KV reuse, and strict evidence boundary auditing.",
						href: "/posts/2024-10-27-llm推理优化与部署/",
						routeBadge: "P-10/P-11 · Q-01/Q-05",
						status: "upcoming",
					},
				],
			},
			{
				volNum: "第四卷 · VOLUME IV",
				titleZh: "从算法到完整 AI 基础设施：内核、通信、DualPipe 与 DualPath",
				titleEn:
					"From Algorithms to AI Infrastructure: Kernels, DeepEP, DualPipe & DualPath",
				epigraphZh:
					"没有贴合硬件内存层次与互连拓扑的算子与调度，纸面上的 FLOPs 削减永远无法变成真实的训练吞吐与线上低延迟。",
				epigraphEn:
					"Without hardware-aware kernels, communication primitives, and storage scheduling, theoretical FLOP reductions stay on paper.",
				chapters: [
					{
						num: "Unit 12",
						titleZh:
							"算子与并行协同：FlashMLA、DeepGEMM、DeepEP 与 DualPipe 双向流水线",
						titleEn:
							"Kernel & Parallelism Co-Design: FlashMLA, DeepGEMM, DeepEP & DualPipe",
						summaryZh:
							"拆解 Hopper/现代 GPU 上的变长 KV 核函数、细粒度 FP8 矩阵乘、NVLink/RDMA 专家并行调度与计算通信重叠数学条件。",
						summaryEn:
							"Inside variable-length MLA kernels, fine-grained FP8 GEMM, MoE all-to-all dispatch, and DualPipe computation-communication overlap.",
						href: "/posts/2026-10-23-llm-training-eval-03-costs/",
						routeBadge: "Sys-01 · Q-02/Q-03",
						status: "upcoming",
					},
					{
						num: "Unit 13",
						titleZh:
							"DualPath 与 Agent 推理存储系统：如何突破多轮长上下文 KV Cache 带宽瓶颈？",
						titleEn:
							"DualPath & Agentic Inference: Breaking the KV Cache Storage Bandwidth Bottleneck",
						summaryZh:
							"分析高缓存命中率下 Prefill 网卡饱和与 Decode 网卡闲置的不对称矛盾，还原存储→Prefill 与存储→Decode 双路径动态调度。",
						summaryEn:
							"Resolving asymmetric NIC utilization in multi-turn agentic workloads via dual-path KV cache loading across Prefill and Decode engines.",
						href: "/posts/2024-10-27-llm推理优化与部署/",
						routeBadge: "P-12 · Q-01/Q-06",
						status: "upcoming",
					},
				],
			},
		],
		sidenotes: [
			{
				marker: "边注 §1",
				titleZh: "研究工程五步法与四级证据审查",
				titleEn: "Five-Step Methodology & Four-Tier Evidence Audit",
				bodyZh:
					"本专著拒绝把“看懂论文”等同于“验证论文”。每章固定区分：[E1 论文报告结论]、[E2 开源源码确认机制]、[E3 第三方独立复现] 与 [E4 本机 M4 16GB 缩模实验 / 待验假设]。",
				bodyEn:
					"Every chapter strictly separates author claims [E1], open-source code verification [E2], independent reproductions [E3], and local M4 16GB probe experiments [E4].",
				linkHref: "/lab/model-pipeline/",
				linkTextZh: "打开全链路模型交互实验台 ↗",
				linkTextEn: "Open Model Pipeline Studio ↗",
			},
			{
				marker: "边注 §2",
				titleZh: "三维研究索引（问题 · 技术 · 论文）",
				titleEn: "Three-Index Knowledge Base",
				bodyZh:
					"全书以 Q-01 至 Q-06 六大核心研究问题为一级锚点（如 KV 带宽墙、稀疏激活、无 Critic 推理 RL、残差与记忆分离），为后续横向对比 OpenAI、Anthropic、DeepMind、Qwen 提供统一坐标系。",
				bodyEn:
					"Indexed by six core research questions (Q-01 to Q-06) to enable rigorous cross-lab comparisons across DeepSeek, OpenAI, Anthropic, DeepMind, and Qwen.",
				linkHref: "/library/",
				linkTextZh: "查阅站内研究文献架 ↗",
				linkTextEn: "Browse Research Library ↗",
			},
		],
	},
	{
		id: "code-as-world-model",
		roman: "MONOGRAPH V",
		titleZh: "代码造物与显式世界模型：程序化动画、物理仿真与数学可视化手记",
		titleEn:
			"Code as World Model: Procedural Animation, Physics Simulation & Mathematical Visualization",
		subtitleZh:
			"从微分方程辛积分器、相空间吸引子、几何连杆逆运动学到偏微分方程连续场的 12 章可交互造物专著",
		subtitleEn:
			"A 12-Chapter Interactive Monograph on Symplectic Integrators, Phase-Space Attractors, Linkage Kinematics, and Emergent Continuous Fields",
		editionBadgeZh: "十二单元全集完结 · v1.0 Complete Edition",
		editionBadgeEn: "12 Units Complete · v1.0 Edition",
		status: "completed",
		completionPercent: 100,
		accentColor: "#0f766e",
		spineColor: "#115e59",
		epigraphZh:
			"“当你拒绝使用预设的 CSS 关键帧，而是亲手写下一行状态递推方程 s_{t+1} = f(s_t, a_t, Δt) 时，你写的就不再是‘动画特效’，而是在数字虚空中立下了一条物理定律——屏幕上的每一帧运动，都是这个微型世界模型在时间流中的必然涌现。”",
		epigraphEn:
			"“When you reject pre-baked keyframes and write an explicit state transition law s_{t+1} = f(s_t, a_t, Δt), you are no longer coding visual effects—you are legislating physical laws for a miniature universe.”",
		abstractZh:
			"为什么一段 60fps 的程序化动画本质上就是一个完备的「显式世界模型」？本专著遵循 Bret Victor、Bartosz Ciechanowski 与 3Blue1Brown 的第一性原理交互标准，将代码动画拆解为「内部状态流形 s_t」、「演化动力学算子 f」与「观测渲染投影 g(s_t)」。全书分为三卷十二章，从时间步进与辛几何数值积分出发，穿越刚体连杆、位置动力学与逆运动学，最终抵达反应扩散偏微分方程与 AI 隐空间世界模型的交汇处。",
		abstractEn:
			"Why is a 60fps procedural animation fundamentally an explicit World Model? Following the interactive standards of Bret Victor and Bartosz Ciechanowski, this three-volume monograph dissects code animation into state manifolds, dynamical operators, and rendering projections—bridging numerical physics and AI world models.",
		featuredLensId: "world-model",
		volumes: [
			{
				volNum: "第一卷 · VOLUME I",
				titleZh: "时间步进、微分方程与数值积分器",
				titleEn:
					"Time Stepping, Differential Equations & Numerical Integrators",
				epigraphZh:
					"如何让静态几何在离散帧率中诚实地演化，而不因数值截断产生伪能量爆炸或相位漂移？",
				epigraphEn:
					"How static geometry evolves faithfully across discrete frames without spurious energy explosion or phase drift.",
				chapters: [
					{
						num: "Unit 01",
						titleZh:
							"世界模型的最小骨架：状态向量 s_t、固定时间步 Δt 与渲染投影 g(s_t)",
						titleEn:
							"Minimal Skeleton of a World Model: State Vector, Fixed Timestep & Projection",
						summaryZh:
							"解剖 requestAnimationFrame 循环、固定物理步长累加器（Accumulator）与帧间线性插值——为什么掉帧绝对不能改变世界的物理结果？",
						summaryEn:
							"Dissecting requestAnimationFrame loops, fixed-timestep accumulators, and render interpolation so frame drops never alter physics.",
						href: "/posts/2026-11-01-code-world-model-01-state-and-accumulator/",
						routeBadge: "Unit 01 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 02",
						titleZh:
							"积分器的艺术：显式欧拉、半隐式辛欧拉（Symplectic）、Verlet 与 RK4",
						titleEn:
							"The Art of Integrators: Explicit Euler, Symplectic Euler, Verlet & RK4",
						summaryZh:
							"为什么仅调换一行速度与位置的更新顺序，就能决定一个代码动画世界是‘相空间面积守恒’还是‘能量无中生有原地爆炸’？",
						summaryEn:
							"Why swapping a single line of velocity-position update order determines whether phase-space volume is conserved or explodes.",
						href: "/posts/2026-11-01-code-world-model-02-symplectic-integrators/",
						routeBadge: "Unit 02 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 03",
						titleZh:
							"二阶动力学与有机运动感：用阻尼谐振子 (ω_n, ζ) 取代死板的贝塞尔缓动",
						titleEn:
							"Second-Order Dynamics: Replacing Bezier Easings with Damped Harmonic Oscillators",
						summaryZh:
							"推导欠阻尼、临界阻尼与过阻尼二阶微分系统，让几何图形与 UI 响应鼠标干预时自带真实的重量、惯性与动量连续性。",
						summaryEn:
							"Deriving underdamped, critically damped, and overdamped second-order systems for momentum-preserving interactive motion.",
						href: "/posts/2026-11-01-code-world-model-03-second-order-dynamics/",
						routeBadge: "Unit 03 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 04",
						titleZh:
							"相空间（Phase Space）、极限环与混沌分岔：从范德波尔振荡器到双摆",
						titleEn:
							"Phase Space, Limit Cycles & Chaotic Bifurcations: Van der Pol to Double Pendulum",
						summaryZh:
							"同屏绘制‘物理空间动画’与‘(θ, dθ/dt) 相空间矢量场’，直观看见极限环吸引子如何编码生命节律、初始微扰如何指数发散。",
						summaryEn:
							"Synchronizing physical-space animation with phase-portrait vector fields to visualize limit-cycle attractors and Lyapunov divergence.",
						href: "/posts/2026-11-01-code-world-model-04-phase-space-and-attractors/",
						routeBadge: "Unit 04 · Published",
						hasInteractive: true,
						status: "published",
					},
				],
			},
			{
				volNum: "第二卷 · VOLUME II",
				titleZh: "几何约束、刚体连杆与逆运动学",
				titleEn: "Geometric Constraints, Rigid Linkages & Inverse Kinematics",
				epigraphZh:
					"当多个自由度被刚性距离与角度约束锁死时，单点旋转如何转化为优雅的仿生机械步态？",
				epigraphEn:
					"How local crank rotation transforms into lifelike locomotion when degrees of freedom are locked by geometric constraints.",
				chapters: [
					{
						num: "Unit 05",
						titleZh:
							"闭环连杆与解析几何约束：从圆交点闭式解到 Strandbeest 仿生风力兽步态",
						titleEn:
							"Closed-Loop Linkages: Circle-Intersection Kinematics & Strandbeest Gait",
						summaryZh:
							"基于 Theo Jansen 十一圣数连杆比例，手推四杆/八杆机构余弦定理闭式解、死点奇异判据与平底支撑步态包络线。",
						summaryEn:
							"Closed-form circle-circle intersections and cosine-law kinematics for Theo Jansen's Strandbeest linkage.",
						href: "/posts/2026-11-01-code-world-model-05-closed-loop-linkages/",
						routeBadge: "Unit 05 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 06",
						titleZh:
							"位置动力学（Position-Based Dynamics, PBD）与雅可比迭代约束松弛",
						titleEn:
							"Position-Based Dynamics (PBD) & Gauss-Seidel Constraint Relaxation",
						summaryZh:
							"跳过繁琐的全局刚度矩阵求逆，仅凭‘Verlet 惯性预测 + 迭代拉回距离约束’的十行代码，实时模拟悬链绳索、布料网格与软体多边形。",
						summaryEn:
							"Simulating ropes, cloth meshes, and soft bodies in real time via Verlet prediction and iterative distance-constraint projection.",
						href: "/posts/2026-11-01-code-world-model-06-position-based-dynamics/",
						routeBadge: "Unit 06 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 07",
						titleZh:
							"正向与逆向运动学（FK / IK）：FABRIK 算法与程序化多足步态合成",
						titleEn:
							"Forward & Inverse Kinematics: FABRIK & Procedural Multi-Legged Locomotion",
						summaryZh:
							"零预制美术关键帧：用前向后向迭代逆运动学（FABRIK）配合重心偏移触发器，写出自动适应地形起伏的程序化蜘蛛与触手动画。",
						summaryEn:
							"Zero pre-baked keyframes: synthesizing terrain-adaptive multi-legged gaits using Forward-And-Backward Reaching Inverse Kinematics.",
						href: "/posts/2026-11-01-code-world-model-07-inverse-kinematics-fabrik/",
						routeBadge: "Unit 07 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 08",
						titleZh: "有向距离场（SDF）、连续碰撞检测与刚体冲量动量守恒",
						titleEn:
							"Signed Distance Fields (SDF), Collision Manifolds & Impulse Conservation",
						summaryZh:
							"用纯数学标量场 d(p) = 0 定义任意几何边界、光滑布尔融合与表面法向量 ∇d，推导弹性恢复系数与旋转冲量响应。",
						summaryEn:
							"Defining exact geometry, smooth Boolean blends, and surface normals via Signed Distance Fields and impulse resolution.",
						href: "/posts/2026-11-01-code-world-model-08-signed-distance-fields-collision/",
						routeBadge: "Unit 08 · Published",
						hasInteractive: true,
						status: "published",
					},
				],
			},
			{
				volNum: "第三卷 · VOLUME III",
				titleZh: "连续场、局部规则涌现与心智仿真连接",
				titleEn: "Continuous Fields, Local Emergence & Mental Simulation",
				epigraphZh:
					"极简的局部代码算子如何涌现出复杂的生命图景？显式代码世界又如何成为检验 AI 隐式世界模型的黄金标尺？",
				epigraphEn:
					"How local operators give rise to macro-scale life—and how explicit code worlds benchmark neural world models.",
				chapters: [
					{
						num: "Unit 09",
						titleZh: "局部感知与群体涌现：Boids 三定律、空间网格哈希与流场寻路",
						titleEn:
							"Local Rules & Flock Emergence: Boids, Spatial Hashing & Flow Fields",
						summaryZh:
							"分离、对齐、凝聚三个局部向量算子，如何在零全局指挥官的前提下涌现出椋鸟群风暴与果蝇觅食聚集。",
						summaryEn:
							"How separation, alignment, and cohesion vectors produce emergent flocking and foraging swarms with O(N) spatial hashing.",
						href: "/posts/2026-11-01-code-world-model-09-boids-flocking-spatial-hashing/",
						routeBadge: "Unit 09 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 10",
						titleZh:
							"网格上的偏微分方程：2D 波动方程、图灵反应扩散（Gray-Scott）与流体平流",
						titleEn:
							"PDEs on a Grid: Wave Equation, Gray-Scott Reaction-Diffusion & Advection",
						summaryZh:
							"把拉普拉斯算子 ∇² 写成二维数组的五点离散卷积，在 Canvas 像素缓冲中实时长出干涉涟漪、珊瑚斑纹与涡旋场。",
						summaryEn:
							"Discretizing the Laplacian operator ∇² on 2D grids to grow interference ripples, Turing morphogenesis patterns, and fluid vortices.",
						href: "/posts/2026-11-01-code-world-model-10-grid-pde-reaction-diffusion/",
						routeBadge: "Unit 10 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 11",
						titleZh:
							"Manim 与 Ciechanowski 式的数学叙事语法：同伦变换、复平面映射与视觉证明",
						titleEn:
							"Mathematical Visual Storytelling: Homotopy, Complex Mappings & Visual Proofs",
						summaryZh:
							"提炼用代码讲清抽象数学的工程范式：基底线性变换动画、流形同伦插值、状态着色编码与30秒顿悟结构。",
						summaryEn:
							"Engineering visual proofs: basis transformations, homotopy interpolation, and state-driven color encoding.",
						href: "/posts/2026-11-01-code-world-model-11-mathematical-visual-storytelling/",
						routeBadge: "Unit 11 · Published",
						hasInteractive: true,
						status: "published",
					},
					{
						num: "Unit 12",
						titleZh:
							"终章回响：显式代码世界模型 vs. 神经网络隐空间做梦（MBRL / Dreamer / JEPA）",
						titleEn:
							"Epilogue: Explicit Code World Models vs. Neural Latent Rollouts",
						summaryZh:
							"将前三卷的手写确定性物理引擎作为 Ground Truth，对照 Model-Based RL、Dreamer 隐空间轨迹推演与视频生成模型：为什么真正的智能需要如代码般守恒且可干预的内部世界模型？",
						summaryEn:
							"Benchmarking MBRL, Dreamer latent rollouts, and video generators against explicit code physics: why causal invariants matter.",
						href: "/posts/2026-11-01-code-world-model-12-neural-vs-code-world-models/",
						routeBadge: "Unit 12 · Published",
						hasInteractive: true,
						status: "published",
					},
				],
			},
		],
		sidenotes: [
			{
				marker: "边注 §1",
				titleZh: "Ciechanowski / Victor 第一性原理铁律",
				titleEn: "The First-Principles Interactive Standard",
				bodyZh:
					"本专著遵循全站交互解释器铁律：严禁使用无数学内核的纯装饰性 CSS 动画。每一个视觉组件都必须由显式物理递推方程、几何约束或微分算子实时驱动，并提供可跨越分岔阈值的控制旋钮。",
				bodyEn:
					"No decorative CSS animations: every visual widget is driven live by explicit differential equations, geometric constraints, or phase-space operators.",
				linkHref: "/essays/strandbeest/",
				linkTextZh: "体验 Strandbeest 连杆视觉随笔 ↗",
				linkTextEn: "Explore Strandbeest Visual Essay ↗",
			},
			{
				marker: "边注 §2",
				titleZh: "连接第 07 与第 12 智能路线",
				titleEn: "Bridging Route 07 (Causality) & Route 12 (Dynamics)",
				bodyZh:
					"用代码写动画是人类正向构造『显式世界模型』（从法则到画面）；而强化学习与认知科学中的世界模型则是从观测经验中逆向拟合『隐式状态转移』（从画面到法则）。两者在相空间与反事实干预 `do(x)` 上殊途同归。",
				bodyEn:
					"Coding simulations constructs explicit world models forward (laws → pixels), while AI world models invert the process (pixels → latent laws).",
				linkHref: "/routes/causal-world-models/",
				linkTextZh: "对照第 07 路线：因果与世界模型 ↗",
				linkTextEn: "Compare Route 07: Causal & World Models ↗",
			},
		],
	},
];

export function getBookById(id: string): LivingBook | undefined {
	return LIVING_BOOKS.find((b) => b.id === id);
}
