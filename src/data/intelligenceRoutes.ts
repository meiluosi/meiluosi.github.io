/**
 * 人类探索智能的 20 个路线家族及其分歧地图
 * An Atlas of Intelligence: 20 Paradigm Families & Divergence Map
 */

export interface RouteBranch {
	nameZh: string;
	nameEn: string;
	descZh: string;
	descEn: string;
}

export interface RouteItem {
	id: string;
	index: number;
	roman: string;
	nameZh: string;
	nameEn: string;
	cluster:
		| "measurement"
		| "representation"
		| "embodiment"
		| "collective"
		| "mind";
	clusterZh: string;
	clusterEn: string;
	color?: string;
	superOrbit: "embodied" | "representation" | "causality" | "recursion";
	superOrbitZh: string;
	questionLevel:
		| "criteria"
		| "measurement"
		| "mechanism"
		| "origin"
		| "boundary"
		| "value";
	questionLevelZh: string;
	coreThesisZh: string;
	coreThesisEn: string;
	valueZh: string;
	valueEn: string;
	limitationsZh: string;
	limitationsEn: string;
	branches: RouteBranch[];
	linkedLabs?: Array<{ nameZh: string; nameEn: string; href: string }>;
	samplePosts?: Array<{ nameZh: string; nameEn: string; slug: string }>;
}

export interface QuestionLevel {
	id: string;
	nameZh: string;
	nameEn: string;
	inquiryZh: string;
	inquiryEn: string;
}

export interface DivergenceAxis {
	nameZh: string;
	nameEn: string;
	poleAZh: string;
	poleAEn: string;
	poleBZh: string;
	poleBEn: string;
	notesZh: string;
	notesEn: string;
}

export interface RouteCluster {
	id: string;
	nameZh: string;
	nameEn: string;
	range: string;
	descZh: string;
	descEn: string;
	color: string;
}

export interface RouteStance {
	axisIndex: number;
	axisNameZh: string;
	axisNameEn: string;
	poleAZh: string;
	poleAEn: string;
	poleBZh: string;
	poleBEn: string;
	/** position: -1 (强烈偏极 A), 0 (兼顾/处于两极张力中), +1 (强烈偏极 B) */
	position: number;
	rationaleZh: string;
	rationaleEn: string;
}

/** 6 大问题层次 */
export const QUESTION_LEVELS: QuestionLevel[] = [
	{
		id: "criteria",
		nameZh: "判定与定义",
		nameEn: "Criteria & Definition",
		inquiryZh: "什么样的表现，可以被称为智能？",
		inquiryEn: "What kinds of performance qualify as intelligent?",
	},
	{
		id: "measurement",
		nameZh: "测量与结构",
		nameEn: "Measurement & Structure",
		inquiryZh: "智能是一个总能力，还是许多不同能力的组合？",
		inquiryEn:
			"Is intelligence a unitary capacity or a manifold of distinct abilities?",
	},
	{
		id: "mechanism",
		nameZh: "机制与实现",
		nameEn: "Mechanism & Realization",
		inquiryZh: "推理、学习、理解、决策是怎样在系统内部发生的？",
		inquiryEn:
			"How do reasoning, learning, and decision-making actually take place?",
	},
	{
		id: "origin",
		nameZh: "发展与起源",
		nameEn: "Development & Origin",
		inquiryZh: "智能如何在个体成长、生命演化和社会历史中逐步形成？",
		inquiryEn:
			"How does intelligence emerge across ontogeny, phylogeny, and cultural history?",
	},
	{
		id: "boundary",
		nameZh: "边界与归属",
		nameEn: "Boundary & Locus",
		inquiryZh: "智能究竟位于大脑内部、具身形态、环境耦合，还是整个群体之中？",
		inquiryEn:
			"Does intelligence reside in the brain, the body, the environment, or the collective?",
	},
	{
		id: "value",
		nameZh: "意义与价值",
		nameEn: "Meaning & Value",
		inquiryZh: "会解决问题是否就意味着理解？智能是否必然包含意识、目的或智慧？",
		inquiryEn:
			"Does task proficiency entail understanding? Must intelligence involve consciousness or wisdom?",
	},
];

/** 8 大主要分歧轴 */
export const DIVERGENCE_AXES: DivergenceAxis[] = [
	{
		nameZh: "评价什么？",
		nameEn: "What to Evaluate?",
		poleAZh: "已经拥有的任务能力（静态储备）",
		poleAEn: "Current task capabilities",
		poleBZh: "获得新能力的效率与泛化潜力（动态过程）",
		poleBEn: "Efficiency and potential in acquiring new skills",
		notesZh:
			"Chollet ARC 等测试主张将先验与经验剥离，测量面对陌生任务的学习效率，而非海量预训练事实储备。",
		notesEn:
			"Contrasts static benchmark performance against dynamic sample-efficiency and generalization on novel tasks.",
	},
	{
		nameZh: "怎样解释行为？",
		nameEn: "How to Explain Behavior?",
		poleAZh: "内部表征、符号推演与世界模型计算",
		poleAEn: "Internal representations, reasoning, and world modeling",
		poleBZh: "主体与环境之间的实时动力学生理耦合",
		poleBEn: "Real-time perceptual-action coupling with environment",
		notesZh:
			"经典认知科学主张‘先内部算好再执行’；行为式机器人与生态心理学主张‘世界本身就是它最好的模型’。",
		notesEn:
			"Representational cognitive architecture versus Brooksian subsumption and ecological direct perception.",
	},
	{
		nameZh: "什么使行动有方向？",
		nameEn: "What Orients Action?",
		poleAZh: "外部给定的标量奖励或损失函数",
		poleAEn: "Externally specified rewards or loss functions",
		poleBZh: "机体自维持、内在动机与自主形成的意义",
		poleBEn: "Autopoietic self-maintenance and intrinsic motivation",
		notesZh:
			"Reinforcement learning relies on external reward; autopoiesis and active inference view orientation as self-preservation.",
		notesEn:
			"Is agency driven by extrinsic optimization or thermodynamic self-maintenance?",
	},
	{
		nameZh: "智能的单位是什么？",
		nameEn: "Locus of Intelligence?",
		poleAZh: "孤立的个体、算法模型或大脑",
		poleAEn: "Isolated individual, neural model, or brain",
		poleBZh: "由身体材料、外部工具、社会群体构成的延展系统",
		poleBEn: "Extended system composed of body, artifacts, and culture",
		notesZh:
			"延展心智与分布式认知认为，笔记本、语言工具和制度本身就构成认知过程的一部分。",
		notesEn:
			"Clark & Chalmers' extended mind vs solitary computational entities.",
	},
	{
		nameZh: "需要多少先验？",
		nameEn: "How Much Innate Prior?",
		poleAZh: "从空白开始的通用纯统计学习（极少预设）",
		poleAEn: "Tabula rasa scaling with generic statistical priors",
		poleBZh: "丰富的核心知识结构、领域几何与演化约束",
		poleBEn: "Rich core knowledge, inductive bias, and evolutionary priors",
		notesZh:
			"Scaling 假说主张通用架构+算力即可；Spelke、Lake 等强调人类学习依赖物理对象、代理意图等核心先验。",
		notesEn: "Pure scaling empiricism versus developmental core cognition.",
	},
	{
		nameZh: "是否存在统一机制？",
		nameEn: "Is There a Master Algorithm?",
		poleAZh: "一个足够一般的原则（如自由能、自监督预测、奖励优化）",
		poleAEn: "A single universal principle (Free Energy, Prediction, Reward)",
		poleBZh: "多模块系统、多重异构机制与发育阶段的协调协作",
		poleBEn: "Heterogeneous cognitive architecture and modular society of mind",
		notesZh:
			"‘Reward is enough’ 或自由能最小化尝试统一解释；ACT-R、Soar 与 Minsky 心智社会强调异构组件协作。",
		notesEn:
			"Universal learning objective vs Minsky's society of mind and cognitive architectures.",
	},
	{
		nameZh: "意识是否必要？",
		nameEn: "Is Consciousness Required?",
		poleAZh: "功能与能力完全可以独立于主观体验研究",
		poleAEn:
			"Functional competence can be studied without phenomenal experience",
		poleBZh: "真正的理解与主体性需要现象体验与自我感受",
		poleBEn:
			"Genuine understanding requires subjectivity and phenomenal feeling",
		notesZh:
			"图灵判定与弱 AI 认为主观感受可抽离；中文屋论证与生物自然主义认为形式操作不等于主观理解。",
		notesEn:
			"Turing operationalism versus Searle's Chinese Room and phenomenal consciousness.",
	},
	{
		nameZh: "聪明是否包含价值？",
		nameEn: "Does Intelligence Entail Wisdom?",
		poleAZh: "高效且工具性地达成任意指定目标",
		poleAEn: "Instrumental rationality: achieving any given goal efficiently",
		poleBZh: "审视目标是否正当、反思后果与何谓好的生活",
		poleBEn: "Practical wisdom (Phronesis): evaluating goals and living well",
		notesZh:
			"现代 IQ 与智能定义往往是价值中立的工具理性；亚里士多德实践智慧与跨文化心性传统强调德性辨别。",
		notesEn:
			"Instrumental optimization versus normative practical wisdom and ethics.",
	},
];

/** 20 个路线家族定义 */
export const INTELLIGENCE_ROUTES: RouteItem[] = [
	{
		id: "behavioral",
		index: 1,
		roman: "01",
		nameZh: "行为表现与操作性路线",
		nameEn: "Behavioral & Operational Route",
		cluster: "measurement",
		clusterZh: "表现、测量与效率",
		clusterEn: "Measurement & Efficiency",
		superOrbit: "causality",
		superOrbitZh: "因果与决策",
		questionLevel: "criteria",
		questionLevelZh: "判定与定义",
		coreThesisZh:
			"将判断放到可观察的交互行为上：先看它能做什么，避免诉诸主观感觉。",
		coreThesisEn:
			"Anchors intelligence in observable behavioral competencies and interaction benchmarks rather than unobservable internal states.",
		valueZh: "提供客观可度量的操作定义，避免陷入唯心式的感觉之争。",
		valueEn:
			"Provides objective, falsifiable criteria for performance assessment.",
		limitationsZh:
			"行为相近可能由不同机制产生；擅长回答‘表现如何’，无法直接回答‘内部机制是什么’。",
		limitationsEn:
			"Identical behavior can arise from vastly different internal mechanisms.",
		branches: [
			{
				nameZh: "行为主义",
				nameEn: "Behaviorism",
				descZh: "从环境条件、学习历史与强化解释能力，避开不可观察的内在心灵。",
				descEn:
					"Explains capacities via stimulus-response, reinforcement histories, and environmental conditions.",
			},
			{
				nameZh: "图灵式行为判定",
				nameEn: "Turing Test",
				descZh: "将‘机器能否思考’转化为语言交互中是否能与人类不可区分。",
				descEn:
					"Operationalizes machine thought through indistinguishable conversational interaction.",
			},
			{
				nameZh: "任务能力路线",
				nameEn: "Task Competence",
				descZh: "按解决复杂问题的正确性、稳定性与覆盖广度综合评价能力。",
				descEn:
					"Evaluates systems across accuracy, stability, and difficulty benchmarks.",
			},
			{
				nameZh: "能力维度模型",
				nameEn: "Dimensional Capability",
				descZh: "将能力水平、适用广度与自主性严格拆解，避免单一‘AGI’标签包办。",
				descEn:
					"Decomposes proficiency, generality, and autonomy into orthogonal axes.",
			},
		],
	},
	{
		id: "psychometric",
		index: 2,
		roman: "02",
		nameZh: "心理测量与能力结构路线",
		nameEn: "Psychometrics & Structural Route",
		cluster: "measurement",
		clusterZh: "表现、测量与效率",
		clusterEn: "Measurement & Efficiency",
		superOrbit: "causality",
		superOrbitZh: "因果与决策",
		questionLevel: "measurement",
		questionLevelZh: "测量与结构",
		coreThesisZh:
			"研究人类认知任务间的共变差异：智能究竟是一种总能力，还是多种能力的层级结构？",
		coreThesisEn:
			"Investigates individual differences in cognitive tasks: whether intelligence reflects a unitary g factor or a modular hierarchy.",
		valueZh: "严格的统计测量方法，揭示了不同认知能力之间的相关性与层级组织。",
		valueEn:
			"Rigorous statistical modeling uncovering cognitive correlations and hierarchy.",
		limitationsZh:
			"观察到统计上的共同因素 (g)，并不等于找到了单一的生物物理或算法机制。",
		limitationsEn:
			"A statistical correlation factor does not entail a single biological or algorithmic engine.",
		branches: [
			{
				nameZh: "一般智力因素 (g)",
				nameEn: "General Intelligence (g)",
				descZh: "不同认知任务成绩存在普遍正相关，可以用一般能力因素总结。",
				descEn:
					"Positive manifold among diverse tasks summarized as a single common factor.",
			},
			{
				nameZh: "CHC 层级模型",
				nameEn: "CHC Model",
				descZh:
					"区分流体智力（处理新问题）与晶体智力（积累知识），组织为层级结构。",
				descEn:
					"Distinguishes fluid reasoning from crystallized knowledge within a stratified hierarchy.",
			},
			{
				nameZh: "过程重叠理论",
				nameEn: "Process Overlap Theory",
				descZh:
					"任务调用部分重叠的基础认知过程，从而在统计上浮现出共同能力因素。",
				descEn:
					"Shared executive processes across distinct tasks generate the statistical g factor.",
			},
			{
				nameZh: "多元智能与成功智力",
				nameEn: "Multiple Intelligences",
				descZh: "强调语言、数理、空间、音乐、人际等多重异构维度的相对独立性。",
				descEn:
					"Emphasizes the modular autonomy of spatial, linguistic, interpersonal, and practical skills.",
			},
		],
	},
	{
		id: "learning-efficiency",
		index: 3,
		roman: "03",
		nameZh: "学习效率与泛化路线",
		nameEn: "Sample Efficiency & Generalization",
		cluster: "measurement",
		clusterZh: "表现、测量与效率",
		clusterEn: "Measurement & Efficiency",
		superOrbit: "recursion",
		superOrbitZh: "递归与自省",
		questionLevel: "measurement",
		questionLevelZh: "测量与结构",
		coreThesisZh:
			"智能不在于‘已经记住多少题’，而在于‘面对未知时怎样高效习得新技能’。",
		coreThesisEn:
			"Intelligence is not a database of acquired skills, but the efficiency of acquiring new skills over unknown tasks.",
		valueZh:
			"刺破了预训练数据污染和死记硬背的假象，将焦点转向真正的先验与样本效率。",
		valueEn:
			"Distinguishes memorized prior training data from genuine adaptive capability.",
		limitationsZh:
			"能够学习不等于拥有全部智能；运行时不改变参数也不等于绝对无智能。",
		limitationsEn:
			"Learning ability is not an all-encompassing proxy for every dimension of cognition.",
		branches: [
			{
				nameZh: "技能获取效率 (Chollet ARC)",
				nameEn: "Skill Acquisition Efficiency",
				descZh:
					"将先验、经验与任务难度纳入评估，测量面对全新未知情境的泛化速度。",
				descEn:
					"Measures adaptive speed against task novelty under constrained prior experience.",
			},
			{
				nameZh: "结构性与组合性泛化",
				nameEn: "Compositional Generalization",
				descZh: "将已理解的构件模块化地组合到陌生语境中，实现概念跃迁。",
				descEn:
					"Recombining known primitives systematically to solve structurally novel challenges.",
			},
			{
				nameZh: "计算学习理论 (Valiant PAC)",
				nameEn: "Computational Learning Theory",
				descZh:
					"以概率近似正确 (PAC) 框架形式化什么能被有限样本和算力有效学习。",
				descEn:
					"Formalizes learnability under sample, computational, and statistical bounds.",
			},
		],
	},
	{
		id: "rational-resource",
		index: 4,
		roman: "04",
		nameZh: "理性、适应与资源约束路线",
		nameEn: "Rationality & Resource Constraints",
		cluster: "measurement",
		clusterZh: "表现、测量与效率",
		clusterEn: "Measurement & Efficiency",
		superOrbit: "causality",
		superOrbitZh: "因果与决策",
		questionLevel: "mechanism",
		questionLevelZh: "机制与实现",
		coreThesisZh:
			"在时间、算力与知识都不充分的前提下，做出协调而适宜的选择：适可而止也是聪明。",
		coreThesisEn:
			"Achieving optimal or adaptive decisions under strict limitations of time, computation, and knowledge.",
		valueZh: "打破‘算得越多越聪明’的虚妄，将智能与计算代价、现实约束紧密锚定。",
		valueEn: "Balances accuracy against computational resource expenditures.",
		limitationsZh:
			"不同学派在数学形式上互不兼容（如 AIXI 的不可计算性与有限启发的经验性）。",
		limitationsEn:
			"Theoretical extremes (AIXI) diverge sharply from empirical heuristics.",
		branches: [
			{
				nameZh: "通用智能形式化 (Legg-Hutter / AIXI)",
				nameEn: "Universal Intelligence",
				descZh: "在所有可计算环境中最大化加权期望奖励的理论极值。",
				descEn:
					"Mathematical formalization of expected performance over all computable environments.",
			},
			{
				nameZh: "有限理性与资源理性 (Simon / Griffiths)",
				nameEn: "Bounded & Resource Rationality",
				descZh:
					"将搜索与推理本身的算力代价计入效用，寻找计算资源的合理分配策略。",
				descEn:
					"Optimizing the trade-off between decision quality and internal thinking cost.",
			},
			{
				nameZh: "生态理性与快速启发式 (Gigerenzer)",
				nameEn: "Ecological Rationality",
				descZh:
					"简单启发式与特定环境结构的共振，往往优于复杂而过度拟合的计算。",
				descEn:
					"Fast-and-frugal heuristics outperforming complex models by exploiting environmental regularities.",
			},
			{
				nameZh: "知识与资源不足下的适应 (NARS / 王培)",
				nameEn: "Adaptation under Insufficient Knowledge",
				descZh: "在假设知识不完备、资源时刻受限的真实条件下持续自主运作。",
				descEn:
					"Defining intelligence as real-time adaptive survival with incomplete knowledge.",
			},
		],
		linkedLabs: [
			{
				nameZh: "Kuhn 扑克博弈与 CFR",
				nameEn: "Kuhn Poker CFR",
				href: "/lab/cfr-poker/",
			},
		],
	},
	{
		id: "symbolic-logic",
		index: 5,
		roman: "05",
		nameZh: "符号、逻辑、搜索与组合路线",
		nameEn: "Symbolic, Logic & Search",
		cluster: "representation",
		clusterZh: "表征、推断与控制",
		clusterEn: "Representation & Control",
		superOrbit: "representation",
		superOrbitZh: "语言与表征",
		questionLevel: "mechanism",
		questionLevelZh: "机制与实现",
		coreThesisZh:
			"智能源于结构化的符号实体及其操作演化：具有产生一般智能行为的充要条件假说。",
		coreThesisEn:
			"Intelligence stems from structured symbol tokens, logical derivations, and heuristic search through state spaces.",
		valueZh: "具备极高的可解释性、可验证性、因果演绎严谨性与组合生成能力。",
		valueEn:
			"Transparent, verifiable, compositional deduction and systematic reasoning.",
		limitationsZh:
			"符号接地困难；在面对模糊、连续、高噪声的感知物理现实时脆弱易碎。",
		limitationsEn:
			"Brittle in noisy perceptual domains; suffers from the classical symbol grounding problem.",
		branches: [
			{
				nameZh: "物理符号系统假说 (Newell & Simon)",
				nameEn: "Physical Symbol System",
				descZh: "主张符号系统的模式组合具有产生通用智能行为的充分必要条件。",
				descEn:
					"Hypothesis that symbol processing constitutes the necessary and sufficient substrate for general intelligence.",
			},
			{
				nameZh: "启发式搜索与规划",
				nameEn: "Heuristic Search & Planning",
				descZh: "将世界建模为状态空间与目标转移，通过估值函数剪枝搜索路径。",
				descEn:
					"Representing problem solving as heuristic tree search and state transitions.",
			},
			{
				nameZh: "结构映射与类比推理 (Gentner)",
				nameEn: "Structure Mapping & Analogy",
				descZh: "发现两个异构系统深层次的关系同构，实现跨领域知识迁移。",
				descEn:
					"Aligning relational systems across domains to project insights analogically.",
			},
			{
				nameZh: "程序归纳与神经符号系统",
				nameEn: "Program Induction & Neuro-Symbolic",
				descZh: "将概念理解为可组合的代码或谓词程序，从少量样本推断生成规则。",
				descEn:
					"Representing concepts as executable probabilistic programs, marrying connectionist priors with discrete syntax.",
			},
		],
		linkedLabs: [
			{
				nameZh: "蒙特卡洛树搜索 (MCTS)",
				nameEn: "MCTS Playground",
				href: "/lab/mcts/",
			},
		],
	},
	{
		id: "connectionist",
		index: 6,
		roman: "06",
		nameZh: "连接主义、统计学习与表征路线",
		nameEn: "Connectionism & Deep Representation",
		cluster: "representation",
		clusterZh: "表征、推断与控制",
		clusterEn: "Representation & Control",
		superOrbit: "representation",
		superOrbitZh: "语言与表征",
		questionLevel: "mechanism",
		questionLevelZh: "机制与实现",
		coreThesisZh:
			"知识分散在海量连接权重的活动模式中：智能来自连续几何空间的深层表征与自监督压缩。",
		coreThesisEn:
			"Knowledge is distributed across continuous parameter spaces and trained end-to-end via statistical representation learning.",
		valueZh:
			"在多模态感知、自然语言直觉、高维特征提取上展现了前所未有的经验威力与 Scaling 规律。",
		valueEn:
			"Unmatched scaling efficiency across high-dimensional perceptual and linguistic domains.",
		limitationsZh:
			"易产生幻觉与对抗脆弱性；‘随规模扩大而改善’并不等价于‘这就是智能的唯一本质’。",
		limitationsEn:
			"Scaling empirical correlation does not logically imply that statistical parameter fitting encompasses all intelligence.",
		branches: [
			{
				nameZh: "分布式表征与连接主义",
				nameEn: "Distributed Representations",
				descZh:
					"概念不是孤立标签，而是由高维隐空间中大量神经元的联合激活来表征。",
				descEn:
					"Concepts encoded as distributed activation vectors across interconnected weights.",
			},
			{
				nameZh: "深层表征与流形学习",
				nameEn: "Deep Manifold Learning",
				descZh:
					"多层非线性变换逐级解耦数据流形，形成适于分类、生成与对齐的高层语义空间。",
				descEn:
					"Successive non-linear layers unfolding entangled high-dimensional data manifolds.",
			},
			{
				nameZh: "预测与自监督学习",
				nameEn: "Next-Token / Self-Supervised",
				descZh:
					"以输入自身为监督信号（如下一个词预测），从无标注海量语料中汲取世界知识压缩。",
				descEn:
					"Learning dense world models by predicting masked or subsequent sensory inputs.",
			},
			{
				nameZh: "Scaling Law 经验路线",
				nameEn: "Scaling Laws",
				descZh: "算力、数据量与参数规模呈现出平滑幂律关系，驱动涌现现象。",
				descEn:
					"Smooth power-law predictability governing test loss across computation, data, and model scale.",
			},
		],
		linkedLabs: [
			{
				nameZh: "RoPE 旋转位置编码星盘",
				nameEn: "RoPE Astrolabe",
				href: "/lab/rope-attention/",
			},
			{
				nameZh: "注意力归纳回路 Lens",
				nameEn: "Attention Lens",
				href: "/lab/attention-lens/",
			},
		],
	},
	{
		id: "causal-world-models",
		index: 7,
		roman: "07",
		nameZh: "概率推断、因果模型与世界模型路线",
		nameEn: "Bayesian, Causal & World Models",
		cluster: "representation",
		clusterZh: "表征、推断与控制",
		clusterEn: "Representation & Control",
		superOrbit: "causality",
		superOrbitZh: "因果与决策",
		questionLevel: "mechanism",
		questionLevelZh: "机制与实现",
		coreThesisZh:
			"智能是对不可见世界的推测：从观察相关走向干预（do）与反事实推演（What if?）。",
		coreThesisEn:
			"Intelligence is counterfactual inference: moving beyond passive statistical correlation into interventions and internal world simulations.",
		valueZh:
			"清晰界定了统计观察、主动干预与反事实推演的三级因果阶梯，为模型赋予因果解释力。",
		valueEn:
			"Formalizes Pearl's causal ladder: association, intervention, and counterfactual simulation.",
		limitationsZh:
			"因果图的发现极为困难；拥有内部仿真器不等于掌握所有不确定性。",
		limitationsEn:
			"Causal graph identification remains computationally intractable without strong structural assumptions.",
		branches: [
			{
				nameZh: "贝叶斯认知科学 (Tenenbaum)",
				nameEn: "Bayesian Brain",
				descZh:
					"在先验信念与局部观测证据间动态贝叶斯更新，从小样本形成抽象归纳。",
				descEn:
					"Hierarchical Bayesian inference updating prior hypotheses from sparse observations.",
			},
			{
				nameZh: "因果阶梯与结构因果模型 (Pearl)",
				nameEn: "Structural Causal Models",
				descZh: "严格区分关联、行动干预与反事实推测，建立无混杂因果图谱。",
				descEn:
					"Distinguishing seeing, doing (interventions), and imagining counterfactual outcomes.",
			},
			{
				nameZh: "预测编码与自由能推断",
				nameEn: "Predictive Processing",
				descZh: "大脑是一个自顶向下发送预测、自底向上反馈残差误差的预测机器。",
				descEn:
					"Hierarchical message passing resolving sensory prediction errors top-down.",
			},
			{
				nameZh: "世界模型与隐空间模拟 (Ha / LeCun)",
				nameEn: "Latent World Models",
				descZh:
					"在内部构建紧凑的环境动力学生成模型，在梦境与想象中进行试错规划。",
				descEn:
					"Simulating physical and transition dynamics inside a learned compact latent environment.",
			},
		],
		linkedLabs: [
			{
				nameZh: "因果效应估计 Playground",
				nameEn: "Causal Lab",
				href: "/lab/causal-playground/",
			},
		],
	},
	{
		id: "cybernetics-rl",
		index: 8,
		roman: "08",
		nameZh: "控制论、强化学习与主动推断路线",
		nameEn: "Cybernetics, RL & Active Inference",
		cluster: "representation",
		clusterZh: "表征、推断与控制",
		clusterEn: "Representation & Control",
		superOrbit: "causality",
		superOrbitZh: "因果与决策",
		questionLevel: "mechanism",
		questionLevelZh: "机制与实现",
		coreThesisZh:
			"智能不是孤立的问答，而是‘感知 $\\rightarrow$ 决策 $\\rightarrow$ 行动 $\\rightarrow$ 环境变化 $\\rightarrow$ 新感知’的闭环动态。",
		coreThesisEn:
			"Intelligence is closed-loop agentic regulation: coupling perception and policy actions with long-term environment dynamics.",
		valueZh: "将智能落实在物理或环境闭环交互中，强调长期收益优化与稳态维持。",
		valueEn:
			"Grounds intelligence in dynamic interaction, feedback homeostasis, and sequential optimization.",
		limitationsZh:
			"标量奖励是否充分（Reward is Enough 假说）仍存巨大争议；样本效率与稀疏奖励探索代价昂贵。",
		limitationsEn:
			"Reward engineering can be vulnerable to specification gaming and severe credit assignment bottlenecks.",
		branches: [
			{
				nameZh: "好调节器定理与控制论 (Wiener / Ashby)",
				nameEn: "Cybernetic Regulation",
				descZh:
					"任何能够良好调节环境系统的控制器，其内部必定构成该系统的一个模型。",
				descEn:
					"Every good regulator of a system must be a model of that system.",
			},
			{
				nameZh: "强化学习与策略优化 (Sutton & Barto)",
				nameEn: "Reinforcement Learning",
				descZh:
					"通过行动的长期累积回报信号调整策略，平衡探索 (Exploration) 与利用 (Exploitation)。",
				descEn:
					"Optimizing sequential policy actions against cumulative discounted scalar reward signals.",
			},
			{
				nameZh: "主动推断与自由能框架 (Friston)",
				nameEn: "Active Inference",
				descZh:
					"不仅改变认知以符合感知，更主动采取行动去使外部环境符合预期稳态。",
				descEn:
					"Minimizing variational free energy through both belief updates and epistemic action.",
			},
			{
				nameZh: "赋能度与行动可控性 (Empowerment)",
				nameEn: "Empowerment",
				descZh:
					"不依赖外部特定奖励，而是主动维持和最大化自身对未来可能状态的因果控制力。",
				descEn:
					"Maximizing the mutual information between an agent's future actions and sensory states.",
			},
		],
		linkedLabs: [
			{
				nameZh: "DPO 偏好对齐工作台",
				nameEn: "Preference Alignment",
				href: "/lab/preference-alignment/",
			},
		],
	},
	{
		id: "cognitive-architectures",
		index: 9,
		roman: "09",
		nameZh: "认知架构与多机制协作路线",
		nameEn: "Cognitive Architectures & Meta-cognition",
		cluster: "embodiment",
		clusterZh: "系统、组织与具身",
		clusterEn: "Systems & Embodiment",
		superOrbit: "recursion",
		superOrbitZh: "递归与自省",
		questionLevel: "mechanism",
		questionLevelZh: "机制与实现",
		coreThesisZh:
			"不存在一个包办全部认知的万能算法：智能源于异构认知组件与元认知的协同调度。",
		coreThesisEn:
			"Rejects the myth of a single silver-bullet learning rule; general intelligence emerges from heterogeneous modular orchestration.",
		valueZh:
			"兼顾记忆、注意、感知、推演与反思，为复杂长程任务提供系统级架构支撑。",
		valueEn:
			"Integrates working memory, long-term declarative store, and meta-control into a unified system.",
		limitationsZh:
			"模块拼接容易带来工程复杂性，难以像端到端网络那样享受纯算力 Scaling 的红利。",
		limitationsEn:
			"Handcrafted architectural integration can be difficult to scale smoothly via unified gradients.",
		branches: [
			{
				nameZh: "统一认知架构 (ACT-R / Soar)",
				nameEn: "Unified Cognitive Architecture",
				descZh: "组织工作记忆、陈述性记忆、程序性生产规则与感知行动总线。",
				descEn:
					"Standard model of mind coordinating declarative chunks and procedural production rules.",
			},
			{
				nameZh: "心智社会 (Minsky)",
				nameEn: "Society of Mind",
				descZh:
					"心智由大量本身没有智能的简单 Agent 构成，智慧从它们的竞合互动中涌现。",
				descEn:
					"Intelligence emerging from complex networks of individually mindless interacting agencies.",
			},
			{
				nameZh: "元认知与自我监控",
				nameEn: "Meta-cognition",
				descZh:
					"不仅解决问题，更评估自身置信度、错误风险并合理分配‘慢思考’算力。",
				descEn:
					"Monitoring internal cognitive states, calibrating confidence, and gating System-2 deliberation.",
			},
			{
				nameZh: "创造性认知与概念空间 (Boden)",
				nameEn: "Creative Cognition",
				descZh:
					"区分组合式、探索式与突破式创造，智能体现为重新定义问题空间的能力。",
				descEn:
					"Exploring and transforming conceptual spaces rather than merely searching within fixed limits.",
			},
		],
	},
	{
		id: "neuroscience",
		index: 10,
		roman: "10",
		nameZh: "神经科学路线",
		nameEn: "Neuroscientific & Biological Brain",
		cluster: "embodiment",
		clusterZh: "系统、组织与具身",
		clusterEn: "Systems & Embodiment",
		superOrbit: "embodied",
		superOrbitZh: "具身与物理",
		questionLevel: "mechanism",
		questionLevelZh: "机制与实现",
		coreThesisZh:
			"解释真实生物神经网络如何实现认知：揭示脑区协同、网络拓扑与神经调质动力学。",
		coreThesisEn:
			"Grounds intelligence in the physical wiring, temporal oscillations, and functional topology of biological brains.",
		valueZh:
			"为人工智能提供天然的原型证据、能量约束启发（20W 功耗奇迹）与脉冲神经启示。",
		valueEn:
			"Inspires biologically realistic models, sparse firing, and ultra-efficient computing principles.",
		limitationsZh:
			"相关机制不等于充分机制；机器智能未必需要事无巨细地复制生物脑的演化冗余。",
		limitationsEn:
			"Neural correlates are not identical to sufficient algorithms; silicon intelligence may follow distinct physics.",
		branches: [
			{
				nameZh: "顶额叶整合理论 (P-FIT)",
				nameEn: "Parieto-Frontal Integration",
				descZh:
					"强调额叶、顶叶及其白质纤维束的高级信息整合与一般智力的相关性。",
				descEn:
					"Frontal-parietal information exchange network supporting general human cognitive capacity.",
			},
			{
				nameZh: "网络神经科学",
				nameEn: "Network Neuroscience",
				descZh:
					"以复杂图论分析全脑功能模块的分离 (Segregation) 与灵活重配置 (Integration)。",
				descEn:
					"Graph-theoretic analysis of modular segregation and dynamic whole-brain integration.",
			},
			{
				nameZh: "共享执行控制资源",
				nameEn: "Executive Control Networks",
				descZh:
					"从工作记忆负荷、多巴胺调控与注意力分配解释不同认知任务间的共变性。",
				descEn:
					"Shared frontoparietal resources modulating working memory and flexible task-switching.",
			},
		],
	},
	{
		id: "developmental-openended",
		index: 11,
		roman: "11",
		nameZh: "发展、建构与开放式学习路线",
		nameEn: "Developmental & Open-Ended Learning",
		cluster: "embodiment",
		clusterZh: "系统、组织与具身",
		clusterEn: "Systems & Embodiment",
		superOrbit: "recursion",
		superOrbitZh: "递归与自省",
		questionLevel: "origin",
		questionLevelZh: "发展与起源",
		coreThesisZh:
			"智能不是装进去的固定软件，而是在与环境互动的开放进程中持续‘长出来’的。",
		coreThesisEn:
			"Intelligence is not a static bundle injected into a system, but an open-ended growth trajectory that constructs its own curriculum.",
		valueZh:
			"将系统从封闭静态任务集解放出来，探讨持续终身学习、自发课程生成与元学习。",
		valueEn:
			"Models lifelong adaptability, intrinsic curiosity, and co-evolutionary curriculum synthesis.",
		limitationsZh:
			"开放式系统极易发散、遗忘或陷入无序噪声（难以稳定收敛于人类期望目标）。",
		limitationsEn:
			"Prone to catastrophic forgetting or drifting towards arbitrary, unaligned complexity.",
		branches: [
			{
				nameZh: "发展建构主义 (Piaget)",
				nameEn: "Developmental Constructivism",
				descZh:
					"通过同化 (Assimilation) 与顺应 (Accommodation)，个体主动建构与迭代图式结构。",
				descEn:
					"Cognitive schema construction through iterative assimilation, accommodation, and equilibration.",
			},
			{
				nameZh: "核心知识与早期先验 (Spelke)",
				nameEn: "Core Knowledge",
				descZh:
					"婴儿早期便具备物体持久性、代理目标等演化预装核心结构，作为后续学习脚手架。",
				descEn:
					"Innate evolutionary core systems serving as building blocks for rapid subsequent learning.",
			},
			{
				nameZh: "内在动机与好奇心驱动 (Oudeyer)",
				nameEn: "Intrinsic Motivation",
				descZh:
					"追逐‘学习进度’与可压缩性提升，主动寻找既非过于简单亦非完全混乱的学习前沿。",
				descEn:
					"Active exploration driven by empirical learning progress rather than random noise.",
			},
			{
				nameZh: "开放式自演化与 RSI (Stanley)",
				nameEn: "Open-Endedness & RSI",
				descZh:
					"持续生成全新挑战与阶梯技能，环境与智能体共同演化，实现递归自我提升。",
				descEn:
					"Never-ending novelty and continual capability ratchet through agent-environment co-evolution.",
			},
		],
	},
	{
		id: "embodied-dynamical",
		index: 12,
		roman: "12",
		nameZh: "具身、生态、生成与动力系统路线",
		nameEn: "Embodied, Ecological & Dynamical",
		cluster: "embodiment",
		clusterZh: "系统、组织与具身",
		clusterEn: "Systems & Embodiment",
		superOrbit: "embodied",
		superOrbitZh: "具身与物理",
		questionLevel: "boundary",
		questionLevelZh: "边界与归属",
		coreThesisZh:
			"挑战‘脑内先算好、外部再执行’的默认前提：身体材料与动力学本身就是计算的一部分。",
		coreThesisEn:
			"Cognition is physically situated: morphological mechanics, passive dynamics, and real-time sensorimotor loops absorb computational burden.",
		valueZh:
			"揭示了无神经系统下形态涌现步态的奥秘，避免了无休止的内部符号膨胀。",
		valueEn:
			"Demonstrates how physical geometry and material properties solve tasks without software bloat.",
		limitationsZh:
			"在低级反应与步态上极具说服力，但很难直接平移解释高阶抽象数学或脱离语境的远景推演。",
		limitationsEn:
			"Excels at sensorimotor locomotion but struggles to naturally explain context-free abstract reasoning.",
		branches: [
			{
				nameZh: "形态计算 (Morphological Computation)",
				nameEn: "Morphological Computation",
				descZh:
					"骨骼力学、弹性组织与连杆几何分担了原本需要中央微处理器完成的大量协调计算。",
				descEn:
					"Mechanical linkages, friction, and passive elasticity offloading central controller calculations.",
			},
			{
				nameZh: "生态心理学与示能 (Gibson)",
				nameEn: "Ecological Psychology & Affordance",
				descZh:
					"直接感知环境对该特定身体而言‘可以怎样行动’，无需先建立内部抽象表征。",
				descEn:
					"Direct perception of action opportunities (affordances) without mediating symbolic representations.",
			},
			{
				nameZh: "行为式机器人 (Brooks / 包容式架构)",
				nameEn: "Behavior-Based Robotics",
				descZh:
					"‘世界本身就是它最好的模型’，由多层紧密感知-行动耦合直接应对扰动。",
				descEn:
					"Subsumption architecture proving complex adaptation without centralized world modeling.",
			},
			{
				nameZh: "行动生成论 (Enactivism)",
				nameEn: "Enactivism",
				descZh:
					"认知并非外部客观世界的内部镜像，而是自主主体在与环境交互中共同‘生成’出来的意义空间。",
				descEn:
					"Cognition is not passive representation but world-making through living interactions.",
			},
		],
		linkedLabs: [
			{
				nameZh: "Strandbeest 连杆动力学步态",
				nameEn: "Strandbeest Mechanics",
				href: "/world/",
			},
		],
	},
	{
		id: "autopoiesis-evolution",
		index: 13,
		roman: "13",
		nameZh: "生命、自维持与演化路线",
		nameEn: "Autopoiesis, Life & Evolution",
		cluster: "embodiment",
		clusterZh: "系统、组织与具身",
		clusterEn: "Systems & Embodiment",
		superOrbit: "embodied",
		superOrbitZh: "具身与物理",
		questionLevel: "origin",
		questionLevelZh: "发展与起源",
		coreThesisZh:
			"智能首先是生命自组织的一部分：目标的真正来源，不是程序员写下的分数，而是系统维持自身存在的必要条件。",
		coreThesisEn:
			"Intelligence is grounded in biological self-preservation: agency originates from the necessity of maintaining system autopoiesis.",
		valueZh:
			"回答了‘意义何来’的难题：任何反馈之所以具有好坏属性，是因为它关系到主体的热力学生存。",
		valueEn:
			"Supplies the intrinsic origin of normative value: things matter because the organism must survive.",
		limitationsZh:
			"目前的人工智能均在无生命硅基设备上运行，如何将自创生原则移植到人工系统仍无定论。",
		limitationsEn:
			"Synthesizing genuine biological self-production in non-living silicon architectures remains an open problem.",
		branches: [
			{
				nameZh: "自创生理论 (Maturana & Varela)",
				nameEn: "Autopoiesis",
				descZh:
					"生命系统通过内部过程持续生产和再生产构成自身的边界与组织，认知即生活的过程。",
				descEn:
					"Living systems actively regenerate their own organizational boundaries; living is cognitive action.",
			},
			{
				nameZh: "多尺度基底认知 (Michael Levin)",
				nameEn: "Basal & Multi-scale Cognition",
				descZh:
					"从细胞、组织形态发生到生物个体，不同生物尺度都具备非神经的解题与目标趋向调节。",
				descEn:
					"Target-seeking homeostatic agency operates across cellular, bioelectric, and anatomical scales.",
			},
			{
				nameZh: "生态演化适应",
				nameEn: "Evolutionary Adaptation",
				descZh:
					"智能形态完全取决于有机体在生态位中遭遇的物理竞争与自然选择压力。",
				descEn:
					"Cognitive capabilities shaped strictly by adaptive niches and selective fitness pressures.",
			},
		],
	},
	{
		id: "sociocultural-language",
		index: 14,
		roman: "14",
		nameZh: "社会文化与语言路线",
		nameEn: "Sociocultural, Culture & Language",
		cluster: "collective",
		clusterZh: "集体、媒介与物理",
		clusterEn: "Collective & Physics",
		superOrbit: "representation",
		superOrbitZh: "语言与表征",
		questionLevel: "boundary",
		questionLevelZh: "边界与归属",
		coreThesisZh:
			"人类高阶智能不是离岛上孤立长出的野兽，而是浸泡在多代人累积的语言、符号工具与社会文化之中。",
		coreThesisEn:
			"High-level intelligence is not a solitary solitary spark, but culturally scaffolding via linguistic artifacts and cumulative traditions.",
		valueZh:
			"指明了人类智能超越黑猩猩的关键在于‘累积式文化齿轮’与代际符号继承。",
		valueEn:
			"Explains how the ratchet effect of cultural transmission unlocks exponential cognitive capability.",
		limitationsZh:
			"过分强调社会语言，容易忽视前语言动物或无文化具身实体的高超感知适应能力。",
		limitationsEn:
			"Risk of anthropocentric overemphasis on language, neglecting deep pre-linguistic competencies.",
		branches: [
			{
				nameZh: "社会文化历史学派 (Vygotsky)",
				nameEn: "Cultural-Historical Theory",
				descZh:
					"高级心理功能最初产生于人际交往，借助语言中介内化 (Internalization) 为个体的私语思考。",
				descEn:
					"Inter-psychological social dialogue becomes internalized into intra-psychological higher cognition.",
			},
			{
				nameZh: "共享意向性与合作 (Tomasello)",
				nameEn: "Shared Intentionality",
				descZh:
					"人类认知跃迁的关键不仅是预测他人，更是形成‘我们在此共同做事’的共同注意与道德契约。",
				descEn:
					"Cooperative communication through joint attention and shared intentionality ('we-mode').",
			},
			{
				nameZh: "累积式文化演化 (Cultural Brain)",
				nameEn: "Cumulative Cultural Evolution",
				descZh:
					"个体不必每一次重新发明车轮，文化基因的代际传递与筛选构成了智能的第二演化引擎。",
				descEn:
					"The cultural ratchet allowing societies to retain and accumulate engineering innovations across millennia.",
			},
		],
	},
	{
		id: "collective-extended",
		index: 15,
		roman: "15",
		nameZh: "集体、分布式与延展心智路线",
		nameEn: "Collective, Distributed & Extended Mind",
		cluster: "collective",
		clusterZh: "集体、媒介与物理",
		clusterEn: "Collective & Physics",
		superOrbit: "embodied",
		superOrbitZh: "具身与物理",
		questionLevel: "boundary",
		questionLevelZh: "边界与归属",
		coreThesisZh:
			"智能的物理单位不必受限于颅骨或皮肤：工具、群体与网络都可以构成认知的有机回路。",
		coreThesisEn:
			"The boundary of mind is not demarcated by the skull: cognition is distributed across tools, environments, and networks.",
		valueZh:
			"将现代科学研究、市场价格体系、维基百科等视为大尺度的超个体智能现象。",
		valueEn:
			"Treats markets, multi-agent systems, and epistemic tooling as genuine cognitive systems.",
		limitationsZh:
			"容易陷入‘认知膨胀’（Coupling-Constitution Fallacy）：影响了思考不等于它本身拥有心智。",
		limitationsEn:
			"Risk of over-generalization: just because a tool aids thought does not make it a sentient mind.",
		branches: [
			{
				nameZh: "延展心智假说 (Clark & Chalmers)",
				nameEn: "Extended Mind",
				descZh:
					"如果外部笔记工具在功能上与大脑内部工作记忆无异，它就应当被视作认知的物理组成部分。",
				descEn:
					"Active externalism: reliably coupled external artifacts literally constitute cognitive circuits.",
			},
			{
				nameZh: "分布式认知 (Hutchins)",
				nameEn: "Distributed Cognition",
				descZh:
					"轮船驾驶舱或航天指挥室的计算与感知，分布在操作规程、仪表与团队协作之中。",
				descEn:
					"Cognitive computations spread across people, charts, instruments, and physical spaces.",
			},
			{
				nameZh: "群体智慧与去中心化协调 (Hayek)",
				nameEn: "Collective Intelligence",
				descZh:
					"任何单个中心脑都无法拥有完整信息，分散知识通过局部交互涌现出全局最优配置。",
				descEn:
					"Decentralized coordination systems solving resource-allocation tasks that no individual mind can comprehend.",
			},
		],
	},
	{
		id: "information-physics",
		index: 16,
		roman: "16",
		nameZh: "信息、压缩、复杂系统与物理路线",
		nameEn: "Information, Compression & Physics",
		cluster: "collective",
		clusterZh: "集体、媒介与物理",
		clusterEn: "Collective & Physics",
		superOrbit: "representation",
		superOrbitZh: "语言与表征",
		questionLevel: "mechanism",
		questionLevelZh: "机制与实现",
		coreThesisZh:
			"试图用最底层的统计物理、香农信息熵与柯尔莫哥洛夫复杂度统领智能：理解即压缩。",
		coreThesisEn:
			"Pursues foundational physical invariants: intelligence as thermodynamic efficiency, predictive information compression, and critical dynamics.",
		valueZh:
			"在极其深邃的数学物理层面上提供了统一描述（如算法概率、信息瓶颈理论）。",
		valueEn:
			"Offers mathematically elegant principles bridging statistical mechanics with representation learning.",
		limitationsZh:
			"一个物理量能够描述智能系统（如临界态），不等于这个量就已经完全定义了智能。",
		limitationsEn:
			"Correlation with a physical metric (e.g. self-organized criticality) does not constitute a full theory of meaning.",
		branches: [
			{
				nameZh: "理解即压缩 (Kolmogorov / 索罗门诺夫归纳)",
				nameEn: "Compression is Intelligence",
				descZh:
					"从有限历史数据中找到最短的生成程序（奥卡姆剃刀形式化），实现最强未来预测。",
				descEn:
					"Finding the shortest generative program consistent with evidence maximizes predictive power.",
			},
			{
				nameZh: "信息瓶颈理论 (Tishby)",
				nameEn: "Information Bottleneck",
				descZh:
					"学习并非保留所有输入细节，而是尽量丢弃无关冗余，压缩出对目标最具互信息的表征。",
				descEn:
					"Trading off minimal input retention against maximal mutual information with target labels.",
			},
			{
				nameZh: "因果熵与未来选择最大化",
				nameEn: "Causal Entropic Forces",
				descZh:
					"主体倾向于自发演化向保留最多未来自由度的宏观路径，表现为目标驱使的自适应。",
				descEn:
					"Maximizing the diversity of accessible future behavioral paths over chosen horizons.",
			},
			{
				nameZh: "自组织临界性 (Criticality)",
				nameEn: "Self-Organized Criticality",
				descZh:
					"神经网络在有序与混沌的边缘（相变临界点）具备最大的动态范围与信息整合能力。",
				descEn:
					"Operating near the edge of chaos optimizes information storage, transfer, and dynamic range.",
			},
		],
	},
	{
		id: "philosophy-of-mind",
		index: 17,
		roman: "17",
		nameZh: "语义、理解与心灵哲学路线",
		nameEn: "Semantics, Mind & Understanding",
		cluster: "mind",
		clusterZh: "语义、意识与价值",
		clusterEn: "Mind & Meaning",
		superOrbit: "recursion",
		superOrbitZh: "递归与自省",
		questionLevel: "value",
		questionLevelZh: "意义与价值",
		coreThesisZh:
			"不争论谁的跑分更高，而是严厉追问：纯粹的形式语法操作，究竟在何时才能‘意味着什么’？",
		coreThesisEn:
			"Asks the foundational foundational philosophical question: how can syntactic manipulation ever produce genuine semantic meaning?",
		valueZh:
			"揭示了符号接地困难、意向性（Intentionality）与概念内涵的不可化约性，防止将工程拟人化错觉当成真理。",
		valueEn:
			"Prevents naive anthropomorphism by rigorously separating functional syntax from semantic intentionality.",
		limitationsZh: "哲学思辨往往无法直接输出可运行的工程算法代码或架构指导。",
		limitationsEn:
			"Primarily diagnostic and critical; rarely provides constructive implementation algorithms.",
		branches: [
			{
				nameZh: "中文屋论证与生物自然主义 (Searle)",
				nameEn: "Chinese Room & Biological Naturalism",
				descZh:
					"单纯根据规则置换符号，即使外部表现不可分辨，内部也不具备真实的理解意向。",
				descEn:
					"Syntax does not suffice for semantics: formal program manipulation lacks intrinsic intentionality.",
			},
			{
				nameZh: "符号接地问题 (Harnad)",
				nameEn: "Symbol Grounding Problem",
				descZh:
					"字典里的符号相互解释只是无穷闭环循环，符号必须锚定到底层感官-运动因果互动中。",
				descEn:
					"How symbolic tokens become causally connected to bottom-up invariant perceptual projections.",
			},
			{
				nameZh: "意向立场 (Dennett)",
				nameEn: "Intentional Stance",
				descZh:
					"将一个复杂系统假定为拥有‘信念’与‘欲望’并依理性行动，是一种极度成功的预测策略。",
				descEn:
					"Predicting a complex entity's behavior by treating it as a rational agent with beliefs and desires.",
			},
			{
				nameZh: "功能主义与计算主义",
				nameEn: "Functionalism & Computationalism",
				descZh:
					"心理状态由系统内部各部分的因果组织关系界定，在不同物理载体上具备多重可实现性。",
				descEn:
					"Mental states are causally realized functional relations independent of specific physical substrates.",
			},
		],
	},
	{
		id: "consciousness-self",
		index: 18,
		roman: "18",
		nameZh: "意识、情感与自我路线",
		nameEn: "Consciousness, Affect & Self",
		cluster: "mind",
		clusterZh: "语义、意识与价值",
		clusterEn: "Mind & Meaning",
		superOrbit: "recursion",
		superOrbitZh: "递归与自省",
		questionLevel: "value",
		questionLevelZh: "意义与价值",
		coreThesisZh:
			"解决问题是否必须对自身状态‘有所感受’？将解决问题的智力与主观体验（Qualia）严格拆分。",
		coreThesisEn:
			"Investigates whether genuine understanding requires phenomenal awareness (what it feels like), separating raw IQ from consciousness.",
		valueZh:
			"直面心灵哲学‘困难问题’（Hard Problem），提醒人们解题机器与有痛感、有体验的主体之间存在深渊。",
		valueEn:
			"Prevents conflating high cognitive problem-solving with phenomenal subjective experience.",
		limitationsZh:
			"缺乏客观公认的意识测量仪；各意识理论（IIT 与 GWT）在机理上尚存重大争鸣。",
		limitationsEn:
			"Phenomenological subjectivity resists third-person empirical verification; competing theories lack consensus.",
		branches: [
			{
				nameZh: "全局工作空间理论 (Baars / Dehaene)",
				nameEn: "Global Workspace Theory (GWT)",
				descZh:
					"当局部模块的信息被广播至全局神经工作空间、被多个系统共同调用时，产生意识通达。",
				descEn:
					"Information broadcast globally across cognitive sub-modules entering conscious awareness.",
			},
			{
				nameZh: "整合信息理论 (Tononi IIT)",
				nameEn: "Integrated Information Theory (IIT)",
				descZh:
					"从系统内部不可约约的因果结构度量意识量值 (Phi)，强调内在因果机制而非外在表现。",
				descEn:
					"Postulates consciousness as intrinsic integrated causal power (Phi) of a system's state space.",
			},
			{
				nameZh: "注意图式理论 (Graziano)",
				nameEn: "Attention Schema Theory",
				descZh:
					"大脑构建了一个关于自身注意状态的简化内部模型，以此监控、引导注意并形成意识觉知。",
				descEn:
					"The brain constructs a descriptive schema of its own attentional mechanisms to control focus.",
			},
			{
				nameZh: "感受态与躯体标记 (Damasio)",
				nameEn: "Somatic Marker & Affect",
				descZh:
					"内感受、情绪与身体状态参与价值打分，智能不是冷冰冰的逻辑推演，而是生于对躯体感受的守护。",
				descEn:
					"Homeostatic bodily affects and visceral sensations anchoring reason and value prioritization.",
			},
		],
	},
	{
		id: "wisdom-values",
		index: 19,
		roman: "19",
		nameZh: "智慧、价值与跨文化路线",
		nameEn: "Wisdom, Ethics & Cultural Traditions",
		cluster: "mind",
		clusterZh: "语义、意识与价值",
		clusterEn: "Mind & Meaning",
		superOrbit: "recursion",
		superOrbitZh: "递归与自省",
		questionLevel: "value",
		questionLevelZh: "意义与价值",
		coreThesisZh:
			"会实现目标，是否就足够‘明智’？反思目标本身的正当性、行动的时机与何谓良好的生活。",
		coreThesisEn:
			"Reflects on normative boundaries: is pursuing arbitrary goals enough, or must intelligence evaluate what is worth pursuing?",
		valueZh:
			"突破西方现代 IQ 的单一工具理性局限，融入亚里士多德实践智慧、儒道佛心性反思与长期公共善。",
		valueEn:
			"Transcends instrumental rationality by integrating classical practical wisdom, virtue, and long-term societal balance.",
		limitationsZh:
			"难以形式化为单一可优化的损失函数，涉及不可公度的伦理与价值权衡。",
		limitationsEn:
			"Resists reduction to mathematical loss functions due to pluralistic normative trade-offs.",
		branches: [
			{
				nameZh: "实践智慧 (Aristotle Phronesis)",
				nameEn: "Practical Wisdom (Phronesis)",
				descZh:
					"不仅通晓普遍命题，更在纷繁复杂的具体处境中洞察恰到好处的分寸、克制与善。",
				descEn:
					"Deliberative wisdom guiding virtuous action within nuanced, particular human contexts.",
			},
			{
				nameZh: "心性智识与修养 (儒道传统)",
				nameEn: "Heart-Mind & Self-Cultivation",
				descZh:
					"认知与道德辨别、情境顺应融为一体；不执于僵化概念分别，顺应物理与天性。",
				descEn:
					"Holistic integration of moral discernment, emotional attunement, and spontaneous wu-wei response.",
			},
			{
				nameZh: "觉察与执念解构 (佛学认知传统)",
				nameEn: "Mindfulness & Insight (Prajna)",
				descZh:
					"深刻剖析概念投射、心识造作与虚妄‘自我’构建，以般若智慧洞见缘起空性。",
				descEn:
					"Rigorous psychological deconstruction of mental projections, habitual grasping, and the illusory self.",
			},
			{
				nameZh: "适应性智能与平衡理论 (Sternberg)",
				nameEn: "Balance Theory of Wisdom",
				descZh: "在自我利益、他人利益与更大公共福祉之间取得长远动态平衡。",
				descEn:
					"Balancing intrapersonal, interpersonal, and extrapersonal interests over the long term.",
			},
		],
	},
	{
		id: "metaphysics-quantum",
		index: 20,
		roman: "20",
		nameZh: "形而上学、非经典计算与量子路线",
		nameEn: "Metaphysics, Non-classical & Quantum",
		cluster: "mind",
		clusterZh: "语义、意识与价值",
		clusterEn: "Mind & Meaning",
		superOrbit: "embodied",
		superOrbitZh: "具身与物理",
		questionLevel: "boundary",
		questionLevelZh: "边界与归属",
		coreThesisZh:
			"探讨心智是否超越了经典的图灵可计算物理假说：从本体论、泛心论到非经典测量模型。",
		coreThesisEn:
			"Examines ontological hypotheses asking whether mind transcends classical Church-Turing physical computability.",
		valueZh:
			"拓宽了人类对物质与心智关系的本体论视野，并在认知决策中成功运用了量子概率公理。",
		valueEn:
			"Deepens foundational ontology and provides powerful non-commutative quantum probability models for human judgment.",
		limitationsZh:
			"本体论假说极具争议；‘用量子概率建模认知决策偏差’不等于‘大脑物理上就是量子计算机’。",
		limitationsEn:
			"High metaphysical speculation; modeling cognitive order effects via quantum math is not proof of biological quantum brains.",
		branches: [
			{
				nameZh: "泛心论与双面论 (Panpsychism)",
				nameEn: "Panpsychism",
				descZh:
					"体验性或心智潜能是物理世界最基底的本质属性之一，而非复杂组织后凭空掉出来的奇迹。",
				descEn:
					"Experiential consciousness is a fundamental ontological building block of the physical universe.",
			},
			{
				nameZh: "非可计算性与量子脑假说 (Penrose & Hameroff)",
				nameEn: "Orch-OR / Non-computable Physics",
				descZh:
					"依据哥德尔不完备定理认为人类数学洞见不可算法化，寄托于微管量子坍缩。",
				descEn:
					"Hypothesizing non-algorithmic mathematical insight mediated by coordinated quantum state reduction.",
			},
			{
				nameZh: "量子认知决策模型 (Busemeyer)",
				nameEn: "Quantum Cognition",
				descZh:
					"运用非对易算符与量子干涉数学精确描述人类在不确定情境下的顺序效应与违背经典概率的判断。",
				descEn:
					"Using quantum probability theory to model contextual order effects in human decision-making.",
			},
		],
	},
];

/** 路线集群元数据 */
export const ROUTE_CLUSTERS: RouteCluster[] = [
	{
		id: "measurement",
		nameZh: "表现、测量与效率",
		nameEn: "Measurement & Efficiency",
		range: "路线 01 — 04",
		descZh:
			"什么样的表现算智能？它是一般能力还是结构组合？面对未知如何高效学到新技能？在资源与知识不足下如何适可而止？",
		descEn:
			"What behaviors qualify as intelligent? Unitary vs multidimensional capacity, sample-efficiency, and bounded resource adaptation.",
		color: "var(--brand-clay)",
	},
	{
		id: "representation",
		nameZh: "表征、推断与控制",
		nameEn: "Representation & Control",
		range: "路线 05 — 08",
		descZh:
			"符号逻辑推演、深度网络连续几何、贝叶斯与因果世界模型，以及闭环控制论与强化学习。",
		descEn:
			"Structured symbolic syntax, distributed continuous embeddings, causal counterfactual models, and closed-loop reinforcement.",
		color: "var(--brand-accent)",
	},
	{
		id: "embodiment",
		nameZh: "系统、组织与具身",
		nameEn: "Systems & Embodiment",
		range: "路线 09 — 13",
		descZh:
			"异构认知架构、脑网络整合、开放自演化学习、仿生连杆具身动力学与热力学自创生。",
		descEn:
			"Cognitive architectures, network neuroscience, open-ended curriculum, morphological passive dynamics, and autopoietic self-creation.",
		color: "var(--brand-blue)",
	},
	{
		id: "collective",
		nameZh: "集体、媒介与物理",
		nameEn: "Collective & Physics",
		range: "路线 14 — 16",
		descZh:
			"语言文化中介、延展至工具与社会的分布式心智，以及统计物理、算法压缩与临界态。",
		descEn:
			"Sociocultural ratchet, extended tool-coupled mind, and physical principles of algorithmic compression and criticality.",
		color: "var(--brand-ink)",
	},
	{
		id: "mind",
		nameZh: "语义、意识与价值",
		nameEn: "Mind & Meaning",
		range: "路线 17 — 20",
		descZh:
			"符号接地与理解本质、主观体验意识理论、实践智慧与跨文化传统，以及量子决策模型。",
		descEn:
			"Symbol grounding semantics, phenomenal consciousness, practical virtue, and non-classical foundational ontology.",
		color: "var(--brand-clay)",
	},
];

/** 辅助检索工具函数 */
export function getRouteById(id: string): RouteItem | undefined {
	return INTELLIGENCE_ROUTES.find((r) => r.id === id);
}

export function getRouteByIndex(idx: number): RouteItem | undefined {
	return INTELLIGENCE_ROUTES.find((r) => r.index === idx);
}

export function getRoutesByCluster(cluster: RouteItem["cluster"]): RouteItem[] {
	return INTELLIGENCE_ROUTES.filter((r) => r.cluster === cluster);
}

/** 获取特定路线在 8 大分歧轴上的精选理论立场与定位 */
export function getRouteStances(routeId: string): RouteStance[] {
	const stancesMap: Record<string, RouteStance[]> = {
		behavioral: [
			{
				axisIndex: 0,
				axisNameZh: "评价什么？",
				axisNameEn: "What to Evaluate?",
				poleAZh: "任务表现能力",
				poleAEn: "Task Competence",
				poleBZh: "获得新能力的效率",
				poleBEn: "Acquisition Efficiency",
				position: -0.8,
				rationaleZh:
					"高度侧重可观察的外部输出与基准任务成功率，避开不可验证的内在状态。",
				rationaleEn:
					"Focuses on external performance metrics rather than unobservable internal states.",
			},
			{
				axisIndex: 1,
				axisNameZh: "怎样解释行为？",
				axisNameEn: "How to Explain Behavior?",
				poleAZh: "内部表征与计算",
				poleAEn: "Internal Computation",
				poleBZh: "环境-主体刺激反应",
				poleBEn: "Environment Coupling",
				position: 0.7,
				rationaleZh:
					"将行为归因于强化历史与刺激-响应配对，拒绝假设复杂的内在符号计算。",
				rationaleEn:
					"Attributes capability to reinforcement histories rather than internal mental models.",
			},
			{
				axisIndex: 6,
				axisNameZh: "意识是否必要？",
				axisNameEn: "Is Consciousness Required?",
				poleAZh: "完全可独立于体验研究",
				poleAEn: "Independent of Experience",
				poleBZh: "需要主观现象感受",
				poleBEn: "Requires Phenomenality",
				position: -1.0,
				rationaleZh:
					"图灵判定与操作主义的基石：行为不可区分即视同智能，意识属于形而上伪问题。",
				rationaleEn:
					"Behavioral indistinguishability suffices; phenomenal experience is methodologically bracketed.",
			},
		],
		connectionist: [
			{
				axisIndex: 1,
				axisNameZh: "怎样解释行为？",
				axisNameEn: "How to Explain Behavior?",
				poleAZh: "内部连续表征流形",
				poleAEn: "Internal Representations",
				poleBZh: "实时动力学生理耦合",
				poleBEn: "Real-time Coupling",
				position: -0.9,
				rationaleZh:
					"核心在于隐空间连续几何向量变换与权重激活模式形成的内部表征。",
				rationaleEn:
					"Explains behavior through continuous geometry, latent embeddings, and forward passes.",
			},
			{
				axisIndex: 4,
				axisNameZh: "需要多少先验？",
				axisNameEn: "How Much Prior?",
				poleAZh: "通用空白统计学习 (Scaling)",
				poleAEn: "Tabula Rasa Scaling",
				poleBZh: "丰富核心知识与领域偏置",
				poleBEn: "Rich Innate Core Biases",
				position: -0.7,
				rationaleZh:
					"Scaling Law 假说认为只需通用 Transformer 架构与海量算力，极力减少手工领域预设。",
				rationaleEn:
					"Favors generic architectural priors scaled with data over handcrafted rules.",
			},
			{
				axisIndex: 5,
				axisNameZh: "是否存在统一机制？",
				axisNameEn: "Unified Mechanism?",
				poleAZh: "统一自监督预测原则",
				poleAEn: "Unified Predictive Loss",
				poleBZh: "异构多机制协同组织",
				poleBEn: "Heterogeneous Modules",
				position: -0.8,
				rationaleZh:
					"主张统一的自监督预测损失与梯度反传可以贯穿视觉、语言到具身的多模态理解。",
				rationaleEn:
					"Next-token prediction and gradient backprop as a potential master learning rule.",
			},
		],
		"causal-world-models": [
			{
				axisIndex: 0,
				axisNameZh: "评价什么？",
				axisNameEn: "What to Evaluate?",
				poleAZh: "静态关联拟合能力",
				poleAEn: "Correlation Fitting",
				poleBZh: "主动干预与反事实推演",
				poleBEn: "Counterfactual Reasoning",
				position: 0.9,
				rationaleZh:
					"严格要求模型能够回答‘若当初采取不同行动会怎样’，超越表层统计相关。",
				rationaleEn:
					"Demands causal intervention (do) and counterfactual invariance beyond correlation.",
			},
			{
				axisIndex: 1,
				axisNameZh: "怎样解释行为？",
				axisNameEn: "How to Explain Behavior?",
				poleAZh: "内部结构因果图与仿真",
				poleAEn: "Internal Causal Simulation",
				poleBZh: "表面反应耦合",
				poleBEn: "Surface Coupling",
				position: -0.8,
				rationaleZh: "依靠内在的因果结构图与环境动力学模型进行反事实虚拟推演。",
				rationaleEn:
					"Relies on explicit directed graphs and internal simulators of world mechanisms.",
			},
			{
				axisIndex: 4,
				axisNameZh: "需要多少先验？",
				axisNameEn: "How Much Prior?",
				poleAZh: "纯数据无偏归纳",
				poleAEn: "Pure Data Induction",
				poleBZh: "因果无混杂图结构先验",
				poleBEn: "Causal Graph Priors",
				position: 0.6,
				rationaleZh:
					"单纯靠观察数据无法完全消解混杂，必须引入因果方向或实验干预先验假设。",
				rationaleEn:
					"Pure observation cannot unmix confounding; structural assumptions are indispensable.",
			},
		],
		"cybernetics-rl": [
			{
				axisIndex: 1,
				axisNameZh: "怎样解释行为？",
				axisNameEn: "How to Explain Behavior?",
				poleAZh: "静态内部表征",
				poleAEn: "Static Representation",
				poleBZh: "闭环感知-行动动态耦合",
				poleBEn: "Closed-loop Action Coupling",
				position: 0.8,
				rationaleZh:
					"智能体现为与动态环境持续博弈、行动、获取反馈并修正策略的闭环过程。",
				rationaleEn:
					"Emphasizes the closed sensorimotor loop: perception, action, consequence, policy shift.",
			},
			{
				axisIndex: 2,
				axisNameZh: "什么使行动有方向？",
				axisNameEn: "What Orients Action?",
				poleAZh: "外部标量奖励信号优化",
				poleAEn: "External Scalar Reward",
				poleBZh: "内在稳态与自维持需求",
				poleBEn: "Intrinsic Homeostasis",
				position: -0.5,
				rationaleZh:
					"主流强化学习依赖环境反馈的奖励函数，主动推断则尝试连接到机体稳态自由能。",
				rationaleEn:
					"RL relies on environmental rewards, while active inference aligns with homeostatic priors.",
			},
			{
				axisIndex: 5,
				axisNameZh: "是否存在统一机制？",
				axisNameEn: "Unified Mechanism?",
				poleAZh: "单一奖励假说 (Reward Enough)",
				poleAEn: "Reward is Enough",
				poleBZh: "异构多机制协作",
				poleBEn: "Heterogeneous Society",
				position: -0.6,
				rationaleZh:
					"存在‘Reward is Enough’等强主张，试图用长期折扣累积回报统摄感知与规划。",
				rationaleEn:
					"Investigates whether cumulative reward maximization alone can induce all faculties.",
			},
		],
		"embodied-dynamical": [
			{
				axisIndex: 1,
				axisNameZh: "怎样解释行为？",
				axisNameEn: "How to Explain Behavior?",
				poleAZh: "脑内先算好再执行",
				poleAEn: "Central Computation First",
				poleBZh: "身体形态力学与实时动力学",
				poleBEn: "Morphological Dynamics",
				position: 0.95,
				rationaleZh:
					"极度强调形态计算与被动动力学：连杆几何与材料弹性本身承担了协调计算。",
				rationaleEn:
					"Mechanisms, geometry, and passive dynamics offload central computational burden.",
			},
			{
				axisIndex: 3,
				axisNameZh: "智能的单位是什么？",
				axisNameEn: "Locus of Intelligence?",
				poleAZh: "孤立中枢大脑/控制器",
				poleAEn: "Isolated Central Controller",
				poleBZh: "身体、环境与动力学统一场",
				poleBEn: "Body-Environment Coupling",
				position: 0.9,
				rationaleZh:
					"智能不是装在颅骨里的软件，而是‘大脑-身体-环境’共振形成的自组织吸引子。",
				rationaleEn:
					"Mind is an emergent attractor distributed across the brain-body-world continuum.",
			},
		],
		"autopoiesis-evolution": [
			{
				axisIndex: 2,
				axisNameZh: "什么使行动有方向？",
				axisNameEn: "What Orients Action?",
				poleAZh: "外部指定的目标或任务",
				poleAEn: "Externally Imposed Task",
				poleBZh: "系统自身的自维持与生存",
				poleBEn: "Autopoietic Self-production",
				position: 1.0,
				rationaleZh:
					"目标的根本来源是维持有机体边界与热力学生存的内在必然性，价值源于生存。",
				rationaleEn:
					"Agency originates from the thermodynamic necessity of regenerating self-identity.",
			},
			{
				axisIndex: 7,
				axisNameZh: "聪明是否包含价值？",
				axisNameEn: "Normative Values?",
				poleAZh: "工具性解题优化",
				poleAEn: "Instrumental Problem Solving",
				poleBZh: "对生命存续具有内在关切",
				poleBEn: "Intrinsic Normative Concern",
				position: 0.8,
				rationaleZh:
					"任何刺激之所以有‘好/坏’区分，是因为系统具备对自身免于解体瓦解的本能关切。",
				rationaleEn:
					"Normativity is rooted in life itself: things matter because existence is precarious.",
			},
		],
		"developmental-openended": [
			{
				axisIndex: 0,
				axisNameZh: "评价什么？",
				axisNameEn: "What to Evaluate?",
				poleAZh: "固定任务基准得分",
				poleAEn: "Static Benchmark Score",
				poleBZh: "持续自我演化与新挑战生成",
				poleBEn: "Continuous Open-ended Growth",
				position: 0.9,
				rationaleZh:
					"关注系统能否在开放环境中持续发明新问题、扩展新技能，而非刷固定的试卷。",
				rationaleEn:
					"Measures continuous novelty generation and perpetual capability expansion.",
			},
			{
				axisIndex: 5,
				axisNameZh: "是否存在统一机制？",
				axisNameEn: "Unified Mechanism?",
				poleAZh: "单阶段固定学习器",
				poleAEn: "Fixed Static Learner",
				poleBZh: "环境-智能体共同演化阶梯",
				poleBEn: "Co-evolutionary Ladder",
				position: 0.7,
				rationaleZh:
					"RSI 与开放式演化认为智能是环境复杂度与智能体自反思相互催化的动态产物。",
				rationaleEn:
					"Intelligence emerges from dynamic co-evolutionary ratchets and reflective loops.",
			},
		],
	};

	// 默认兜底立场（当某条路线未单独定制时提供通用的平衡投影）
	return (
		stancesMap[routeId] ?? [
			{
				axisIndex: 0,
				axisNameZh: "评价什么？",
				axisNameEn: "What to Evaluate?",
				poleAZh: "既有任务表现能力",
				poleAEn: "Existing Capabilities",
				poleBZh: "获得新技能的效率与潜力",
				poleBEn: "Learning Efficiency & Potential",
				position: 0.0,
				rationaleZh:
					"在该路线的理论视野中，两者保持着动态平衡与互为前提的关系。",
				rationaleEn:
					"Maintains a dynamic equilibrium between performance and adaptive potential.",
			},
			{
				axisIndex: 1,
				axisNameZh: "怎样解释行为？",
				axisNameEn: "How to Explain Behavior?",
				poleAZh: "内部表征与结构推演",
				poleAEn: "Internal Representations",
				poleBZh: "环境实时动态耦合",
				poleBEn: "Real-time Coupling",
				position: 0.0,
				rationaleZh:
					"深入探索内部信息组织如何有效映射并应对外部物理环境的不确定性。",
				rationaleEn:
					"Balances representational structure against situated environment interaction.",
			},
			{
				axisIndex: 5,
				axisNameZh: "是否存在统一机制？",
				axisNameEn: "Unified Mechanism?",
				poleAZh: "一般化单一原则",
				poleAEn: "Universal Principle",
				poleBZh: "异构多机制协同组织",
				poleBEn: "Heterogeneous Architecture",
				position: 0.0,
				rationaleZh:
					"既追求理论形式上的抽象概括力，亦尊重系统在实现层面的组件异质性。",
				rationaleEn:
					"Respects both theoretical generality and structural heterogeneity.",
			},
		]
	);
}

/** 路线之间的思想交锋（谁在批驳它）与互补桥梁（谁在补完它） */
export interface RouteDialogueLink {
	targetId: string;
	relation: "critique" | "synergy";
	titleZh: string;
	titleEn: string;
	summaryZh: string;
	summaryEn: string;
}

const ROUTE_DIALOGUE_MAP: Record<string, RouteDialogueLink[]> = {
	behavioral: [
		{
			targetId: "philosophy-of-mind",
			relation: "critique",
			titleZh: "塞尔中文房间的正面狙击",
			titleEn: "Challenged by Searle's Chinese Room",
			summaryZh:
				"纯靠查表操作字形也能通过图灵测试，证明‘外显行为等价’绝不自动蕴含真实的语义理解。",
			summaryEn:
				"Passing behavioral tests via syntactic lookup tables does not instantiate genuine semantic understanding.",
		},
		{
			targetId: "learning-efficiency",
			relation: "synergy",
			titleZh: "从静态表现升级为少样本泛化",
			titleEn: "Extended by ARC Acquisition Efficiency",
			summaryZh:
				"保留操作性可测量的优点，但将评测对象从‘死记题库得分’升级为‘面对未知任务的即时转化率’。",
			summaryEn:
				"Preserves operational measurability while shifting focus from static memory to novel skill acquisition.",
		},
	],
	psychometric: [
		{
			targetId: "neuroscience",
			relation: "synergy",
			titleZh: "P-FIT 脑网络提供生物学锚点",
			titleEn: "Grounded in P-FIT Brain Networks",
			summaryZh:
				"统计上的 g 因素并非虚构数字，它在顶额叶大尺度功能同调与过程重叠中找到了神经基质。",
			summaryEn:
				"Statistical g-factor variance maps onto parieto-frontal functional connectivity and overlapping neural hubs.",
		},
		{
			targetId: "sociocultural-language",
			relation: "critique",
			titleZh: "警惕脱离文化语境的单一智商还原",
			titleEn: "Critiqued by Sociocultural Contextuality",
			summaryZh:
				"纸笔测验衡量的是特定工业文化筛选出的符号技能，无法涵盖分布式社会协作与生态适应。",
			summaryEn:
				"Standardized tests reflect culturally specific schooling rather than universal ecological adaptability.",
		},
	],
	"learning-efficiency": [
		{
			targetId: "connectionist",
			relation: "critique",
			titleZh: "直指暴力规模定律的样本低效盲区",
			titleEn: "Exposes Sample Inefficiency of Pure Scaling",
			summaryZh:
				"吞吐万亿 Token 却在 1 像素平移的新规则网格前溃败，说明海量插值不等于高泛化转化率。",
			summaryEn:
				"Massive interpolation across trillions of tokens masks low sample efficiency on out-of-distribution structure.",
		},
		{
			targetId: "developmental-openended",
			relation: "synergy",
			titleZh: "与婴儿核心认知先验深度合流",
			titleEn: "Unified with Core Knowledge Priors",
			summaryZh:
				"人类之所以能极少样本泛化，正因为演化预装了物体性、几何对称与因果意图等发生学先验。",
			summaryEn:
				"Few-shot induction relies on evolutionary core priors of objectness, geometry, and agency.",
		},
	],
	"rational-resource": [
		{
			targetId: "symbolic-logic",
			relation: "critique",
			titleZh: "打破全知逻辑推理的无限算力神话",
			titleEn: "Refutes Unbounded Logical Omniscience",
			summaryZh:
				"完备的全局形式搜索在现实时间与能耗约束下往往不可行，智能必须懂得‘适可而止’（满意化）。",
			summaryEn:
				"Exhaustive formal deduction collapses under real-world time constraints; smart agents satisfice.",
		},
		{
			targetId: "information-physics",
			relation: "synergy",
			titleZh: "与热力学自由能及信息瓶颈同构",
			titleEn: "Isomorphic to Thermodynamic Information Bounds",
			summaryZh:
				"有限理性的计算代价在物理底层直接对应兰道尔擦除耗散与变分自由能上界。",
			summaryEn:
				"Cognitive deliberation costs map directly onto Landauer energetic costs and variational complexity.",
		},
	],
	"symbolic-logic": [
		{
			targetId: "embodied-dynamical",
			relation: "critique",
			titleZh: "莫拉维克悖论：高阶推理易，具身步态难",
			titleEn: "Hit by Moravec's Paradox & Embodied Dynamics",
			summaryZh:
				"能证明微积分定理的符号系统却连一岁婴儿的感知抓握都难以编码，暴露出离散符号与连续物理世界的脱节。",
			summaryEn:
				"Symbolic theorem provers struggle with basic sensorimotor coordination that passive mechanics solve effortlessly.",
		},
		{
			targetId: "connectionist",
			relation: "synergy",
			titleZh: "神经符号合流（Neuro-Symbolic AI）",
			titleEn: "Converging into Neuro-Symbolic Hybrids",
			summaryZh:
				"用连续向量网络负责感知流形降维，用符号搜索树负责可验证的多步严谨推演（如 AlphaGeometry）。",
			summaryEn:
				"Marries neural intuition for pattern proposal with formal symbolic search for rigorous verification.",
		},
	],
	connectionist: [
		{
			targetId: "causal-world-models",
			relation: "critique",
			titleZh: "珀尔因果之梯的拷问：关联拟合≠因果干预",
			titleEn: "Critiqued by Pearl's Ladder of Causation",
			summaryZh:
				"纯统计高维流形困于第一层‘观察关联’，面对混杂偏倚与分布外干预 do(X) 时极易产生幻觉。",
			summaryEn:
				"High-dimensional curve fitting remains trapped on Rung 1 (Association) and fails under structural interventions.",
		},
		{
			targetId: "symbolic-logic",
			relation: "synergy",
			titleZh: "引入测试时计算与树搜索克服自回归短视",
			titleEn: "Augmented by Test-Time Tree Search",
			summaryZh:
				"将直觉式前向向量生成（系统 1）与显式符号搜索回溯（系统 2）结合，突破纯自回归瓶颈。",
			summaryEn:
				"Combines fast vector intuition (System 1) with explicit tree search and verification (System 2).",
		},
	],
	"causal-world-models": [
		{
			targetId: "connectionist",
			relation: "critique",
			titleZh: "如何从非结构化高维像素中自动发现因果变量？",
			titleEn: "Challenged by Raw High-Dimensional Perception",
			summaryZh:
				"传统因果图预设抽象节点已由人工定义好，但在真实原始感官流中提取解耦因果变量仍需表征学习配合。",
			summaryEn:
				"Classical causal graphs assume pre-defined variables, requiring representation learning to extract them from raw pixels.",
		},
		{
			targetId: "cybernetics-rl",
			relation: "synergy",
			titleZh: "主动干预行动是揭开因果结构的唯一钥匙",
			titleEn: "Powered by Active Interventional Agency",
			summaryZh:
				"被动观察永远无法区分共因混杂，唯有像控制论主体那样主动摇晃世界（do-算子），才能锁定真实因果。",
			summaryEn:
				"Passive observation cannot resolve confounders; active closed-loop intervention unlocks causal structure.",
		},
	],
	"cybernetics-rl": [
		{
			targetId: "autopoiesis-evolution",
			relation: "critique",
			titleZh: "外生标量奖励函数从何而来？",
			titleEn: "Questioned on the Origin of Extrinsic Rewards",
			summaryZh:
				"强化学习预设环境直接下发标量 Reward，却回避了生命体如何在没有程序员打分的世界里自己产生生存目的。",
			summaryEn:
				"Assumes a pre-given scalar reward signal, whereas living organisms generate their own intrinsic existential goals.",
		},
		{
			targetId: "wisdom-values",
			relation: "synergy",
			titleZh: "受价值对齐与古德哈特定律约束",
			titleEn: "Tempered by Value Alignment & Wisdom",
			summaryZh:
				"单极最大化代理奖励必将引发回形针式灾难，闭环强化学习必须引入多目标伦理反思。",
			summaryEn:
				"Unchecked reward maximization triggers Goodhart's catastrophe; agency requires normative value balance.",
		},
	],
	"cognitive-architectures": [
		{
			targetId: "connectionist",
			relation: "critique",
			titleZh: "手工切分模块边界的脆弱性",
			titleEn: "Challenged by End-to-End Differentiable Learning",
			summaryZh:
				"人为硬编码记忆槽与产生式规则接口，在面对大规模模糊现实时常不如端到端可微网络灵活。",
			summaryEn:
				"Hand-crafted module interfaces often lose out to end-to-end differentiable training at scale.",
		},
		{
			targetId: "collective-extended",
			relation: "synergy",
			titleZh: "现代大模型 Agent 架构的直接先驱",
			titleEn: "Blueprint for Modern LLM Agent Frameworks",
			summaryZh:
				"ACT-R/Soar 的‘工作记忆总线 + 检索库 + 工具动作’正在当代 LLM Agent 工程中全面复兴。",
			summaryEn:
				"Working memory buffers, RAG declarative memory, and tool actions directly mirror classical cognitive architectures.",
		},
	],
	neuroscience: [
		{
			targetId: "philosophy-of-mind",
			relation: "critique",
			titleZh: "多重可实现性：飞翔未必需要羽毛",
			titleEn: "Debated by Functionalist Multiple Realizability",
			summaryZh:
				"功能主义指出飞机无需模仿鸟类拍打羽毛即可翱翔，硅基智能未必需要照搬生物离子通道细节。",
			summaryEn:
				"Airplanes fly without flapping feathers; intelligence may depend on functional organization rather than carbon biology.",
		},
		{
			targetId: "consciousness-self",
			relation: "synergy",
			titleZh: "与全局工作空间共同锁定意识神经相关物",
			titleEn: "Maps Neural Correlates of Consciousness",
			summaryZh:
				"顶额叶长程振荡同调直接为全局工作空间点火理论提供了可观测的脑电与脑成像实证。",
			summaryEn:
				"Long-range frontoparietal phase synchrony provides empirical grounding for Global Neuronal Workspace ignition.",
		},
	],
	"developmental-openended": [
		{
			targetId: "behavioral",
			relation: "critique",
			titleZh: "批判静态出厂测评：智能是动词而非名词",
			titleEn: "Rejects Static Frozen Evaluation",
			summaryZh:
				"冻结参数的模型即便考出高分也停留在过去，真正的发展智能在于持续改写自身认知基式。",
			summaryEn:
				"Frozen-weight evaluation misses the essence of ontogenetic growth and schema accommodation.",
		},
		{
			targetId: "sociocultural-language",
			relation: "synergy",
			titleZh: "皮亚杰个体建构与维果茨基社会脚手架合流",
			titleEn: "Bridges Piagetian Construction & Vygotskian Scaffolding",
			summaryZh:
				"个体好奇心探索与长辈/文化符号的‘最近发展区’引导共同铸就了人类心智发育。",
			summaryEn:
				"Individual intrinsic exploration flourishes within the Zone of Proximal Development scaffolded by culture.",
		},
	],
	"embodied-dynamical": [
		{
			targetId: "symbolic-logic",
			relation: "critique",
			titleZh: "面临‘表示饥渴’（Representation-Hungry）高阶任务考验",
			titleEn: "Tested by Representation-Hungry Abstract Tasks",
			summaryZh:
				"被动步态无需表征，但思考三年后的反事实规划或抽象数学证明仍离不开脱离当下感官的内部符号。",
			summaryEn:
				"Pure sensorimotor coupling excels at locomotion but struggles with counterfactual planning about absent things.",
		},
		{
			targetId: "autopoiesis-evolution",
			relation: "synergy",
			titleZh: "与生命自创生共同构成 4E 认知基石",
			titleEn: "Co-anchors 4E Enactive Cognition",
			summaryZh:
				"具身动力学解释了身体如何卸载计算，自创生解释了身体为何关切自身存续。",
			summaryEn:
				"Morphological dynamics offloads computation while autopoiesis supplies intrinsic existential purpose.",
		},
	],
	"autopoiesis-evolution": [
		{
			targetId: "cybernetics-rl",
			relation: "critique",
			titleZh: "揭示硅基程序缺乏真实生存关切（Precariousness）",
			titleEn: "Exposes Lack of Existential Precariousness in Silicon AI",
			summaryZh:
				"软件代码不会因为输掉游戏而在物理结构上解体消亡，因此不具备生物体那样真实的生存内驱力。",
			summaryEn:
				"Software agents do not physically dissolve when reward drops, lacking genuine biological precariousness.",
		},
		{
			targetId: "information-physics",
			relation: "synergy",
			titleZh: "自由能原理将生命自维持公理化",
			titleEn: "Formalized by the Free Energy Principle",
			summaryZh:
				"弗里斯顿主动推断将自创生边界形式化为‘马尔可夫毯’，把生命存续转化为最小化感官惊奇。",
			summaryEn:
				"Friston's Markov blankets mathematically formalize autopoietic boundary maintenance as surprise minimization.",
		},
	],
	"sociocultural-language": [
		{
			targetId: "psychometric",
			relation: "critique",
			titleZh: "解构‘孤岛鲁滨逊式’个体天才神话",
			titleEn: "Dismantles the Solitary Genius Myth",
			summaryZh:
				"脱离了数千年累积的语言词汇、数学记号与社会分工，任何高智商个体都会退回石器时代水平。",
			summaryEn:
				"Stripped of cumulative linguistic notations and cultural ratchets, solitary human brains revert to baseline.",
		},
		{
			targetId: "collective-extended",
			relation: "synergy",
			titleZh: "文化传承通过外部媒介工具固化为延展心智",
			titleEn: "Crystallizes into Extended Cognitive Artifacts",
			summaryZh:
				"代际文化棘轮不仅靠口耳相传，更沉淀为结绳、算盘、书籍与硅基代码等外部认知脚手架。",
			summaryEn:
				"Cumulative culture is anchored into physical external scaffolding, books, and computational media.",
		},
	],
	"collective-extended": [
		{
			targetId: "neuroscience",
			relation: "critique",
			titleZh: "面临‘颅骨内外机制非对称性’质疑",
			titleEn: "Debated on Brain-Artifact Asymmetry",
			summaryZh:
				"批评者（Adams & Aizawa）认为使用笔记本虽能辅助认知，但笔记本内部物理过程与神经元激活机制并不等同。",
			summaryEn:
				"Critics argue that causal coupling to a notebook does not make the paper itself literally cognitive.",
		},
		{
			targetId: "cognitive-architectures",
			relation: "synergy",
			titleZh: "赋能外挂检索（RAG）与代码沙盒调用",
			titleEn: "Empowers Tool-Augmented & RAG Agents",
			summaryZh:
				"将生物脑/神经网络从死记硬背与多位数心算中解放出来，专注于高层直觉调度与外部工具协同。",
			summaryEn:
				"Frees neural weights from brittle memorization by offloading exact state tracking to external tools.",
		},
	],
	"information-physics": [
		{
			targetId: "wisdom-values",
			relation: "critique",
			titleZh: "‘暗室问题’（Dark Room Problem）：最小化惊奇≠有意义的生活",
			titleEn: "Challenged by the Dark Room Problem",
			summaryZh:
				"如果智能仅仅是最小化预测误差与惊奇，为什么生物不躲进完全漆黑安静的房间里一动不动？",
			summaryEn:
				"If agents only minimize prediction error, why not sit forever in a pitch-black room where surprise is zero?",
		},
		{
			targetId: "rational-resource",
			relation: "synergy",
			titleZh: "构建信息压缩率与热力学能耗的统一帕累托边界",
			titleEn: "Unifies Compression and Energetic Efficiency",
			summaryZh:
				"将抽象概念学习严格证明为在有限兰道尔能耗预算下提取最小充分统计量的物理过程。",
			summaryEn:
				"Proves abstraction is the thermodynamic extraction of minimal sufficient statistics under energy constraints.",
		},
	],
	"philosophy-of-mind": [
		{
			targetId: "behavioral",
			relation: "critique",
			titleZh: "彻底终结朴素行为主义：表现相似≠心智等价",
			titleEn: "Refutes Naive Behavioral Equivalence",
			summaryZh:
				"中文房间与块头（Blockhead）思想实验证明，不审视内部因果接地结构，仅凭外部对话无法判定理解。",
			summaryEn:
				"Proves that external conversational fluency cannot guarantee genuine semantic grounding or intentionality.",
		},
		{
			targetId: "embodied-dynamical",
			relation: "synergy",
			titleZh: "通过具身感知运动闭环打破‘中文字典旋转木马’",
			titleEn: "Resolves Symbol Grounding via Sensorimotor Coupling",
			summaryZh:
				"哈纳德指出，唯有让离散符号锚定在非符号的具身感知与物理因果交互上，语义才能真正接地。",
			summaryEn:
				"Anchoring abstract tokens into non-symbolic sensorimotor causal loops breaks dictionary circularity.",
		},
	],
	"consciousness-self": [
		{
			targetId: "behavioral",
			relation: "critique",
			titleZh: "哲学僵尸（Philosophical Zombie）难题",
			titleEn: "Poses the Philosophical Zombie Dilemma",
			summaryZh:
				"在逻辑上可设想一个行为表现与常人完全相同、内部却毫无第一人称主观感受质（Qualia）的系统。",
			summaryEn:
				"Highlights the explanatory gap between third-person functional behavior and first-person phenomenal feel.",
		},
		{
			targetId: "neuroscience",
			relation: "synergy",
			titleZh: "以全局工作空间广播与整合信息 Φ 推进实证化",
			titleEn: "Operationalized via Global Ignition & Phi",
			summaryZh:
				"将原本不可言说的哲学意识转化为可检测的全脑非线性点火阈值与因果不可约性度量。",
			summaryEn:
				"Translates philosophical awareness into testable nonlinear cortical ignition and causal integration metrics.",
		},
	],
	"wisdom-values": [
		{
			targetId: "cybernetics-rl",
			relation: "critique",
			titleZh: "警示‘高智商低智慧’的工具理性异化",
			titleEn: "Warns Against Instrumental Paperclip Maximization",
			summaryZh:
				"算力与优化效率越高，一旦目标设定偏狭，其破坏生态与长期公共善的速度就越快。",
			summaryEn:
				"Raw optimization power without normative reflection accelerates Goodhart collapse and ethical blind spots.",
		},
		{
			targetId: "sociocultural-language",
			relation: "synergy",
			titleZh: "植根于跨文明伦理共同体与历史长周期检验",
			titleEn: "Rooted in Cross-Civilizational Practical Ethics",
			summaryZh:
				"无论是亚里士多德实践智慧还是儒道佛心性修养，都在具体社会情境与长远共同体利益中锤炼价值判断。",
			summaryEn:
				"Grounds normative judgment in pluralistic human traditions and long-term interpersonal equilibrium.",
		},
	],
	"metaphysics-quantum": [
		{
			targetId: "neuroscience",
			relation: "critique",
			titleZh: "温热潮湿大脑中的量子退相干挑战（Tegmark）",
			titleEn: "Challenged by Thermal Decoherence Timescales",
			summaryZh:
				"泰格马克计算指出，生物脑在室温下的退相干时间仅为飞秒级，远短于神经元毫秒级放电（注：量子认知概率模型不受此物理批判影响）。",
			summaryEn:
				"Physical quantum coherence in warm wet brains decoheres in femtoseconds, though quantum probability models of judgment remain valid.",
		},
		{
			targetId: "rational-resource",
			relation: "synergy",
			titleZh: "类量子非对易投影精准刻画人类语境决策偏差",
			titleEn: "Models Contextual Order Effects in Human Judgment",
			summaryZh:
				"放弃物理量子脑执念、将其作为希尔伯特空间概率工具时，完美解释了经典布尔概率无法解释的提问次序效应。",
			summaryEn:
				"Hilbert-space projection math elegantly captures human contextuality and question-order interference.",
		},
	],
};

export function getRouteDialogues(routeId: string): RouteDialogueLink[] {
	return ROUTE_DIALOGUE_MAP[routeId] ?? [];
}

/**
 * 20 条路线在全部 8 条核心分歧轴上的完整 8 维坐标矩阵 [-1.0, +1.0]
 * 用于 `/routes/` 全景页的 2D「思想引力象限仪」(Interactive Paradigm Compass)
 * 轴索引：
 * 0: 评价什么？ (-1 既有静态表现 <-> +1 新技能习得泛化效率)
 * 1: 怎样解释行为？ (-1 内部表征/符号计算 <-> +1 主体-环境实时动力学耦合)
 * 2: 什么使行动有方向？ (-1 外部给定奖励/损失 <-> +1 机体自维持/内生目的)
 * 3: 智能的单位是什么？ (-1 孤立个体/颅内大脑 <-> +1 具身-工具-社会延展系统)
 * 4: 需要多少先验？ (-1 白板通用统计缩放 <-> +1 丰富核心先验与结构偏置)
 * 5: 是否存在统一机制？ (-1 单一主算法/第一性原理 <-> +1 多模块异构协作架构)
 * 6: 意识是否必要？ (-1 可完全独立于主观体验 <-> +1 必须包含现象意识与内省)
 * 7: 聪明是否包含价值？ (-1 纯工具性解题效率 <-> +1 必须包含实践智慧与价值平衡)
 */
export const ROUTE_AXIS_MATRIX: Record<string, number[]> = {
	behavioral: [-0.85, 0.72, -0.75, -0.5, -0.6, -0.35, -0.95, -0.8],
	psychometric: [-0.7, -0.55, -0.6, -0.8, 0.35, -0.45, -0.7, -0.65],
	"learning-efficiency": [0.95, -0.45, 0.25, -0.4, 0.82, 0.3, -0.4, -0.3],
	"rational-resource": [0.35, -0.25, -0.3, -0.2, 0.4, -0.5, -0.6, 0.15],
	"symbolic-logic": [-0.45, -0.95, -0.65, -0.75, 0.88, 0.45, -0.85, -0.7],
	connectionist: [-0.25, -0.65, -0.8, -0.7, -0.92, -0.8, -0.75, -0.75],
	"causal-world-models": [0.65, -0.82, -0.2, -0.45, 0.78, -0.3, -0.5, -0.25],
	"cybernetics-rl": [0.45, 0.35, -0.55, 0.2, -0.45, -0.7, -0.65, -0.6],
	"cognitive-architectures": [
		0.15, -0.75, -0.25, -0.3, 0.65, 0.95, -0.2, -0.35,
	],
	neuroscience: [0.1, -0.3, 0.35, -0.55, 0.55, 0.75, 0.35, -0.2],
	"developmental-openended": [0.92, 0.4, 0.85, 0.25, 0.6, 0.65, 0.2, 0.3],
	"embodied-dynamical": [0.3, 0.95, 0.55, 0.82, 0.45, 0.5, 0.15, 0.1],
	"autopoiesis-evolution": [0.55, 0.85, 0.98, 0.6, 0.7, -0.25, 0.65, 0.75],
	"sociocultural-language": [0.5, 0.45, 0.4, 0.92, 0.3, 0.6, 0.4, 0.65],
	"collective-extended": [0.4, 0.65, 0.2, 0.98, 0.15, 0.8, -0.1, 0.25],
	"information-physics": [0.25, -0.15, 0.45, 0.1, -0.35, -0.95, -0.3, -0.45],
	"philosophy-of-mind": [0.2, -0.4, 0.5, 0.15, 0.5, 0.25, 0.75, 0.4],
	"consciousness-self": [0.15, -0.2, 0.75, -0.35, 0.4, 0.35, 0.98, 0.55],
	"wisdom-values": [0.6, 0.25, 0.9, 0.75, 0.55, 0.7, 0.8, 0.98],
	"metaphysics-quantum": [0.05, -0.1, 0.6, 0.35, 0.25, -0.4, 0.88, 0.35],
};
