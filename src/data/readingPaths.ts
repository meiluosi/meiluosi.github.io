export interface ReadingStep {
	sourceId: string;
	focus: string;
	focusEn: string;
}
export interface ReadingPath {
	slug: string;
	number: string;
	title: string;
	titleEn: string;
	question: string;
	questionEn: string;
	audience: string;
	audienceEn: string;
	prerequisites: string;
	prerequisitesEn: string;
	outcome: string;
	outcomeEn: string;
	reflection: string;
	reflectionEn: string;
	caseHref: string;
	steps: ReadingStep[];
}
export const readingPaths: ReadingPath[] = [
	{
		slug: "search-to-self-play",
		number: "01",
		title: "从搜索到自我对弈",
		titleEn: "From search to self-play",
		question: "搜索产生的反馈，怎样成为下一轮学习的材料？",
		questionEn:
			"How does feedback from search become material for the next round of learning?",
		audience: "想理解 AlphaZero 学习循环，以及搜索与训练如何配合的读者。",
		audienceEn:
			"For readers exploring the AlphaZero learning loop and the relationship between search and training.",
		prerequisites: "能阅读基础 Python；知道策略、奖励与神经网络训练的含义。",
		prerequisitesEn:
			"Basic Python, plus familiarity with policies, rewards, and neural-network training.",
		outcome:
			"区分搜索时的策略改进、训练时的参数更新，以及评估时需要控制的条件。",
		outcomeEn:
			"Distinguish policy improvement during search, parameter updates during training, and the conditions needed for evaluation.",
		reflection:
			"如果只增加搜索次数，棋力变强了，这与模型本身学到了新能力有什么区别？",
		reflectionEn:
			"If more search makes play stronger, how does that differ from the model learning a new capability?",
		caseHref: "/work/alphazero-gomoku/",
		steps: [
			{
				sourceId: "2024-08-20-mcts算法深度解析",
				focus:
					"先理解选择、扩展、模拟和回传。留意探索项如何分配有限的搜索预算。",
				focusEn:
					"Start with selection, expansion, simulation, and backup. Watch how exploration allocates a limited search budget.",
			},
			{
				sourceId: "2024-6-30-alpha-zero算法实现五子棋",
				focus: "把搜索接上策略价值网络，沿着自我对弈、数据回放与训练走完一轮。",
				focusEn:
					"Connect search to a policy-value network, then follow one loop of self-play, replay, and training.",
			},
			{
				sourceId: "2024-07-12-actor-critic算法家族",
				focus:
					"换到 Actor–Critic 视角，比较价值估计与策略更新在不同学习方法中的作用。",
				focusEn:
					"Compare how value estimation and policy updates work from an actor–critic perspective.",
			},
		],
	},
	{
		slug: "evaluating-improvement",
		number: "02",
		title: "怎样判断改进有效",
		titleEn: "How to evaluate improvement",
		question: "观察到差异之后，怎样追问差异从何而来？",
		questionEn: "After observing a difference, how do we ask what caused it?",
		audience: "准备比较模型、策略或干预效果，希望更清楚地解释结果的读者。",
		audienceEn:
			"For readers comparing models, policies, or interventions and seeking a clearer interpretation of results.",
		prerequisites: "理解均值、方差与概率；代码部分使用 Python 数据分析工具。",
		prerequisitesEn:
			"Means, variance, and probability; the code uses Python data-analysis tools.",
		outcome:
			"区分统计差异、相关性与因果效应，并识别比较中的人群差异和混杂因素。",
		outcomeEn:
			"Distinguish statistical differences, associations, and causal effects, and recognize selection and confounding.",
		reflection:
			"如果两组样本的组成不同，即使指标提升，也能归因于你的方法吗？还需要哪些假设？",
		reflectionEn:
			"If the groups have different compositions, can a metric gain be attributed to your method? What assumptions are still needed?",
		caseHref: "/work/ylearn-causal-inference/",
		steps: [
			{
				sourceId: "2024-08-15-统计学基础与假设检验",
				focus: "先建立假设检验、效应量和不确定性的语言，避免只盯着一个平均分。",
				focusEn:
					"Build a vocabulary for hypothesis tests, effect sizes, and uncertainty beyond a single average score.",
			},
			{
				sourceId: "2024-6-30-应用ylearn框架实现因果推断",
				focus:
					"沿着模拟优惠券案例，理解 ATE、CATE、混杂和处理效应估计之间的关系。",
				focusEn:
					"Follow the simulated coupon example to connect ATE, CATE, confounding, and effect estimation.",
			},
			{
				sourceId: "2024-07-22-ducg建模实战指南",
				focus:
					"把问题改写为图：有哪些变量、哪些因果假设，以及怎样组织推理路径。",
				focusEn:
					"Represent the problem as a graph: variables, causal assumptions, and inference paths.",
			},
		],
	},
	{
		slug: "models-to-systems",
		number: "03",
		title: "从模型到智能系统",
		titleEn: "From models to intelligent systems",
		question: "模型能力如何在训练、部署和工具使用之间形成完整工作流？",
		questionEn:
			"How do training, deployment, and tool use fit into a complete model workflow?",
		audience: "希望把大模型各个环节联系起来，而后动手构建系统的读者。",
		audienceEn:
			"For readers connecting the pieces of language-model development before building a system.",
		prerequisites: "基础深度学习与 Python；熟悉张量、梯度和训练集的基本概念。",
		prerequisitesEn:
			"Basic deep learning and Python, including tensors, gradients, and training data.",
		outcome:
			"画出架构、训练、推理、Agent 与评估之间的数据流，并找出每一环的约束。",
		outcomeEn:
			"Map the flow between architecture, training, inference, agents, and evaluation, including each stage’s constraints.",
		reflection:
			"一次回答变好，可能来自模型、检索、工具还是提示？怎样设计对照才能分清？",
		reflectionEn:
			"Did a better answer come from the model, retrieval, a tool, or the prompt? What comparison would tell them apart?",
		caseHref: "/work/local-1b-model-workflow/",
		steps: [
			{
				sourceId: "2024-09-01-transformer架构详解",
				focus: "从自注意力与位置编码开始，理解模型怎样组织输入信息。",
				focusEn:
					"Start with self-attention and positional encoding to understand how the model organizes input.",
			},
			{
				sourceId: "2024-09-15-llm训练技术详解",
				focus: "区分预训练、监督微调与偏好对齐各自使用的数据和优化目标。",
				focusEn:
					"Distinguish the data and objectives used in pretraining, supervised fine-tuning, and preference alignment.",
			},
			{
				sourceId: "2026-07-08-1b模型全链路实验计划",
				focus:
					"以小模型为起点，串联数据、分词、随机初始化预训练、后训练与固定集评测。",
				focusEn:
					"Use a small model to connect data, tokenization, pretraining from scratch, post-training, and held-out evaluation.",
			},
			{
				sourceId: "2024-10-27-llm推理优化与部署",
				focus: "进入运行阶段，理解量化、缓存和部署条件怎样影响体验。",
				focusEn:
					"Move into runtime: quantization, caching, and deployment conditions shape the experience.",
			},
			{
				sourceId: "2024-10-13-llm-agent开发指南",
				focus: "加入规划、记忆与工具调用，观察系统能力怎样超出单次模型调用。",
				focusEn:
					"Add planning, memory, and tools to examine capabilities beyond a single model call.",
			},
		],
	},
];
