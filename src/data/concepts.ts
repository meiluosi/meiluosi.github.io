export interface ResearchConcept {
	id: string;
	term: string;
	name: string;
	nameEn: string;
	definition: string;
	definitionEn: string;
	href: string;
	hrefEn: string;
}
export const concepts: ResearchConcept[] = [
	{
		id: "rsi",
		term: "RSI",
		name: "递归自我改进",
		nameEn: "Recursive self-improvement",
		definition:
			"一个系统的改进，使它进一步提升自身产生改进的能力。研究这个过程，需要同时考虑反馈、更新机制与持续有效的评估。",
		definitionEn:
			"Improvements to a system enhance its ability to produce further improvements. Studying this process involves feedback, update mechanisms, and evaluations that remain informative across iterations.",
		href: "/rsi/",
		hrefEn: "/rsi/",
	},
	{
		id: "agi",
		term: "AGI",
		name: "通用人工智能",
		nameEn: "Artificial general intelligence",
		definition:
			"指能够跨多种任务与情境学习、推理和行动的通用智能目标。不同研究对能力范围与判断标准仍有不同定义。",
		definitionEn:
			"The goal of intelligence that learns, reasons, and acts across diverse tasks and contexts. Definitions differ in the scope of capabilities and criteria used to assess them.",
		href: "/rsi/",
		hrefEn: "/rsi/",
	},
	{
		id: "mcts",
		term: "MCTS",
		name: "蒙特卡洛树搜索",
		nameEn: "Monte Carlo tree search",
		definition:
			"通过反复选择、扩展、模拟与回传，逐步分配搜索预算的决策方法。树中积累的访问次数与奖励统计帮助选择动作。",
		definitionEn:
			"A decision method that allocates search through repeated selection, expansion, simulation, and backup. Visit and reward statistics in the tree inform action selection.",
		href: "/lab/mcts/",
		hrefEn: "/lab/mcts/",
	},
	{
		id: "uct",
		term: "UCT",
		name: "树上的置信上界选择",
		nameEn: "Upper confidence bounds applied to trees",
		definition:
			"MCTS 中的一种选择规则：把已观察到的平均奖励与探索项相加，兼顾表现较好的分支和尝试较少的分支。",
		definitionEn:
			"A selection rule for MCTS that adds an exploration bonus to the observed mean reward, balancing promising branches with less-visited ones.",
		href: "/lab/mcts/",
		hrefEn: "/lab/mcts/",
	},
	{
		id: "puct",
		term: "PUCT",
		name: "带策略先验的树搜索",
		nameEn: "Tree search with policy priors",
		definition:
			"在价值估计之外，利用策略先验与访问次数决定探索强度。AlphaZero 风格的搜索用策略网络引导动作探索。",
		definitionEn:
			"Uses policy priors and visit counts alongside value estimates to guide exploration. AlphaZero-style search uses a policy network to direct which actions to explore.",
		href: "/posts/2024-6-30-alpha-zero算法实现五子棋/",
		hrefEn: "/en/posts/alpha-zero-gomoku/",
	},
	{
		id: "ate",
		term: "ATE",
		name: "平均处理效应",
		nameEn: "Average treatment effect",
		definition:
			"目标人群中，接受与不接受某项处理的潜在结果之差的平均值。它不一定等于观察到的处理组和对照组均值之差。",
		definitionEn:
			"The average difference between potential outcomes with and without treatment in a target population. It need not equal the observed difference between treated and control groups.",
		href: "/lab/causal-playground/",
		hrefEn: "/lab/causal-playground/",
	},
	{
		id: "cate",
		term: "CATE",
		name: "条件平均处理效应",
		nameEn: "Conditional average treatment effect",
		definition:
			"给定一组特征时的平均处理效应，用来描述不同人群可能有不同的干预效果。可靠估计仍取决于识别假设与数据。",
		definitionEn:
			"The average treatment effect conditional on a set of features, describing how effects may vary across groups. Reliable estimation still depends on identification assumptions and data.",
		href: "/posts/2024-6-30-应用ylearn框架实现因果推断/",
		hrefEn: "/en/posts/ylearn-causal-inference/",
	},
	{
		id: "rag",
		term: "RAG",
		name: "检索增强生成",
		nameEn: "Retrieval-augmented generation",
		definition:
			"先从外部资料检索相关内容，再让模型结合这些内容生成回答。检索质量、上下文组织和回答依据都需要分别评估。",
		definitionEn:
			"Retrieves relevant external material and uses it as context for generation. Retrieval quality, context construction, and answer grounding each need evaluation.",
		href: "/posts/2024-10-06-rag检索增强生成实战/",
		hrefEn: "/en/posts/rag-retrieval-augmented-generation/",
	},
];
