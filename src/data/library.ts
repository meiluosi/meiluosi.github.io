export interface LibraryEntry {
	id: string;
	title: string;
	authors: string;
	year: string;
	kind: string;
	kindEn: string;
	url: string;
	path: string;
	focus: string;
	focusEn: string;
	notes: string[];
}
export const library: LibraryEntry[] = [
	{
		id: "minimind",
		title: "MiniMind",
		authors: "jingyaogong and contributors",
		year: "Project",
		kind: "训练项目",
		kindEn: "Training project",
		url: "https://github.com/jingyaogong/minimind",
		path: "models-to-systems",
		focus:
			"对照小模型的数据、预训练与后训练组织。固定上游版本，单独测量本机资源与吞吐。",
		focusEn:
			"Compare the organization of data, pretraining, and post-training. Pin an upstream version and measure resources and throughput locally.",
		notes: ["2026-07-08-1b模型全链路实验计划"],
	},
	{
		id: "nanochat",
		title: "nanochat",
		authors: "Andrej Karpathy and contributors",
		year: "Project",
		kind: "训练项目",
		kindEn: "Training project",
		url: "https://github.com/karpathy/nanochat",
		path: "models-to-systems",
		focus:
			"沿分词器、预训练、后训练、评测与推理追踪产物。区分GPU规模示例与CPU/MPS教育配置。",
		focusEn:
			"Trace artifacts through tokenization, pretraining, post-training, evaluation, and inference. Distinguish GPU-scale examples from educational CPU/MPS configurations.",
		notes: ["2026-07-08-1b模型全链路实验计划"],
	},

	{
		id: "mcts-survey",
		title: "A Survey of Monte Carlo Tree Search Methods",
		authors: "Cameron Browne et al.",
		year: "2012",
		kind: "综述",
		kindEn: "Survey",
		url: "https://doi.org/10.1109/TCIAIG.2012.2186810",
		path: "search-to-self-play",
		focus:
			"从四个搜索阶段读起，再看 UCT 如何权衡探索与利用。对照互动实验，理解访问次数和奖励如何沿树回传。",
		focusEn:
			"Start with the four search phases, then examine the exploration–exploitation trade-off in UCT. Compare the interactive tree’s visit and reward updates.",
		notes: ["2024-08-20-mcts算法深度解析"],
	},
	{
		id: "alphazero",
		title:
			"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm",
		authors: "David Silver et al.",
		year: "2017",
		kind: "论文",
		kindEn: "Paper",
		url: "https://arxiv.org/abs/1712.01815",
		path: "search-to-self-play",
		focus:
			"关注自我对弈、搜索与策略价值网络的连接方式。把搜索产生的动作分布，与训练时更新的模型参数区分开。",
		focusEn:
			"Trace the connection between self-play, search, and the policy-value network. Distinguish the action distribution produced by search from the parameters updated during training.",
		notes: ["2024-6-30-alpha-zero算法实现五子棋"],
	},
	{
		id: "causal-remix",
		title: "Causal Inference: The Remix",
		authors: "Scott Cunningham",
		year: "Online edition",
		kind: "教材",
		kindEn: "Textbook",
		url: "https://mixtape.scunning.com/",
		path: "evaluating-improvement",
		focus:
			"从因果图与潜在结果开始，追问每一种比较成立的条件。作者的在线第二版仍在更新；旧笔记引用的书名为 The Mixtape。",
		focusEn:
			"Begin with causal graphs and potential outcomes, asking which assumptions justify each comparison. The author’s online second edition is a work in progress; older notes cite The Mixtape.",
		notes: [
			"2024-08-15-统计学基础与假设检验",
			"2024-6-30-应用ylearn框架实现因果推断",
		],
	},
	{
		id: "ylearn",
		title: "YLearn",
		authors: "DataCanvasIO",
		year: "Project",
		kind: "工具",
		kindEn: "Tool",
		url: "https://github.com/DataCanvasIO/YLearn",
		path: "evaluating-improvement",
		focus:
			"结合官方文档理解处理效应估计的工作流。使用具体估计器前，先明确处理、结果、协变量与识别假设。",
		focusEn:
			"Use the official documentation to follow a treatment-effect workflow. Before choosing an estimator, specify treatment, outcome, covariates, and identification assumptions.",
		notes: ["2024-6-30-应用ylearn框架实现因果推断"],
	},
	{
		id: "transformer",
		title: "Attention Is All You Need",
		authors: "Ashish Vaswani et al.",
		year: "2017",
		kind: "论文",
		kindEn: "Paper",
		url: "https://arxiv.org/abs/1706.03762",
		path: "models-to-systems",
		focus:
			"先追踪注意力、多头机制与位置表示之间的数据流，再回到模型训练笔记，理解架构决定了哪些计算。",
		focusEn:
			"Follow the data flow through attention, multiple heads, and positional representations. Then return to the training notes to connect architecture with computation.",
		notes: ["2024-09-01-transformer架构详解", "2024-09-15-llm训练技术详解"],
	},
	{
		id: "react",
		title: "ReAct: Synergizing Reasoning and Acting in Language Models",
		authors: "Shunyu Yao et al.",
		year: "2022",
		kind: "论文",
		kindEn: "Paper",
		url: "https://arxiv.org/abs/2210.03629",
		path: "models-to-systems",
		focus:
			"看推理、动作与环境观察如何交替出现。将工具返回的反馈与模型内部的推断分开，思考错误会在哪一步累积。",
		focusEn:
			"Observe how reasoning, actions, and observations alternate. Separate feedback returned by tools from model-generated inferences, and ask where errors can accumulate.",
		notes: ["2024-10-13-llm-agent开发指南"],
	},
];
