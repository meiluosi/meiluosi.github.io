export const researchArchive = [
	{
		match: "1b模型全链路实验计划",
		caseSlug: "local-1b-model-workflow",
		kind: "PRETRAINING · POST-TRAINING",
		kindZh: "预训练 · 后训练",
		thread: "MODEL SYSTEMS",
		threadZh: "模型系统",
		status: "TINY PIPELINE VERIFIED · 30M PLANNED",
		statusZh: "tiny 工程链路已验证 · 30M 待实验",
		summary:
			"从数据与分词器出发，以约 30M 模型规划预训练、SFT、DPO、评测与部署，逐步探索更大的模型。",
		titleEn: "From data to deployment: small-model training",
		summaryEn:
			"Start with data and tokenization, then plan pretraining, SFT, DPO, evaluation, and deployment around a 30M model before scaling.",
		evidence: [
			"Apple M4 / 16 GB 统一内存",
			"数据 → 分词 → 随机初始化预训练 → SFT → DPO → 评测 → 部署",
			"约 30M 起步，约 100M 进阶；0.5B 在资源测量后决定",
		],
		evidenceEn: [
			"Apple M4 / 16 GB unified memory",
			"Data → tokenizer → pretraining from scratch → SFT → DPO → evaluation → deployment",
			"Start around 30M, progress toward 100M; measure resources before the 0.5B tier",
		],
	},
	{
		match: "alphazero算法实现五子棋",
		caseSlug: "alphazero-gomoku",
		kind: "SELF-PLAY · GOMOKU",
		kindZh: "自我对弈 · 五子棋",
		thread: "SELF-PLAY & LEARNING",
		threadZh: "自我对弈与学习",
		status: "POLICY-VALUE · PUCT · SELF-PLAY",
		statusZh: "策略价值 · PUCT · 自我对弈",
		summary:
			"把策略价值网络、PUCT MCTS、自我对弈和训练回放连接成一个 AlphaZero 风格的学习循环。",
		titleEn: "AlphaZero Gomoku implementation",
		summaryEn:
			"An AlphaZero-style learning loop connecting a policy-value network, PUCT MCTS, self-play, and replayed training data.",
		evidence: [
			"文章中的实现范围：PyTorch 策略-价值网络与 PUCT 搜索",
			"包含 replay buffer、自我对弈与训练循环代码",
			"文章建议从 9×9 棋盘、约 200–800 次搜索和八向数据增强开始",
		],
		evidenceEn: [
			"Implementation scope in the note: a PyTorch policy-value network and PUCT search",
			"Includes replay buffer, self-play, and a training loop",
			"The note suggests starting with a 9×9 board, roughly 200–800 searches, and eight-way augmentation",
		],
	},
	{
		match: "应用ylearn框架实现因果推断",
		caseSlug: "ylearn-causal-inference",
		kind: "CAUSAL INFERENCE · SIMULATED DATA",
		kindZh: "因果推断 · 模拟数据",
		thread: "CAUSAL INFERENCE",
		threadZh: "因果推断",
		status: "ATE · CATE · UPLIFT",
		statusZh: "ATE · CATE · Uplift",
		summary:
			"用模拟优惠券数据比较处理效应估计方法，并延伸到 uplift modeling、因果图发现与敏感性分析。",
		titleEn: "Causal inference with YLearn",
		summaryEn:
			"A YLearn study on simulated coupon data, spanning treatment-effect estimation, uplift modeling, causal discovery, and sensitivity analysis.",
		evidence: [
			"文章中的示例由代码生成 5000 条模拟优惠券数据",
			"实现并比较 S/T/X/DR-Learner 的 ATE，并估计 CATE",
			"另含 uplift curve、PC/GES 因果图发现、IPW 与敏感性分析示例",
		],
		evidenceEn: [
			"The note generates 5,000 rows of simulated coupon data",
			"Compares S/T/X/DR-Learner ATE estimates and estimates CATE",
			"Also includes uplift curves, PC/GES graph discovery, IPW, and sensitivity analysis examples",
		],
	},
	{
		match: "应用cfr实现德州扑克对战",
		caseSlug: "kuhn-poker-cfr",
		kind: "IMPERFECT INFORMATION · KUHN POKER",
		kindZh: "不完全信息博弈 · Kuhn Poker",
		thread: "FEEDBACK & STRATEGY",
		threadZh: "反馈与策略",
		status: "CFR · REGRET MATCHING · STRATEGY",
		statusZh: "CFR · 后悔匹配 · 策略",
		summary:
			"从 Kuhn Poker 的递归 CFR 与后悔匹配出发，理解不完全信息博弈中的策略更新。",
		titleEn: "Kuhn Poker CFR implementation",
		summaryEn:
			"A recursive CFR and regret-matching study using Kuhn Poker to explore strategy updates in imperfect-information games.",
		evidence: [
			"文章包含 Kuhn Poker 的递归 CFR 与 regret matching Python 实现",
			"代码示例以 pass/bet 两个动作训练 50,000 次迭代",
			"另行讨论德州扑克抽象、CFR+ 与 exploitability/对战胜率评估方式",
		],
		evidenceEn: [
			"The note includes a recursive CFR and regret-matching Python implementation for Kuhn Poker",
			"The example trains with pass/bet actions for 50,000 iterations",
			"It also discusses Texas Hold’em abstractions, CFR+, and exploitability or match-win evaluation",
		],
	},
	{
		match: "fly-connectome-01-circuit-and-framework",
		caseSlug: "flywire-connectome-audit",
		kind: "CONNECTOMICS · PREREGISTERED AUDIT",
		kindZh: "全脑连接组 · 预注册实证审计",
		thread: "EMBODIED & NEUROSCIENCE",
		threadZh: "生物连接组与具身",
		status: "11 CHAPTERS · NEGATIVE RESULTS REPORTED",
		statusZh: "11 章完结 · 含阴性结果诚实报告",
		summary:
			"基于 FlyWire 果蝇全脑突触连接组与蘑菇体 15 个多巴胺舱室，在严格保度随机重连零模型（Degree-Preserving Shuffled Controls）下检验觅食、持续学习与组合优化边界。",
		titleEn: "FlyWire Drosophila connectome preregistered audit",
		summaryEn:
			"A preregistered 11-chapter empirical audit of the FlyWire fruit fly connectome against degree-preserving shuffled controls across foraging, continual learning, and negative-result boundaries.",
		evidence: [
			"严格锁定保度随机重连零模型（Degree-Preserving Null Models）与相等读出层容量",
			"覆盖气味觅食、自然气味协方差阳性对照、15 个多巴胺舱室冲突仲裁",
			"专门以整章报告静态连接组在通用持续学习与 TSP 上的阴性证伪结论",
		],
		evidenceEn: [
			"Enforces degree-preserving shuffled null models and equal readout capacity",
			"Covers odor foraging, natural odor covariance positive controls, and 15 dopaminergic compartments",
			"Dedicates full chapters to honest negative results on continual learning and TSP",
		],
	},
	{
		match: "re-evolving-the-strandbeest-leg",
		caseSlug: "strandbeest-linkage-evolution",
		kind: "EMBODIED MECHANICS · EVOLUTION",
		kindZh: "具身连杆力学 · 演化复现",
		thread: "PHYSICAL EMBODIMENT",
		threadZh: "物理具身与形态计算",
		status: "11-BAR KINEMATICS · WIND AUDIT",
		statusZh: "11 杆闭环运动学 · 风能实测边界",
		summary:
			"无需微控制器，仅靠 11 根平面连杆的长度比例（Holy Numbers）将圆周曲柄旋转转化为平底步态，并检验演化算法复现与风能转化边界。",
		titleEn: "Re-evolving the Strandbeest planar linkage",
		summaryEn:
			"Reconstructing Theo Jansen’s 11-bar linkage ratios via evolutionary optimization, closed-loop kinematics, and wind-propulsion boundary audits.",
		evidence: [
			"完整推导 11 根连杆几何约束与曲柄全周可转性（Grashof 判据）",
			"对比演化算法搜索轨迹与 Jansen 原始圣数的步态平直度与地面冲击",
			"配套风力推进效率与机械死点阴性结果实证记录",
		],
		evidenceEn: [
			"Full geometric kinematics and Grashof crank-rotatability verification",
			"Compares evolutionary search trajectories against Jansen’s original Holy Numbers",
			"Includes empirical wind-propulsion efficiency and mechanical lockup negative results",
		],
	},
] as const;
