export const caseStudies = [
	{
		slug: "local-1b-model-workflow",
		match: "1b模型全链路实验计划",
		title: "从数据到部署的小模型训练",
		titleEn: "Small-model training, from data to deployment",
		dek: "用一个可逐步扩大的本地实验，把数据、预训练、后训练与评测完整连接起来。",
		dekEn:
			"Connect data, pretraining, post-training, and evaluation in a local experiment that can grow in stages.",
		question: "一个语言模型，怎样从原始文本和随机权重走到可评估的回答？",
		questionEn:
			"How does a language model go from raw text and random weights to answers we can evaluate?",
		claim:
			"从约 30M 模型开始，理解每一阶段的输入、产物和评价方法；再根据数据与资源测量，把同一条流程扩展到更大的规模。",
		claimEn:
			"Begin around 30M parameters to understand each stage’s inputs, artifacts, and evaluation, then scale the workflow using data and resource measurements.",
		approach: [
			{
				title: "先固定数据与资源预算",
				body: "在 Apple M4、16 GB 统一内存上，从来源清单、文档划分和分词器开始，固定模型配置与 token 预算。",
				titleEn: "Fix the data and resource budget",
				bodyEn:
					"On Apple M4 with 16 GB unified memory, start with source manifests, document splits, and a tokenizer, then fix model settings and token budgets.",
			},
			{
				title: "从随机权重走到后训练",
				body: "主线依次研究预训练、assistant-only SFT 和带冻结参考模型的 DPO；现成 0.5B 基座的 LoRA 作为另一条应用对照。",
				titleEn: "From random weights to post-training",
				bodyEn:
					"Study pretraining, assistant-only SFT, and DPO with a frozen reference; use an existing 0.5B base model with LoRA as a separate comparison.",
			},
			{
				title: "让每次迭代可以比较",
				body: "各阶段保留模型、配置和评测记录，用固定数据、提示与解码设置观察能力变化、退化和运行成本。",
				titleEn: "Make iterations comparable",
				bodyEn:
					"Keep checkpoints, configurations, and evaluation records at every stage; use fixed data, prompts, and decoding to observe changes, regressions, and cost.",
			},
		],
		result:
			"已加入全链路互动导读，并在 M4 的 MPS 后端跑通 138,560 参数 tiny 基线：预训练 12 步、SFT 8 步、DPO 8 步，另验证恢复与 CLI 加载。这是自写样本上的工程短跑，不代表语言能力；约 30M 及以上仍需正式数据与资源测量。",
		resultEn:
			"The interactive guide is backed by a 138,560-parameter tiny MPS run on M4: 12 pretraining steps, 8 SFT steps, and 8 DPO steps, with resume and CLI-loading checks. This brief engineering run uses authored fixtures and establishes no language capability; 30M and larger experiments still need real datasets and resource measurements.",
		rsi: "每轮迭代都从可检查的反馈出发：分析错误，调整数据或训练方法，再用一致的评测比较。这个过程把持续改进拆成可逐步研究的环节。",
		rsiEn:
			"Each iteration starts with inspectable feedback: analyze errors, adjust data or training, and compare with consistent evaluation. This makes continual improvement a sequence of researchable steps.",
		boundaries: [
			"约 30M 模型完成闭环需要多少数据与训练预算？",
			"增加模型容量、改变数据和后训练方法，各自带来什么变化？",
			"在本机测量结果下，约 100M 与 0.5B 分别适合怎样的实验？",
		],
		boundariesEn: [
			"What data and training budget are needed for a complete run around 30M?",
			"How do capacity, data changes, and post-training each affect behavior?",
			"Which experiments fit the 100M and 0.5B tiers under measured local limits?",
		],
	},
	{
		slug: "alphazero-gomoku",
		match: "alphazero算法实现五子棋",
		title: "用自我对弈训练五子棋策略",
		titleEn: "Self-play for Gomoku strategy learning",
		dek: "把策略-价值网络、PUCT 搜索、自我对弈和训练数据回放连成一个 AlphaZero 风格的学习环。",
		dekEn:
			"Connect a policy-value network, PUCT search, self-play, and replayed training data in an AlphaZero-style learning loop.",
		question: "搜索产生的反馈，怎样变成更好的下一步策略？",
		questionEn:
			"How can feedback from search become a better policy on the next iteration?",
		claim:
			"这个实验把策略价值网络、PUCT 搜索与自我对弈连接起来，观察搜索产生的更强策略如何反过来训练网络。",
		claimEn:
			"This experiment connects a policy-value network, PUCT search, and self-play to study how a stronger search policy can train the network in return.",
		approach: [
			{
				title: "编码棋盘并估计策略与价值",
				body: "状态用当前玩家和对手的两张棋盘平面表示。PyTorch 网络由卷积主干分出策略头与价值头，分别给出落子先验和当前局面的价值估计。",
				titleEn: "Encode the board and estimate policy and value",
				bodyEn:
					"The state uses two board planes for the current player and opponent. A PyTorch convolutional network branches into policy and value heads to estimate move priors and the value of the position.",
			},
			{
				title: "用 PUCT 把网络与树搜索结合",
				body: "文章给出 PUCT 节点结构与选择、扩展、评估、回传流程，并将合法动作的网络先验归一化后用于搜索。",
				titleEn: "Combine the network with PUCT tree search",
				bodyEn:
					"The article sketches PUCT nodes and the selection, expansion, evaluation, and backup steps, normalizing network priors over legal actions before search.",
			},
			{
				title: "从对局生成训练样本",
				body: "自我对弈流程以搜索分布作为策略目标，将局面、策略分布和终局结果写入 replay buffer，再用策略与价值损失更新网络。",
				titleEn: "Generate training examples through self-play",
				bodyEn:
					"The self-play sketch uses the search distribution as a policy target, stores positions, policies, and final outcomes in a replay buffer, then updates the network with policy and value losses.",
			},
		],
		result:
			"当前实现梳理了从棋盘编码、树搜索到自我对弈数据回放的核心循环。下一步是补齐稳定训练、基线对手和固定赛制，观察 loss、胜率与搜索预算之间的关系。",
		resultEn:
			"The implementation lays out the core loop from board encoding and tree search to replayed self-play data. The next step is stable training with baseline opponents and a fixed match protocol, tracking loss, win rate, and search budget together.",
		rsi: "自我对弈提供了一个紧凑的改进循环：系统产生行为，从结果中获得反馈，再更新下一轮策略。它让“如何变得更好”成为可以观察和比较的过程。",
		rsiEn:
			"Self-play creates a compact improvement loop: act, receive feedback from outcomes, then update the next policy. It makes the process of getting better observable and comparable.",
		boundaries: [
			"补齐环境状态推进与完整训练脚本，让搜索、自我对弈和网络更新可以端到端复现。",
			"建立固定基线、对弈赛制和棋力曲线，区分网络学习与搜索预算各自带来的提升。",
		],
		boundariesEn: [
			"Complete environment stepping and the training script so search, self-play, and network updates can run end to end.",
			"Add fixed baselines, a match protocol, and a strength curve to separate gains from learning and gains from search budget.",
		],
	},
	{
		slug: "kuhn-poker-cfr",
		match: "应用cfr实现德州扑克对战",
		title: "从 Kuhn Poker 理解 CFR",
		titleEn: "CFR through Kuhn Poker",
		dek: "先在小型不完全信息博弈中实现反事实后悔最小化，再把规模化德州扑克作为后续工程问题。",
		dekEn:
			"Implement counterfactual regret minimization in a small imperfect-information game, then treat full-scale poker as a separate engineering challenge.",
		question: "在看不见对手私有信息时，策略如何利用反事实反馈改进？",
		questionEn:
			"How can a strategy improve from counterfactual feedback when opponents' private information is hidden?",
		claim:
			"这个实验先用 Kuhn Poker 的小型博弈结构实现 CFR，再思考信息集、动作抽象和求解规模如何扩展到更复杂的扑克环境。",
		claimEn:
			"This experiment first implements CFR in the compact game of Kuhn Poker, then asks how information sets, action abstraction, and solver scale change in larger poker environments.",
		approach: [
			{
				title: "用信息集表示不完全信息",
				body: "Kuhn Poker 使用 J、Q、K 三张牌。信息集由玩家手牌和行动历史组成，使策略在不观察对手底牌的条件下做决策。",
				titleEn: "Represent hidden information with information sets",
				bodyEn:
					"Kuhn Poker uses three cards—J, Q, and K. An information set combines a player's private card with the public action history, so decisions do not reveal the opponent's card.",
			},
			{
				title: "累计反事实后悔并匹配策略",
				body: "递归遍历可能行动，计算各动作相对当前策略的价值差，将对手到达该信息集的概率用于累计后悔，再按正后悔比例形成新策略，同时累积平均策略。",
				titleEn: "Accumulate counterfactual regret and match strategies",
				bodyEn:
					"The recursive traversal evaluates possible actions, accumulates action-value differences weighted by the opponent's reach probability, then uses positive regret proportions to form a new strategy while tracking the average strategy.",
			},
			{
				title: "把更大规模留作下一步",
				body: "示例代码将 pass/bet 两个动作设置为 50,000 次训练迭代。原文另讨论公共牌抽样、下注尺度抽象、CFR+ 和 exploitability 等扩展与评估方向。",
				titleEn: "Treat larger poker as a separate next step",
				bodyEn:
					"The example configures 50,000 training iterations with pass/bet actions. The article separately discusses public-card sampling, bet-size abstraction, CFR+, and evaluation with exploitability.",
			},
		],
		result:
			"当前实现把信息集、递归遍历、后悔累计与平均策略连接成一条完整思路。下一步会记录平均博弈价值与策略收敛过程，并加入 exploitability 或固定对手评估。",
		resultEn:
			"The implementation connects information sets, recursive traversal, regret accumulation, and average strategy into one clear path. The next step is to record game value and strategy convergence, then add exploitability or fixed-opponent evaluation.",
		rsi: "CFR 展示了策略如何累积反事实反馈并逐步减少遗憾。这个小型博弈把反馈驱动的策略改进压缩成一个容易观察的实验环境。",
		rsiEn:
			"CFR shows a strategy accumulating counterfactual feedback and reducing regret over time. The compact game turns feedback-driven improvement into an environment that is easy to inspect.",
		boundaries: [
			"记录 Kuhn Poker 策略随训练迭代的变化，并用理论均衡或 exploitability 检查收敛。",
			"再单独研究完整扑克环境中的状态、下注尺度与公共牌抽象，而不混淆两种问题规模。",
		],
		boundariesEn: [
			"Track how the Kuhn Poker strategy changes across iterations and test convergence against the theoretical equilibrium or exploitability.",
			"Study state, bet-size, and public-card abstraction for full poker separately, without conflating the two problem scales.",
		],
	},
	{
		slug: "ylearn-causal-inference",
		match: "应用ylearn框架实现因果推断",
		title: "用模拟优惠券数据比较因果估计方法",
		titleEn: "Comparing causal estimators on simulated coupon data",
		dek: "通过已知生成机制的模拟数据，检查混淆如何影响朴素比较，并演示多种处理效应估计方法。",
		dekEn:
			"Use data with a known synthetic mechanism to inspect confounding in naive comparisons and demonstrate several treatment-effect estimators.",
		question: "当更活跃的用户更容易收到优惠券时，如何区分选择偏差与干预效应？",
		questionEn:
			"When more active users are more likely to receive a coupon, how can selection bias be separated from the treatment effect?",
		claim:
			"这个实验用生成机制已知的模拟优惠券数据，把选择偏差、处理效应与个体差异放进同一个可控环境中比较。",
		claimEn:
			"This experiment uses simulated coupon data with a known generating process to compare selection bias, treatment effects, and individual differences in one controlled setting.",
		approach: [
			{
				title: "构造带混淆的已知数据",
				body: "代码以随机种子 2024 生成 5,000 条记录，用年龄、收入和活跃度生成优惠券倾向，并按 50 + 10·(年龄>30) − 5·(收入>30,000) 构造异质处理效应，再加入消费基线与噪声。",
				titleEn: "Generate data with a known confounding mechanism",
				bodyEn:
					"With seed 2024, the code generates 5,000 records. Age, income, and activity influence coupon assignment; heterogeneous treatment effects follow 50 + 10·(age>30) − 5·(income>30,000), alongside a spending baseline and noise.",
			},
			{
				title: "比较 ATE 与个体差异估计",
				body: "文章将数据按 70/30 划分训练集与测试集，示例比较 S、T、X 和 DR-Learner 的 ATE，并用 X-Learner 估计 CATE。",
				titleEn: "Compare ATE and heterogeneous-effect estimates",
				bodyEn:
					"The article splits the synthetic data 70/30 into train and test sets, compares S-, T-, X-, and DR-Learner ATE estimates, and uses X-Learner to estimate CATE.",
			},
			{
				title: "延伸到 uplift 与因果图",
				body: "其余代码示例涉及 uplift 排序与曲线、PC/GES 因果发现、IPW 加权和敏感性分析，覆盖从估计到验证假设的多个环节。",
				titleEn: "Extend the analysis to uplift and causal graphs",
				bodyEn:
					"Additional code sketches uplift ranking and curves, PC/GES causal discovery, IPW weighting, and sensitivity analysis, spanning several steps from estimation to assumption checks.",
			},
		],
		result:
			"当前分析已经搭起从数据生成、ATE/CATE 估计到 uplift、因果发现与敏感性分析的路径。下一步会统一保存各方法的误差、稳定性和排序质量，让比较更直观。",
		resultEn:
			"The analysis now connects data generation, ATE/CATE estimation, uplift, causal discovery, and sensitivity analysis. The next step is to save error, stability, and ranking-quality results for every method in one comparable view.",
		rsi: "它对应 RSI 路径中的评估问题：系统发生变化之后，怎样判断改进来自干预本身，而不是选择偏差或环境波动？因果方法为这种判断提供了一套语言。",
		rsiEn:
			"It addresses the evaluation side of RSI: after a system changes, how can we tell whether improvement came from the intervention rather than selection bias or environmental variation? Causal methods provide a language for that judgment.",
		boundaries: [
			"在已知真实效应的模拟数据上保存各估计器的偏差、方差与排序质量。",
			"再引入更复杂的混淆、模型错设和数据漂移，观察哪些结论仍然稳定。",
		],
		boundariesEn: [
			"Save bias, variance, and ranking quality for each estimator on simulated data with known effects.",
			"Introduce stronger confounding, model misspecification, and data drift to see which conclusions remain stable.",
		],
	},
] as const;
