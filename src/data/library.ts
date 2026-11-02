export interface LibraryEntry {
	id: string;
	title: string;
	authors: string;
	year: string;
	kind: string;
	kindEn: string;
	url: string;
	topic: "frontier" | "models" | "search" | "evaluation" | "embodied";
	questionId?: string;
	focus: string;
	focusEn: string;
	notes: string[];
}

export const library: LibraryEntry[] = [
	{
		id: "deepseek-moe",
		title:
			"DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models",
		authors: "Damai Dai, Chengqi Deng, Chenggang Zhao et al. (DeepSeek-AI)",
		year: "2024",
		kind: "论文 · P-01",
		kindEn: "Paper · P-01",
		url: "https://arxiv.org/abs/2401.06066",
		topic: "frontier",
		questionId: "Q-02 稀疏激活与专家分工",
		focus:
			"研究细粒度专家切分（Fine-Grained Expert Segmentation）与共享专家隔离（Shared Expert Isolation）如何消除传统粗粒度 MoE 的知识冗余与路由退化。",
		focusEn:
			"Examine how fine-grained expert segmentation and shared expert isolation eliminate knowledge redundancy and routing collapse in conventional MoE.",
		notes: [
			"2026-10-20-llm-training-pretrain-03-forward",
			"2024-09-01-transformer架构详解",
		],
	},
	{
		id: "deepseek-math",
		title:
			"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
		authors: "Zhihong Shao, Peiyi Wang, Qihao Zhu et al. (DeepSeek-AI)",
		year: "2024",
		kind: "论文 · P-02",
		kindEn: "Paper · P-02",
		url: "https://arxiv.org/abs/2402.03300",
		topic: "frontier",
		questionId: "Q-04 无 Critic 推理强化学习",
		focus:
			"追踪大规模数学语料挖掘与 GRPO（Group Relative Policy Optimization）的起源：如何用同题组内采样奖励归一化省去与策略同规模的 Critic 网络。",
		focusEn:
			"Trace the origin of GRPO and mathematical web corpus mining: replacing a full-scale value critic with group-relative reward normalization.",
		notes: [
			"2026-10-23-llm-training-rl-05-grpo",
			"2026-10-15-llm-training-data-03-quality",
		],
	},
	{
		id: "deepseek-v2",
		title:
			"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
		authors: "DeepSeek-AI",
		year: "2024",
		kind: "论文与开源 · P-03",
		kindEn: "Paper & Repo · P-03",
		url: "https://github.com/deepseek-ai/DeepSeek-V2",
		topic: "frontier",
		questionId: "Q-01 KV Cache 显存与带宽墙",
		focus:
			"重点拆解 MLA（多头潜在注意力）：如何通过低秩 KV 联合压缩、推理期矩阵吸收（Matrix Absorption）与解耦 RoPE 大幅削减 KV Cache 显存而不损失表达力。",
		focusEn:
			"Dissect Multi-Head Latent Attention (MLA): low-rank KV joint compression, inference matrix absorption, and decoupled RoPE for massive KV cache reduction.",
		notes: [
			"2026-10-20-llm-training-pretrain-03-forward",
			"2024-10-27-llm推理优化与部署",
		],
	},
	{
		id: "deepseek-v3",
		title: "DeepSeek-V3 Technical Report",
		authors: "DeepSeek-AI",
		year: "2024",
		kind: "技术报告 · P-04",
		kindEn: "Tech Report · P-04",
		url: "https://github.com/deepseek-ai/DeepSeek-V3",
		topic: "frontier",
		questionId: "Q-02 / Q-03 / Q-05 全链路协同",
		focus:
			"还原 671B（激活 37B）稀疏大模型的稳定低成本训练：无辅助损失负载均衡（动态偏置更新）、多 Token 预测（MTP）、细粒度 Tile/Block FP8 混合精度与 DualPipe 双向流水线。",
		focusEn:
			"Reconstruct 671B MoE co-design: auxiliary-loss-free load balancing, Multi-Token Prediction (MTP), fine-grained FP8 quantization, and DualPipe overlap.",
		notes: [
			"2026-10-20-llm-training-pretrain-05-optimization",
			"2026-10-23-llm-training-eval-03-costs",
		],
	},
	{
		id: "deepseek-r1",
		title:
			"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
		authors: "DeepSeek-AI",
		year: "2025",
		kind: "论文与开源 · P-05",
		kindEn: "Paper & Repo · P-05",
		url: "https://github.com/deepseek-ai/DeepSeek-R1",
		topic: "frontier",
		questionId: "Q-04 推理涌现与冷启动 RL",
		focus:
			"对比 R1-Zero（纯规则 RL 激发自反思长思维链但伴随可读性退化）与 R1（冷启动可读数据 + 多阶段 RL + 拒绝采样 SFT + 蒸馏）两条路径的能力与边界。",
		focusEn:
			"Compare R1-Zero pure-RL emergence against R1 multi-stage cold-start RL and distillation across reasoning accuracy and readability.",
		notes: [
			"2026-10-23-llm-training-rl-05-grpo",
			"2026-10-23-llm-training-rl-06-evaluation",
		],
	},
	{
		id: "deepseek-v3-2",
		title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models",
		authors: "DeepSeek-AI",
		year: "2025",
		kind: "技术报告 · P-06",
		kindEn: "Tech Report · P-06",
		url: "https://arxiv.org/abs/2512.02556",
		topic: "frontier",
		questionId: "Q-01 / Q-06 稀疏注意力与 Agent RL",
		focus:
			"剖析动态稀疏注意力（DSA 与闪电索引器 Lightning Indexer）如何将长上下文复杂度降维，并与面向大规模合成工具环境的 Agent 强化学习协同。",
		focusEn:
			"Analyze how Dynamic Sparse Attention (DSA with Lightning Indexer) pairs with large-scale synthetic agentic tool-use reinforcement learning.",
		notes: [
			"2026-10-23-llm-training-tool-01-trajectories",
			"2026-10-23-llm-training-tool-03-rewards",
		],
	},
	{
		id: "deepseek-math-v2",
		title: "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning",
		authors: "DeepSeek-AI",
		year: "2025",
		kind: "论文 · P-07",
		kindEn: "Paper · P-07",
		url: "https://arxiv.org/abs/2511.22570",
		topic: "frontier",
		questionId: "Q-04 从答案奖励走向过程验证",
		focus:
			"当最终标量答案不足以保证推导严密性或问题本身是定理证明时，如何训练可自验证的证明验证器与生成器闭环。",
		focusEn:
			"Move from outcome-only scalar rewards to self-verifiable mathematical proof generation and process verification loops.",
		notes: [
			"2026-10-23-llm-training-rl-02-verifiers",
			"2026-10-23-llm-training-rl-06-evaluation",
		],
	},
	{
		id: "deepseek-mhc",
		title: "mHC: Manifold-Constrained Hyper-Connections",
		authors: "Zhenda Xie, Yixuan Wei, Huanqi Cao et al. (DeepSeek-AI)",
		year: "2025",
		kind: "论文 · P-08",
		kindEn: "Paper · P-08",
		url: "https://arxiv.org/abs/2512.24880",
		topic: "frontier",
		questionId: "Q-03 残差连接扩展与深层稳定性",
		focus:
			"研究无约束超连接（HC）为何破坏残差恒等映射导致深层梯度/激活爆炸，以及如何通过 Sinkhorn-Knopp 算法将连接矩阵投影到双随机流形恢复稳定性。",
		focusEn:
			"Examine how projecting multi-stream hyper-connection matrices onto the Birkhoff polytope restores identity-mapping stability at scale.",
		notes: [
			"2026-10-20-llm-training-pretrain-03-forward",
			"2026-10-20-llm-training-pretrain-05-optimization",
		],
	},
	{
		id: "deepseek-engram",
		title:
			"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram)",
		authors: "Xin Cheng, Wangding Zeng, Damai Dai et al. (DeepSeek-AI & PKU)",
		year: "2026",
		kind: "论文与开源 · P-09",
		kindEn: "Paper & Repo · P-09",
		url: "https://github.com/deepseek-ai/Engram",
		topic: "frontier",
		questionId: "Q-02 静态记忆查找与动态计算分离",
		focus:
			"探索与 MoE 互补的第二条稀疏轴：用现代哈希 N-gram 查找表实现 O(1) 条件记忆检索，将静态世界知识存储从动态 FFN 计算中解耦并支持主机内存卸载。",
		focusEn:
			"Decouple O(1) hashed N-gram conditional memory lookup from dynamic MoE neural computation, enabling host-DRAM offloading.",
		notes: [
			"2026-10-20-llm-training-pretrain-01-tokenization",
			"2026-10-20-llm-training-pretrain-03-forward",
		],
	},
	{
		id: "deepseek-v4",
		title:
			"DeepSeek-V4 & V4.1-Flash Research Line: Million-Token Context, CED & Cross-Layer KV Reuse",
		authors: "DeepSeek-AI (Four-Tier Evidence Audit Entry)",
		year: "2026",
		kind: "前沿追踪 · P-10/P-11",
		kindEn: "Frontier Track · P-10/P-11",
		url: "https://www.deepseek.com/en/transparency/",
		topic: "frontier",
		questionId: "Q-01 / Q-05 不对称编解码与跨层 KV",
		focus:
			"围绕百万 Token 上下文与输入/输出不对称计算（Causal Encoder–Decoder、CSA2、跨层 KV 复用）开展机制推演，并严格按四级证据标准区分官方公开材料与待验假设。",
		focusEn:
			"Investigate million-token context, Causal Encoder–Decoder (CED) asymmetric computation, and cross-layer KV reuse under strict four-tier evidence auditing.",
		notes: [
			"2024-10-27-llm推理优化与部署",
			"2026-10-23-llm-training-eval-02-evidence",
		],
	},
	{
		id: "deepseek-dualpath",
		title:
			"DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference",
		authors:
			"Yongtong Wu, Shaoyuan Chen, Yinmin Zhong et al. (DeepSeek-AI, PKU, Tsinghua)",
		year: "2026",
		kind: "系统论文 · P-12",
		kindEn: "Systems Paper · P-12",
		url: "https://arxiv.org/abs/2602.21548",
		topic: "frontier",
		questionId: "Q-01 / Q-06 Agent 推理 KV 存储带宽墙",
		focus:
			"剖析多轮 Agent 推理高达 95%+ KV 缓存命中率下的 PD 分离瓶颈：如何同时打通“存储→Prefill”与“存储→Decode（经 RDMA 转发）”双路径以消除存储网卡不对称拥塞。",
		focusEn:
			"Break the KV-cache storage I/O bottleneck in multi-turn agentic inference via dual-path loading across disaggregated Prefill and Decode engines.",
		notes: [
			"2024-10-27-llm推理优化与部署",
			"2026-10-23-llm-training-tool-01-trajectories",
		],
	},
	{
		id: "minimind",
		title: "MiniMind",
		authors: "jingyaogong and contributors",
		year: "Project",
		kind: "训练项目",
		kindEn: "Training project",
		url: "https://github.com/jingyaogong/minimind",
		topic: "models",
		focus:
			"对照小模型的数据、预训练与后训练组织。固定上游版本，单独测量本机资源与吞吐。",
		focusEn:
			"Compare the organization of data, pretraining, and post-training. Pin an upstream version and measure resources and throughput locally.",
		notes: [
			"2026-07-08-1b模型全链路实验计划",
			"2026-10-20-llm-training-pretrain-06-experiment",
		],
	},
	{
		id: "nanochat",
		title: "nanochat",
		authors: "Andrej Karpathy and contributors",
		year: "Project",
		kind: "训练项目",
		kindEn: "Training project",
		url: "https://github.com/karpathy/nanochat",
		topic: "models",
		focus:
			"沿分词器、预训练、后训练、评测与推理追踪产物。区分GPU规模示例与CPU/MPS教育配置。",
		focusEn:
			"Trace artifacts through tokenization, pretraining, post-training, evaluation, and inference. Distinguish GPU-scale examples from educational CPU/MPS configurations.",
		notes: [
			"2026-07-08-1b模型全链路实验计划",
			"2026-10-20-llm-training-pretrain-01-tokenization",
		],
	},
	{
		id: "transformer",
		title: "Attention Is All You Need",
		authors: "Ashish Vaswani et al.",
		year: "2017",
		kind: "论文",
		kindEn: "Paper",
		url: "https://arxiv.org/abs/1706.03762",
		topic: "models",
		focus:
			"先追踪注意力、多头机制与位置表示之间的数据流，再回到现代 Pre-RMSNorm / RoPE / SwiGLU 演进笔记，理解架构改变了哪些计算。",
		focusEn:
			"Follow the data flow through attention, multiple heads, and positional representations, then compare with modern Pre-RMSNorm, RoPE, and SwiGLU architectures.",
		notes: [
			"2026-10-20-llm-training-pretrain-03-forward",
			"2024-09-01-transformer架构详解",
		],
	},
	{
		id: "instructgpt",
		title:
			"Training language models to follow instructions with human feedback (InstructGPT)",
		authors: "Long Ouyang et al.",
		year: "2022",
		kind: "论文",
		kindEn: "Paper",
		url: "https://arxiv.org/abs/2203.02155",
		topic: "models",
		focus:
			"认识监督微调（SFT）、成对人类偏好排序、奖励模型（RM）与强化学习微调的三阶段衔接，区分对齐表面格式与真实能力边界。",
		focusEn:
			"Understand the three-stage pipeline of SFT, pairwise preference reward modeling, and RLHF, separating format alignment from capability boundaries.",
		notes: [
			"2026-10-20-llm-training-sft-01-demonstrations",
			"2026-10-20-llm-training-preference-01-feedback",
		],
	},
	{
		id: "dpo",
		title:
			"Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
		authors: "Rafael Rafailov et al.",
		year: "2023",
		kind: "论文",
		kindEn: "Paper",
		url: "https://arxiv.org/abs/2305.18290",
		topic: "models",
		focus:
			"从 KL 约束奖励最大化目标的解析最优解出发，理解 DPO 如何用策略与参考模型的对数概率比直接替代显式奖励模型，并审视离线分布偏移问题。",
		focusEn:
			"Derive how DPO reparameterizes KL-constrained reward maximization using policy-to-reference log-probability ratios, and examine offline distribution shift.",
		notes: [
			"2026-10-20-llm-training-preference-04-dpo",
			"2026-10-20-llm-training-preference-05-offline-vs-online",
		],
	},
	{
		id: "react",
		title: "ReAct: Synergizing Reasoning and Acting in Language Models",
		authors: "Shunyu Yao et al.",
		year: "2022",
		kind: "论文",
		kindEn: "Paper",
		url: "https://arxiv.org/abs/2210.03629",
		topic: "models",
		focus:
			"看推理、动作与环境观察如何交替出现。将工具返回的反馈与模型内部的推断分开，思考错误会在哪一步累积。",
		focusEn:
			"Observe how reasoning, actions, and observations alternate. Separate feedback returned by tools from model-generated inferences, and ask where errors can accumulate.",
		notes: [
			"2026-10-23-llm-training-tool-01-trajectories",
			"2024-10-13-llm-agent开发指南",
		],
	},
	{
		id: "mcts-survey",
		title: "A Survey of Monte Carlo Tree Search Methods",
		authors: "Cameron Browne et al.",
		year: "2012",
		kind: "综述",
		kindEn: "Survey",
		url: "https://doi.org/10.1109/TCIAIG.2012.2186810",
		topic: "search",
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
		topic: "search",
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
		topic: "evaluation",
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
		topic: "evaluation",
		focus:
			"结合官方文档理解处理效应估计的工作流。使用具体估计器前，先明确处理、结果、协变量与识别假设。",
		focusEn:
			"Use the official documentation to follow a treatment-effect workflow. Before choosing an estimator, specify treatment, outcome, covariates, and identification assumptions.",
		notes: ["2024-6-30-应用ylearn框架实现因果推断"],
	},
	{
		id: "flywire-connectome",
		title:
			"Neuronal wiring diagram of an adult brain (FlyWire Whole-Brain Drosophila Connectome)",
		authors:
			"Sven Dorkenwald, Arie Matsliah, Amy R. Sterling et al. (FlyWire Consortium)",
		year: "2024",
		kind: "论文与图谱",
		kindEn: "Paper & Atlas",
		url: "https://doi.org/10.1038/s41586-024-07558-y",
		topic: "embodied",
		focus:
			"基于约 14 万神经元与 5000 万突触的果蝇全脑连接组，开展预注册保度打乱零模型（Degree-Preserving Shuffled Controls）对照实验与阴性结果边界审计。",
		focusEn:
			"Audit the whole-brain fruit fly connectome (~140k neurons) against degree-preserving shuffled null models and preregistered falsification criteria.",
		notes: [
			"2026-10-21-fly-connectome-01-circuit-and-framework",
			"2026-10-21-fly-connectome-11-summary",
		],
	},
	{
		id: "strandbeest-kinematics",
		title: "The Great Pretender / Strandbeest Planar Linkage Evolution",
		authors: "Theo Jansen (with evolutionary & wind-tunnel verification notes)",
		year: "2007",
		kind: "机械具身专著",
		kindEn: "Embodied Monograph",
		url: "https://www.strandbeest.com/",
		topic: "embodied",
		focus:
			"追踪 11 根平面连杆“圣数（Holy Numbers）”的演化算法复现、闭环步态运动学以及风洞能量转化与阴性边界实测。",
		focusEn:
			"Reconstruct the evolutionary optimization of Theo Jansen’s 11-bar linkage ratios, closed-loop gait kinematics, and wind-energy empirical limits.",
		notes: [
			"2026-10-04-re-evolving-the-strandbeest-leg",
			"2026-10-28-strandbeest-wind-evidence-and-negative-results",
		],
	},
];
