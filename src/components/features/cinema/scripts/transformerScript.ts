import type { CinemaScript } from "../types";

export const TRANSFORMER_CINEMA_SCRIPT: CinemaScript = {
	id: "transformer-token-odyssey",
	titleZh: "一个 Token 的时空之旅：90秒纯代码交互长片",
	titleEn: "A Token's Odyssey in Transformer: 90s Interactive Code Cinema",
	subtitleZh:
		"多头分立特化、缩放点积方差稳定、Value 向量加权聚合、MLP 键值记忆与 KV Cache 增量推理闭环",
	subtitleEn:
		"Multi-head specialization, scaled dot-product variance stabilization, Value vector aggregation, MLP key-value memory recall, and KV Cache inference closure",
	totalDurationSec: 90,
	acts: [
		{
			id: "act-1-vocab-cosmos",
			actNumber: "ACT I",
			titleZh: "BPE 分词子词切分与高维嵌入映射",
			titleEn: "BPE Tokenization & Embedding Space Mapping",
			subtitleZh:
				"连续文本切分为离散词元序列，词元【思】在嵌入空间映射为向量并叠加正弦位置编码 [POS=3]",
			subtitleEn:
				"Continuous text is segmented into discrete tokens; token [思] maps to an embedding vector and receives sinusoidal [POS=3]",
			startSec: 0,
			durationSec: 18,
			subtitles: [
				{
					timeRange: [0, 5],
					speakerZh: "输入序列",
					speakerEn: "Input Sequence",
					tagZh: "输入提示词 Prompt",
					tagEn: "Prompt Input",
					textZh:
						"人类在终端输入自然语言问句：“人工智能能够思考吗”。在张量计算引擎中，连续文本需要先被离散符号化。",
					textEn:
						"A user inputs the natural language query: 'Can AI think?'. At the tensor computation layer, continuous text must first be discretized into discrete symbolic units.",
				},
				{
					timeRange: [5, 11],
					speakerZh: "分词算法 BPE",
					speakerEn: "BPE Tokenizer",
					tagZh: "BPE 子词合并",
					tagEn: "BPE Subword Merging",
					textZh:
						"BPE（字节对编码）基于词频统计迭代合并子词（Sennrich et al., 2016），切分出离散词元序列。目标主角词元【思】（ID=28419）位于序列第 4 位。",
					textEn:
						"The Byte-Pair Encoding (BPE) algorithm iteratively merges frequent subwords (Sennrich et al., 2016), segmenting text into tokens. Target token [思] (ID=28419) occupies index 3.",
				},
				{
					timeRange: [11, 18],
					speakerZh: "词嵌入与位置编码",
					speakerEn: "Embedding & Positional Encoding",
					tagZh: "X = E + P 嵌入加和",
					tagEn: "X = E + P Synthesis",
					textZh:
						"查表取出语义嵌入向量 E[3]，并叠加标准正弦余弦位置编码 P[3]（Vaswani et al., 2017），为无序的自注意力运算注入确定性时序位置信息：X = E + P。",
					textEn:
						"The model looks up embedding vector E[3] and adds sinusoidal positional encoding P[3] (Vaswani et al., 2017), equipping self-attention with explicit sequence order: X = E + P.",
				},
			],
		},
		{
			id: "act-2-multi-head-projection",
			actNumber: "ACT II",
			titleZh: "多头分立与 QKV 线性投影",
			titleEn: "Multi-Head Attention: Q, K, V Linear Projections",
			subtitleZh:
				"克服单头表征容量限制：模型分立 Head 1 (句法关注头) 与 Head 2 (语义关注头) 多重子空间 (Clark et al., 2019)",
			subtitleEn:
				"Overcoming single-head capacity limits: Head 1 (syntax) and Head 2 (semantics) project into specialized subspaces (Clark et al., 2019)",
			startSec: 18,
			durationSec: 20,
			subtitles: [
				{
					timeRange: [18, 24],
					speakerZh: "线性投影变换",
					speakerEn: "Linear Projections",
					tagZh: "矩阵乘法 X · W",
					tagEn: "Matrix Multiplication X · W",
					textZh:
						"静态词嵌入缺乏上下文交互能力。输入向量 X 与投影权重矩阵 W_Q, W_K, W_V 相乘，线性映射至注意力子空间。",
					textEn:
						"Static embeddings lack context-dependent interaction. Multiplying input X by learned projection matrices W_Q, W_K, and W_V maps token representations into attention subspaces.",
				},
				{
					timeRange: [24, 31],
					speakerZh: "多头注意力分立",
					speakerEn: "Multi-Head Specialization",
					tagZh: "Head 1 句法 vs Head 2 语义",
					tagEn: "Head 1 Syntax vs Head 2 Semantics",
					textZh:
						"单一注意力头受限于单一表征子空间。多头机制使模型能并行关注不同维度（Clark et al., 2019）：Head 1 侧重句法依存，Head 2 侧重主语语义实体关联。",
					textEn:
						"A single attention head is limited to one representation subspace. Multi-head attention enables parallel specialized focus (Clark et al., 2019): syntax tracking in Head 1 and semantic entity alignment in Head 2.",
				},
				{
					timeRange: [31, 38],
					speakerZh: "QKV 三重表征",
					speakerEn: "Q, K, V Representations",
					tagZh: "寻址 Q · 键名 K · 载荷 V",
					tagEn: "Query Q · Key K · Value V",
					textZh:
						"Query 代表当前词元的寻址探针，Key 充当匹配标识索引，Value 则承载将被注意力加权抽取的真实语义信息。",
					textEn:
						"Query acts as the addressing probe for the current token, Key provides the matching feature index, and Value carries the semantic payload to be weighted and aggregated.",
				},
			],
		},
		{
			id: "act-3-attention-and-mask",
			actNumber: "ACT III",
			titleZh: "缩放点积防饱和与 Value 向量加权聚合",
			titleEn: "Scaled Dot-Product, Causal Mask & Value Aggregation",
			subtitleZh:
				"除以 √d_k 稳定点积方差（Vaswani et al., 2017），因果掩码消除未来泄漏，注意力权重加权聚合 Value 生成上下文表示 Z",
			subtitleEn:
				"Dividing by √d_k stabilizes variance (Vaswani et al., 2017); causal mask prevents future leakage; weights aggregate Values into Z",
			startSec: 38,
			durationSec: 20,
			subtitles: [
				{
					timeRange: [38, 44],
					speakerZh: "缩放点积打分",
					speakerEn: "Scaled Dot-Product Attention",
					tagZh: "除以 √d_k 抑噪",
					tagEn: "Variance Scaling / √d_k",
					textZh:
						"主角 Query 与前驱各 Key 计算点积相似度。除以缩放因子 √d_k = 1.41（Vaswani et al., 2017）使内积方差维持在 1，防止进入 Softmax 梯度饱和区。",
					textEn:
						"The Query computes dot-product scores with predecessor Keys. Dividing by √d_k = 1.41 (Vaswani et al., 2017) maintains unit variance, preventing Softmax gradient saturation.",
				},
				{
					timeRange: [44, 51],
					speakerZh: "自回归因果掩码",
					speakerEn: "Autoregressive Causal Mask",
					tagZh: "因果掩码屏蔽未来 (-∞)",
					tagEn: "Causal Mask (-∞ -> 0.00)",
					textZh:
						"自回归生成要求单向因果流，未来位置词元【吗】被因果掩码阻断：分数置为 -∞，经 Softmax 指数运算后权重严格为 0.00，消除未来信息泄漏。",
					textEn:
						"In autoregressive generation, future token [吗] must not leak information. Masking its score to -∞ forces its Softmax weight strictly to 0.00, preserving causal directionality.",
				},
				{
					timeRange: [51, 58],
					speakerZh: "Value 向量加权聚合",
					speakerEn: "Value Vector Aggregation",
					tagZh: "Z = ∑ α_i V_i 上下文新生",
					tagEn: "Z = ∑ α_i V_i Synthesis",
					textZh:
						"注意力权重决定表征的线性组合：Head 1 以 50.7% 权重聚焦【能够】，Head 2 以 49.4% 权重聚焦【智能】，加权汇聚 Value 向量生成融合语境的新向量 Z。",
					textEn:
						"Attention weights dictate linear combination: Head 1 assigns 50.7% weight to [能够], while Head 2 assigns 49.4% to [智能], aggregating Value vectors into updated context vector Z.",
				},
			],
		},
		{
			id: "act-4-temperature-bifurcation",
			actNumber: "ACT IV",
			titleZh: "MLP 键值记忆检索与 Softmax 温度分岔 (交互驻留)",
			titleEn: "MLP Key-Value Memory & Interactive Softmax Bifurcation",
			subtitleZh:
				"前馈层充当键值记忆检索知识（Geva et al., 2021），Softmax 交互驻留：调节温度 T 观察概率分布相变",
			subtitleEn:
				"FFN operates as key-value memory (Geva et al., 2021); interactive pause allows tuning temperature T to observe distribution phase shifts",
			startSec: 58,
			durationSec: 20,
			checkpoint: {
				id: "temperature-checkpoint",
				pauseAtSec: 8,
				promptZh: "⏸ 动画已自动驻留：请调节 Softmax 温度系数 (T)",
				promptEn: "⏸ Cinema Paused: Adjust Softmax Temperature (T)",
				guidanceZh:
					"拖动温度滑块观察相变：极低温度 (T=0.2) 导致信息熵塌缩为 0 bit 并走向贪婪 Argmax 独选；极高温度 (T=2.0) 抹平语义落差逼近高熵均匀分布；T=0.7 兼顾确定性与生成多样性。",
				guidanceEn:
					"Drag the temperature slider: low temperature (T=0.2) forces entropy collapse to 0 bits and greedy Argmax; high temperature (T=2.0) flattens logits toward uniform noise; T=0.7 balances confidence and diversity.",
				params: [
					{
						id: "temperature",
						labelZh: "采样温度 (Temperature T)",
						labelEn: "Sampling Temperature (T)",
						min: 0.1,
						max: 2.2,
						step: 0.05,
						defaultVal: 0.7,
					},
					{
						id: "topP",
						labelZh: "核采样截断门限 (Top-P)",
						labelEn: "Nucleus Cutoff (Top-P)",
						min: 0.2,
						max: 1.0,
						step: 0.05,
						defaultVal: 0.95,
					},
				],
			},
			subtitles: [
				{
					timeRange: [58, 65],
					speakerZh: "前馈神经网络 FFN",
					speakerEn: "Feed-Forward Network (FFN)",
					tagZh: "4 ➜ 8 ➜ 4 事实神经元激活",
					tagEn: "Fact Neurons Firing",
					textZh:
						"自注意力汇聚语境关联，前馈网络（FFN）则被证明扮演键值记忆库角色（Geva et al., 2021）。隐层 8 个神经元通过 GELU 激活特定概念知识并经投影层写回。",
					textEn:
						"While attention aggregates contextual dependencies, FFN layers function as key-value memories (Geva et al., 2021). Eight hidden neurons activate specific concepts via GELU before linear read-out.",
				},
				{
					timeRange: [65, 72],
					speakerZh: "Softmax 温度调节",
					speakerEn: "Softmax Temperature Scaling",
					tagZh: "⏸ 交互驻留 · 请调温度 T",
					tagEn: "⏸ Pause: Tune Temperature T",
					textZh:
						"【播放自动驻留】请读者交互调节采样温度 T：低温度 (T → 0.2) 压制分布熵并趋向 Argmax 贪心选择；高温度 (T → 2.0) 趋向均匀分布增加生成随机性。",
					textEn:
						"[Cinema Paused] Interactive control active: adjust temperature T. Low T (T → 0.2) minimizes entropy toward greedy Argmax selection; high T (T → 2.0) flattens logits toward uniform noise.",
				},
				{
					timeRange: [72, 78],
					speakerZh: "Top-p 核采样截断",
					speakerEn: "Top-p Nucleus Truncation",
					tagZh: "核采样截断 Top-p",
					tagEn: "Top-p Nucleus Truncation",
					textZh:
						"核采样（Top-p = 0.95，Holtzman et al., 2020）动态保留累积概率前 95% 的最小候选子集，截除不可靠的低概率长尾，并将剩余候选概率重新归一化。",
					textEn:
						"Nucleus sampling (Top-p = 0.95, Holtzman et al., 2020) preserves the smallest subset exceeding 95% cumulative probability, pruning the low-probability tail and renormalizing.",
				},
			],
		},
		{
			id: "act-5-autoregressive-zoomout",
			actNumber: "ACT V",
			titleZh: "随机采样胜出与 KV Cache 增量推断",
			titleEn: "Stochastic Sampling & KV Cache Autoregressive Inference",
			subtitleZh:
				"胜出词元【考】确定性结晶，新词追加至 KV Cache 实现 O(1) 增量前向推理，驱动自回归连续生成",
			subtitleEn:
				"Token [考] crystallizes; appended to KV Cache for O(1) incremental forward pass, driving continuous autoregressive rollout",
			startSec: 78,
			durationSec: 12,
			subtitles: [
				{
					timeRange: [78, 84],
					speakerZh: "随机采样输出",
					speakerEn: "Stochastic Sampling",
					tagZh: "新词结晶：【考】",
					tagEn: "Token [考] Sampled",
					textZh:
						"多项式采样锁定胜出词元：【考】以 90.3% 的基础概率（Top-p 归一化后 92.9%）胜出，成功完成当前子词预测，组成序列：“人工智能能够思考”！",
					textEn:
						"Categorical sampling selects the winning token: [考] emerges with 90.3% base probability (92.9% after Top-p renormalization), yielding the sequence: '人工智能能够思考'!",
				},
				{
					timeRange: [84, 90],
					speakerZh: "KV Cache 增量复用",
					speakerEn: "KV Cache Incremental Inference",
					tagZh: "历史常驻 · O(1) 增量前向",
					tagEn: "O(1) Memory Reuse",
					textZh:
						"为避免重复计算，所有前驱词元的 Key 与 Value 缓存在 KV Cache 中。自回归步只需计算新词【考】的一组向量并追加缓存，以 O(1) 增量复杂度持续生成。",
					textEn:
						"To eliminate redundant computation, predecessor Keys and Values remain in the KV Cache. Each step only computes and appends vectors for new token [考], achieving O(1) incremental generation.",
				},
			],
		},
	],
};
