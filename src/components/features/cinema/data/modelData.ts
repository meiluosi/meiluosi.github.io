/**
 * modelData.ts — 一个可完全复算的玩具 Transformer（单层、双头）。
 *
 * 设计原则（数值闭环）：
 * - 本文件只手写「源参数」：词嵌入 E、权重矩阵 W_*、位置编码缩放系数；
 * - 影片里出现的一切派生量（Q/K/V、注意力分数、权重、Z、LayerNorm、FFN 激活、
 *   logits、概率、KV Cache）全部由 tensor.ts 中的纯函数现场计算，禁止手填；
 * - modelData.test.ts 断言整条链路闭合（运行：npx tsx --test <该文件>）。
 *
 * 设定：d_model = 4，头数 h = 2，每头 d_k = d_v = 2，d_ff = 8，候选词表 5 个。
 * 权重为教学构造的玩具值（经确定性搜索挑选，使两头呈现可读的分工），
 * 不代表任何真实模型；「句法头 / 语义头」等称呼是示意性解读。
 */
import {
	add,
	argmax,
	dot,
	entropyBits,
	gaussian,
	gelu,
	layerNorm,
	type Mat,
	matMul,
	mulberry32,
	nucleusKeep,
	round,
	softmax,
	type Vec,
	vecMat,
} from "./tensor";

/* ──────────────── 词元与输入 ──────────────── */

export interface TokenItem {
	text: string;
	pos: number;
	id: number;
	isHero?: boolean;
}

export const TOKENS: TokenItem[] = [
	{ text: "人工", pos: 0, id: 1102 },
	{ text: "智能", pos: 1, id: 2408 },
	{ text: "能够", pos: 2, id: 5931 },
	{ text: "思", pos: 3, id: 28419, isHero: true },
];

/** 主角下标（Query 所在位置）。 */
export const HERO = 3;

/** 序列中下一个真实词（用于演示因果掩码：它在训练时存在，但必须被遮住）。 */
export const FUTURE_TOKEN: TokenItem = { text: "吗", pos: 4, id: 1332 };

/** 采样胜出后被追加的新词。 */
export const NEW_TOKEN: TokenItem = { text: "考", pos: 4, id: 31088 };

/** 词嵌入 E（源参数，4 × 4）。 */
export const EMBED_MATRIX: Mat = [
	[0.8, 0.1, -0.3, 0.5],
	[0.7, 0.4, -0.2, 0.6],
	[-0.2, 0.9, 0.5, 0.1],
	[0.1, 0.8, 0.6, -0.3], // 主角 "思"
];

/** 新词 "考" 的嵌入（源参数）。 */
export const NEW_TOKEN_EMBED: Vec = [0.6, -0.3, 0.4, 0.5];

/** 位置编码缩放系数：原论文幅值为 1，这里缩到 0.1 以免淹没 4 维玩具嵌入。 */
export const PE_SCALE = 0.1;

/**
 * 正弦位置编码（Vaswani et al., 2017）：
 * PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))。
 * d = 4 时两组频率分别为 1 与 1/100。
 */
export const positionalEncoding = (pos: number): Vec =>
	[Math.sin(pos), Math.cos(pos), Math.sin(pos / 100), Math.cos(pos / 100)].map(
		(v) => PE_SCALE * v,
	);

export const POS_MATRIX: Mat = TOKENS.map((t) => positionalEncoding(t.pos));

/** X = E + P（派生）。 */
export const X_MATRIX: Mat = EMBED_MATRIX.map((row, i) =>
	add(row, POS_MATRIX[i]),
);

export const X_NEW: Vec = add(NEW_TOKEN_EMBED, positionalEncoding(4));

/* ──────────────── 多头注意力 ──────────────── */

export const D_K = 2;
export const SQRT_DK: number = Math.sqrt(D_K);

interface HeadWeights {
	WQ: Mat;
	WK: Mat;
	WV: Mat;
}

const HEAD_WEIGHTS: HeadWeights[] = [
	{
		WQ: [
			[-0.9, -0.2],
			[0.1, -1],
			[-0.7, -0.8],
			[0.4, 0.5],
		],
		WK: [
			[0.8, 0.7],
			[-0.6, -1],
			[0.8, -0.9],
			[-0.4, -0.6],
		],
		WV: [
			[0.2, -0.1],
			[0.7, 0.3],
			[-0.7, 0.1],
			[-0.5, 0.2],
		],
	},
	{
		WQ: [
			[0.5, 0.3],
			[-1, -0.9],
			[-0.8, -0.8],
			[0.5, 0],
		],
		WK: [
			[-0.8, -0.9],
			[-0.8, -0.7],
			[0.3, 0.5],
			[-0.6, -0.7],
		],
		WV: [
			[0.7, -0.1],
			[-0.5, 0.8],
			[0.5, -0.4],
			[-0.6, 0],
		],
	},
];

/** 多头拼接后的输出投影 W_O（4 × 4）。 */
export const W_O: Mat = [
	[0.4, 0.2, -1, -0.1],
	[0.7, -0.9, 0.2, -0.9],
	[-0.5, -0.9, -0.6, 0.6],
	[0.1, -0.9, -0.7, 0.7],
];

export interface AttentionHeadData {
	nameZh: string;
	nameEn: string;
	descZh: string;
	descEn: string;
	WQ: Mat;
	WK: Mat;
	WV: Mat;
	/** 所有位置的 Q / K / V（4 × 2）。 */
	Qall: Mat;
	K: Mat;
	V: Mat;
	/** 主角的 Query。 */
	Q: Vec;
	/** 未缩放内积 q·k_j。 */
	rawScores: Vec;
	/** 缩放后 q·k_j / √d_k。 */
	scores: Vec;
	/** softmax 后的注意力权重。 */
	weights: Vec;
	/** 主要关注的位置（argmax）。 */
	focus: number;
	focusTarget: string;
	/** Z = Σ_j α_j V_j。 */
	Z: Vec;
}

/**
 * 通用注意力计算：任意 Query 位置 + 因果掩码。
 * 位置 j > queryIdx 的分数被置为 -Infinity，softmax 后严格为 0。
 */
export const computeAttention = (
	headIdx: number,
	queryIdx: number,
	scaled = true,
): { q: Vec; raw: Vec; scores: Vec; weights: Vec; Z: Vec } => {
	const w = HEAD_WEIGHTS[headIdx];
	const Q = matMul(X_MATRIX, w.WQ);
	const K = matMul(X_MATRIX, w.WK);
	const V = matMul(X_MATRIX, w.WV);
	const raw = K.map((k, j) =>
		j <= queryIdx ? dot(Q[queryIdx], k) : Number.NEGATIVE_INFINITY,
	);
	const scores = raw.map((s) => (scaled ? s / SQRT_DK : s));
	const weights = softmax(scores);
	const Z = V.reduce<Vec>(
		(acc, v, j) => acc.map((a, d) => a + weights[j] * v[d]),
		new Array(D_K).fill(0),
	);
	return { q: Q[queryIdx], raw, scores, weights, Z };
};

const HEAD_META = [
	{
		nameZh: "Head 1 · 句法头（示意）",
		nameEn: "Head 1 · Syntax head (illustrative)",
		descZh: "在本玩具权重下主要看向相邻助动词",
		descEn:
			"With these toy weights it mostly attends to the adjacent modal verb",
	},
	{
		nameZh: "Head 2 · 语义头（示意）",
		nameEn: "Head 2 · Semantic head (illustrative)",
		descZh: "在本玩具权重下主要看向主语实体",
		descEn: "With these toy weights it mostly attends to the subject entity",
	},
];

const buildHead = (h: number): AttentionHeadData => {
	const w = HEAD_WEIGHTS[h];
	const Qall = matMul(X_MATRIX, w.WQ);
	const K = matMul(X_MATRIX, w.WK);
	const V = matMul(X_MATRIX, w.WV);
	const { raw, scores, weights, Z } = computeAttention(h, HERO);
	const focus = argmax(weights);
	return {
		...HEAD_META[h],
		...w,
		Qall,
		K,
		V,
		Q: Qall[HERO],
		rawScores: raw,
		scores,
		weights,
		focus,
		focusTarget: TOKENS[focus].text,
		Z,
	};
};

export const HEAD1: AttentionHeadData = buildHead(0);
export const HEAD2: AttentionHeadData = buildHead(1);
export const HEADS: AttentionHeadData[] = [HEAD1, HEAD2];

export interface DynamicHeadResult {
	q: Vec;
	raw: Vec;
	scores: Vec;
	weights: Vec;
	Z: Vec;
	focus: number;
	focusTarget: string;
}

export interface FullAttentionPipelineResult {
	queryIdx: number;
	h1: DynamicHeadResult;
	h2: DynamicHeadResult;
	zConcat: Vec;
	attnOut: Vec;
	residual1: Vec;
	ln1: Vec;
}

/**
 * 动态主角前向推导：针对任意 queryIdx (0..3) 计算完整的多头注意力及残差汇总。
 */
export const computeFullAttentionPipeline = (
	queryIdx: number,
): FullAttentionPipelineResult => {
	const clampedIdx = Math.max(0, Math.min(TOKENS.length - 1, queryIdx));
	const h1 = computeAttention(0, clampedIdx);
	const h2 = computeAttention(1, clampedIdx);
	const focus1 = argmax(h1.weights);
	const focus2 = argmax(h2.weights);
	const zConcat = [...h1.Z, ...h2.Z];
	const attnOut = vecMat(zConcat, W_O);
	const residual1 = add(X_MATRIX[clampedIdx], attnOut);
	const ln1 = layerNorm(residual1);
	return {
		queryIdx: clampedIdx,
		h1: { ...h1, focus: focus1, focusTarget: TOKENS[focus1].text },
		h2: { ...h2, focus: focus2, focusTarget: TOKENS[focus2].text },
		zConcat,
		attnOut,
		residual1,
		ln1,
	};
};

/** Concat(Z_1, Z_2)（4 维）。 */
export const Z_CONCAT: Vec = [...HEAD1.Z, ...HEAD2.Z];

/** 多头输出 = Concat · W_O。 */
export const ATTN_OUT: Vec = vecMat(Z_CONCAT, W_O);

/** 残差 1：h₁ = x + Attn(x)。 */
export const RESIDUAL_1: Vec = add(X_MATRIX[HERO], ATTN_OUT);

/** LayerNorm(h₁)。 */
export const LN_1: Vec = layerNorm(RESIDUAL_1);

/* ──────────────── 前馈网络 FFN（4 → 8 → 4） ──────────────── */

export const W_1: Mat = [
	[1, 0.8, 0.7, 0, -0.3, -0.2, -0.7, -0.3],
	[-0.3, -0.2, 0.5, 0.7, 0.7, -0.8, -0.7, -0.7],
	[-0.1, 0.9, 1, 0.1, 0.7, 0.5, 0.6, -0.2],
	[-0.6, -0.4, 0.2, -0.3, 0.1, -0.2, 0.8, 0.7],
];

export const W_2: Mat = [
	[0.2, -0.1, -0.5, -0.4],
	[-0.4, 0.2, 0.2, -0.1],
	[0, -0.4, 0.3, -0.2],
	[0.2, 0.5, -0.2, -0.4],
	[0, 0.2, 0.3, 0.1],
	[0.3, -0.1, -0.3, -0.4],
	[-0.4, -0.3, 0.2, -0.4],
	[0.2, 0.5, -0.3, 0.2],
];

export interface FfnActivation {
	id: number;
	labelZh: string;
	labelEn: string;
	act: number;
}

/**
 * 神经元标签纯属示意：真实模型中单个神经元往往是多义的（polysemantic），
 * 「FFN 像键值记忆」的解读参见 Geva et al., 2021。
 */
const NEURON_LABELS: [string, string][] = [
	["认知 / 思维", "Cognition"],
	["图灵测试", "Turing test"],
	["心灵哲学", "Philosophy of mind"],
	["论证 / 推理", "Reasoning"],
	["语气助词", "Modal particle"],
	["硬件", "Hardware"],
	["网络协议", "Networking"],
	["噪声", "Noise"],
];

const FFN_PRE: Vec = vecMat(LN_1, W_1);

export const FFN_ACTIVATIONS: FfnActivation[] = FFN_PRE.map((z, i) => ({
	id: i + 1,
	labelZh: NEURON_LABELS[i][0],
	labelEn: NEURON_LABELS[i][1],
	act: gelu(z),
}));

export const FFN_OUT: Vec = vecMat(
	FFN_ACTIVATIONS.map((n) => n.act),
	W_2,
);

/** 残差 2：h₂ = h₁ + FFN(LN(h₁))。 */
export const RESIDUAL_2: Vec = add(RESIDUAL_1, FFN_OUT);

/** 最终 LayerNorm。 */
export const LN_FINAL: Vec = layerNorm(RESIDUAL_2);

/* ──────────────── 反嵌入 → logits → 采样 ──────────────── */

export interface CandidateToken {
	word: string;
	logit: number;
}

const CANDIDATE_WORDS = ["考", "辨", "想", "索", "维"];

/** 反嵌入矩阵 W_U（4 × 5，只保留 5 个候选词的列）。 */
export const W_U: Mat = [
	[0.7, 0.42, 0.32, 0.06, -0.18],
	[1.01, 0.61, 0.46, 0.09, -0.26],
	[0.23, 0.14, 0.1, 0.02, -0.06],
	[-1.93, -1.18, -0.88, -0.17, 0.5],
];

export const LOGITS: Vec = vecMat(LN_FINAL, W_U);

export const CANDIDATES: CandidateToken[] = CANDIDATE_WORDS.map((word, i) => ({
	word,
	logit: LOGITS[i],
}));

export const DEFAULT_TEMPERATURE = 0.7;
export const DEFAULT_TOP_P = 0.95;

export interface SamplingResult {
	probs: Vec;
	cumProbs: Vec;
	entropy: number;
	keep: number[];
	/** 截断并重新归一化后的分布。 */
	renormalized: Vec;
}

export const sample = (temperature: number, topP: number): SamplingResult => {
	const probs = softmax(LOGITS, temperature);
	let cum = 0;
	const cumProbs = probs.map((p) => {
		cum += p;
		return cum;
	});
	const keep = nucleusKeep(probs, topP);
	const keptMass = keep.reduce((a, i) => a + probs[i], 0);
	const renormalized = probs.map((p, i) =>
		keep.includes(i) ? p / keptMass : 0,
	);
	return { probs, cumProbs, entropy: entropyBits(probs), keep, renormalized };
};

export const DEFAULT_SAMPLING: SamplingResult = sample(
	DEFAULT_TEMPERATURE,
	DEFAULT_TOP_P,
);

/** 默认设置下胜出词的概率（截断重归一化后）。 */
export const WINNER_PROB: number = DEFAULT_SAMPLING.renormalized[0];

/* ──────────────── KV Cache（以 Head 1 为例） ──────────────── */

export interface KvCacheRow {
	pos: number;
	token: string;
	k: Vec;
	v: Vec;
	isNew: boolean;
}

export const KV_CACHE_TABLE: KvCacheRow[] = [
	...TOKENS.map((t, i) => ({
		pos: t.pos,
		token: t.text,
		k: HEAD1.K[i],
		v: HEAD1.V[i],
		isNew: false,
	})),
	{
		pos: NEW_TOKEN.pos,
		token: NEW_TOKEN.text,
		k: vecMat(X_NEW, HEAD1.WK),
		v: vecMat(X_NEW, HEAD1.WV),
		isNew: true,
	},
];

/* ──────────────── √d_k 方差演示 ──────────────── */

/**
 * 若 q、k 各分量独立且服从 N(0,1)，则 q·k 的方差为 d_k；
 * 除以 √d_k 后方差回到 1（Vaswani et al., 2017, §3.2.1 脚注）。
 * 返回确定性采样得到的经验方差，用于可复现的演示。
 */
export const empiricalDotVariance = (
	dk: number,
	scaled: boolean,
	n = 400,
	seed = 11,
): { samples: Vec; variance: number } => {
	const rand = mulberry32(seed + dk);
	const samples: Vec = [];
	for (let s = 0; s < n; s++) {
		let acc = 0;
		for (let d = 0; d < dk; d++) acc += gaussian(rand) * gaussian(rand);
		samples.push(scaled ? acc / Math.sqrt(dk) : acc);
	}
	const mean = samples.reduce((a, b) => a + b, 0) / n;
	const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
	return { samples, variance };
};

/** 展示格式化：向量 -> "[0.12, -0.34]"。 */
export const fmtVec = (v: Vec, digits = 2): string =>
	`[${v.map((x) => round(x, digits).toFixed(digits)).join(", ")}]`;
