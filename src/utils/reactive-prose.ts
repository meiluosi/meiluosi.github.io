/**
 * Reactive Prose & Explorable Explanations mathematical state engine.
 *
 * Models real-time generation dynamics from inline scrubbable parameters:
 *   - Temperature τ and Top-P nucleus truncation
 *   - Softmax distribution, entropy H(p), and perplexity
 *   - KV Cache memory footprint scaling linearly with context length L
 */

export interface CandidateToken {
	token: string;
	tokenZh: string;
	baseLogit: number;
	prob: number;
	cumulativeProb: number;
	inNucleus: boolean;
}

export interface ReactiveProseState {
	temperature: number;
	topP: number;
	contextLength: number;
	layers: number;
	hiddenDim: number;
	entropy: number;
	perplexity: number;
	kvCacheGiB: number;
	activeTokensCount: number;
	tokens: CandidateToken[];
	proseSummaryZh: string;
	proseSummaryEn: string;
}

const CANDIDATE_SEED: { token: string; tokenZh: string; baseLogit: number }[] =
	[
		{ token: "because", tokenZh: "因为", baseLogit: 3.4 },
		{ token: "since", tokenZh: "鉴于", baseLogit: 2.7 },
		{ token: "although", tokenZh: "虽然", baseLogit: 1.8 },
		{ token: "however", tokenZh: "然而", baseLogit: 1.1 },
		{ token: "therefore", tokenZh: "所以", baseLogit: 0.5 },
		{ token: "banana", tokenZh: "香蕉", baseLogit: -1.8 },
	];

export function computeReactiveProse(
	temperature: number,
	topP: number,
	contextLength: number,
	layers = 24,
	hiddenDim = 2048,
): ReactiveProseState {
	const safeTau = Math.max(0.05, Math.min(2.5, temperature));
	const safeTopP = Math.max(0.1, Math.min(1.0, topP));
	const safeContext = Math.max(1, Math.min(131072, Math.floor(contextLength)));

	// 1. Softmax with temperature scaling
	const scaled = CANDIDATE_SEED.map((c) => ({
		...c,
		scaledLogit: c.baseLogit / safeTau,
	}));
	const maxLogit = Math.max(...scaled.map((s) => s.scaledLogit));
	const exps = scaled.map((s) => Math.exp(s.scaledLogit - maxLogit));
	const sumExp = exps.reduce((a, b) => a + b, 0);

	// Sort descending by raw softmax prob
	const sorted = CANDIDATE_SEED.map((c, idx) => ({
		token: c.token,
		tokenZh: c.tokenZh,
		baseLogit: c.baseLogit,
		rawProb: exps[idx] / sumExp,
	})).sort((a, b) => b.rawProb - a.rawProb);

	// 2. Nucleus Top-P filtering
	let cum = 0;
	let nucleusSum = 0;
	const filtered = sorted.map((item) => {
		cum += item.rawProb;
		// Include tokens until cumulative sum exceeds topP (at least 1 token is always kept)
		const inNucleus = cum <= safeTopP || item === sorted[0];
		if (inNucleus) nucleusSum += item.rawProb;
		return {
			...item,
			cumulativeProb: cum,
			inNucleus,
		};
	});

	// 3. Renormalize probabilities within the active nucleus
	let entropy = 0;
	let activeTokensCount = 0;
	const tokens: CandidateToken[] = filtered.map((item) => {
		const normProb = item.inNucleus ? item.rawProb / nucleusSum : 0;
		if (item.inNucleus) {
			activeTokensCount += 1;
			if (normProb > 1e-12) {
				entropy -= normProb * Math.log(normProb);
			}
		}
		return {
			token: item.token,
			tokenZh: item.tokenZh,
			baseLogit: item.baseLogit,
			prob: normProb,
			cumulativeProb: item.cumulativeProb,
			inNucleus: item.inNucleus,
		};
	});

	const perplexity = Math.exp(entropy);

	// 4. KV Cache memory calculation: 2 * layers * hiddenDim * contextLength * 2 bytes (FP16)
	const bytesPerToken = 2 * layers * hiddenDim * 2;
	const totalBytes = bytesPerToken * safeContext;
	const kvCacheGiB = totalBytes / 1024 ** 3;

	const proseSummaryZh =
		safeTau < 0.35
			? `此时分布极度尖锐，首选词「${tokens[0].tokenZh}」概率高达 ${(tokens[0].prob * 100).toFixed(0)}%，模型表现出高度确定性与重复倾向。`
			: safeTau > 1.2
				? `高温促使低概率词元复苏，有效词元池扩大至 ${activeTokensCount} 个，困惑度升至 ${perplexity.toFixed(1)}，输出更具发散创造力。`
				: `分布在确定性与多样性间达成平衡，Top-1 词元概率为 ${(tokens[0].prob * 100).toFixed(0)}%，常用于标准推理与问答。`;

	const proseSummaryEn =
		safeTau < 0.35
			? `The distribution is sharply peaked: top token "${tokens[0].token}" dominates with ${(tokens[0].prob * 100).toFixed(0)}% probability, showing deterministic greed.`
			: safeTau > 1.2
				? `High temperature flattens logits, expanding the active nucleus to ${activeTokensCount} tokens with perplexity ${perplexity.toFixed(1)}, encouraging diverse exploration.`
				: `Balanced regime: Top-1 token probability is ${(tokens[0].prob * 100).toFixed(0)}% with perplexity ${perplexity.toFixed(1)}, typical for grounded generation.`;

	return {
		temperature: safeTau,
		topP: safeTopP,
		contextLength: safeContext,
		layers,
		hiddenDim,
		entropy,
		perplexity,
		kvCacheGiB,
		activeTokensCount,
		tokens,
		proseSummaryZh,
		proseSummaryEn,
	};
}

export interface BpeCostState {
	vocabSize: number;
	textLengthChars: number;
	hiddenDim: number;
	compressionRatio: number; // chars per token
	tokensCount: number;
	embedParamsMillion: number;
	embedMemoryMiB: number;
	proseSummaryZh: string;
	proseSummaryEn: string;
}

export function computeBpeCost(
	vocabSize: number,
	textLengthChars: number,
	hiddenDim = 2048,
): BpeCostState {
	const safeV = Math.max(4000, Math.min(256000, Math.floor(vocabSize)));
	const safeChars = Math.max(
		100,
		Math.min(1000000, Math.floor(textLengthChars)),
	);

	// Compression ratio grows logarithmically with vocabulary size V
	const compressionRatio = Math.max(1.1, 0.45 * Math.log2(safeV / 1000) + 0.85);
	const tokensCount = Math.max(1, Math.round(safeChars / compressionRatio));

	// Embedding parameters: V * hiddenDim
	const embedParamsMillion = (safeV * hiddenDim) / 1e6;
	// FP16: 2 bytes per param
	const embedMemoryMiB = (safeV * hiddenDim * 2) / (1024 * 1024);

	const proseSummaryZh =
		safeV <= 16000
			? `当前为紧凑小词表（${(safeV / 1000).toFixed(0)}K），Embedding 仅占 ${embedMemoryMiB.toFixed(1)} MB 显存，但序列膨胀率高，相同文本需切分为 ${tokensCount} 个 token，显著抬高长上下文二次注意力计算。`
			: safeV >= 96000
				? `当前为超大词表（${(safeV / 1000).toFixed(0)}K），字符压缩比高达 ${compressionRatio.toFixed(2)}，序列收敛至仅 ${tokensCount} 个 token，但 Embedding 显存飙升至 ${embedMemoryMiB.toFixed(1)} MB，稀疏条目需要更多语料才能充分拟合。`
				: `词表规模（${(safeV / 1000).toFixed(0)}K）处于当代平衡区间：压缩比为 ${compressionRatio.toFixed(2)}，${safeChars} 字符转换为约 ${tokensCount} 个 token，Embedding 占用 ${embedMemoryMiB.toFixed(1)} MB 显存。`;

	const proseSummaryEn =
		safeV <= 16000
			? `Compact vocabulary (${(safeV / 1000).toFixed(0)}K): embedding requires only ${embedMemoryMiB.toFixed(1)} MB, but sequence expands to ${tokensCount} tokens, increasing O(L²) quadratic attention cost.`
			: safeV >= 96000
				? `Large vocabulary (${(safeV / 1000).toFixed(0)}K): high compression (${compressionRatio.toFixed(2)} chars/token), yielding only ${tokensCount} tokens, but embedding weights consume ${embedMemoryMiB.toFixed(1)} MB.`
				: `Balanced vocabulary (${(safeV / 1000).toFixed(0)}K): compression ratio is ${compressionRatio.toFixed(2)}, turning ${safeChars} characters into ${tokensCount} tokens at ${embedMemoryMiB.toFixed(1)} MB embedding cost.`;

	return {
		vocabSize: safeV,
		textLengthChars: safeChars,
		hiddenDim,
		compressionRatio: Math.round(compressionRatio * 100) / 100,
		tokensCount,
		embedParamsMillion: Math.round(embedParamsMillion * 10) / 10,
		embedMemoryMiB: Math.round(embedMemoryMiB * 10) / 10,
		proseSummaryZh,
		proseSummaryEn,
	};
}

export interface DpoDynamicsState {
	beta: number;
	chosenMargin: number; // log(pi_theta(yw)/pi_ref(yw))
	rejectMargin: number; // log(pi_theta(yl)/pi_ref(yl))
	implicitRewardMargin: number; // beta * (chosenMargin - rejectMargin)
	loss: number; // -log(sigmoid(implicitRewardMargin))
	gradientWeight: number; // sigmoid(-implicitRewardMargin)
	approxKlDrift: number; // 0.5 * deltaLog^2
	proseSummaryZh: string;
	proseSummaryEn: string;
}

export function computeDpoDynamics(
	beta: number,
	chosenMargin: number,
	rejectMargin: number,
): DpoDynamicsState {
	const safeBeta = Math.max(0.01, Math.min(2.0, beta));
	const deltaLog = chosenMargin - rejectMargin;
	const implicitRewardMargin = safeBeta * deltaLog;

	const sigmoid = (z: number) => {
		if (z >= 0) return 1 / (1 + Math.exp(-z));
		const ez = Math.exp(z);
		return ez / (1 + ez);
	};

	const loss =
		implicitRewardMargin >= 0
			? Math.log(1 + Math.exp(-implicitRewardMargin))
			: -implicitRewardMargin + Math.log(1 + Math.exp(implicitRewardMargin));

	const gradientWeight = sigmoid(-implicitRewardMargin);
	const approxKlDrift = Math.max(0, 0.5 * deltaLog * deltaLog);

	const proseSummaryZh =
		safeBeta < 0.05
			? `当前 β=${safeBeta.toFixed(2)} 极小，约束软化，隐式奖励放大为 ${implicitRewardMargin.toFixed(2)}，策略模型极易偏离参考模型（KL 漂移度估计 ${approxKlDrift.toFixed(1)}），面临古德哈特过优化风险。`
			: safeBeta > 0.6
				? `当前 β=${safeBeta.toFixed(2)} 偏大，对偏离参考施加严厉惩罚，隐式奖励被强行压缩至 ${implicitRewardMargin.toFixed(2)}，损失对偏好对的更新斜率较低（${(gradientWeight * 100).toFixed(0)}%）。`
				: `当前 β=${safeBeta.toFixed(2)} 处于黄金平衡带，在偏好对间产生 ${implicitRewardMargin.toFixed(2)} 的可信奖励差，梯度权重为 ${(gradientWeight * 100).toFixed(0)}%，有效避免崩塌。`;

	const proseSummaryEn =
		safeBeta < 0.05
			? `Beta=${safeBeta.toFixed(2)} is very low: weak KL constraint causes large implicit reward margin (${implicitRewardMargin.toFixed(2)}) and high drift (${approxKlDrift.toFixed(1)}), risking Goodhart reward hacking.`
			: safeBeta > 0.6
				? `Beta=${safeBeta.toFixed(2)} is high: strong reference anchoring restricts implicit reward to ${implicitRewardMargin.toFixed(2)}, resulting in conservative gradient updates (${(gradientWeight * 100).toFixed(0)}%).`
				: `Beta=${safeBeta.toFixed(2)} is in the canonical zone: produces ${implicitRewardMargin.toFixed(2)} reward margin with ${(gradientWeight * 100).toFixed(0)}% gradient sensitivity, preventing policy collapse.`;

	return {
		beta: Math.round(safeBeta * 100) / 100,
		chosenMargin: Math.round(chosenMargin * 100) / 100,
		rejectMargin: Math.round(rejectMargin * 100) / 100,
		implicitRewardMargin: Math.round(implicitRewardMargin * 100) / 100,
		loss: Math.round(loss * 1000) / 1000,
		gradientWeight: Math.round(gradientWeight * 1000) / 1000,
		approxKlDrift: Math.round(approxKlDrift * 100) / 100,
		proseSummaryZh,
		proseSummaryEn,
	};
}
