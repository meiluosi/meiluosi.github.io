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
