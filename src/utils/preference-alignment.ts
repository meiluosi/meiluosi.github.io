/**
 * DPO (Direct Preference Optimization) & KL-Regularized Preference Alignment Model.
 *
 * Models the closed-form Gibbs distribution and iterative DPO preference dynamics:
 *   π*(y|x) ∝ π_ref(y|x) · exp(r_proxy(x, y) / β)
 * where r_proxy(x, y) = r_true(x, y) + α_bias · b_hack(y)
 * and implicit DPO reward is:
 *   r_hat(x, y) = β · ln(π_θ(y|x) / π_ref(y|x))
 */

export interface ResponseArchetype {
	id: "calibrated" | "verbose" | "sycophantic" | "degenerate";
	labelZh: string;
	labelEn: string;
	sampleZh: string;
	sampleEn: string;
	refProb: number;
	trueReward: number;
	hackSensitivity: number;
}

export interface AlignedCandidate extends ResponseArchetype {
	policyProb: number;
	proxyReward: number;
	implicitReward: number;
	logRatio: number;
}

export interface FrontierPoint {
	beta: number;
	klDivergence: number;
	expectedTrueReward: number;
	expectedProxyReward: number;
}

export interface AlignmentSnapshot {
	beta: number;
	hackBias: number;
	klDivergence: number;
	expectedTrueReward: number;
	expectedProxyReward: number;
	dpoMarginProb: number;
	overoptimized: boolean;
	candidates: AlignedCandidate[];
	frontier: FrontierPoint[];
}

export const RESPONSE_ARCHETYPES: ResponseArchetype[] = [
	{
		id: "calibrated",
		labelZh: "A · 诚实校准回答 (Calibrated)",
		labelEn: "A · Calibrated & Direct",
		sampleZh:
			"先给结论：该实验样本量不足以证明因果效应（p=0.18），建议补充组内分层对照。",
		sampleEn:
			"Short answer: the sample is too small to establish causality (p=0.18); run a stratified within-group check first.",
		refProb: 0.42,
		trueReward: 0.92,
		hackSensitivity: 0.05,
	},
	{
		id: "verbose",
		labelZh: "B · 冗长格式套路 (Verbose / Styled)",
		labelEn: "B · Verbose & Heavily Formatted",
		sampleZh:
			"这是一个非常深刻的问题！我们可以从第一性原理出发，分五个维度、十二个子要点来全面拆解……",
		sampleEn:
			"That is a profound question! Let us unpack this across five overarching pillars and twelve bulleted dimensions...",
		refProb: 0.35,
		trueReward: 0.58,
		hackSensitivity: 0.68,
	},
	{
		id: "sycophantic",
		labelZh: "C · 迎合偏见 / 奖励黑客 (Sycophantic)",
		labelEn: "C · Sycophantic / Reward-Hacked",
		sampleZh:
			"您的直觉完全正确！正因为两组均值差了 15%，无需再做繁琐检验，直接全量推全即可获得最大收益！",
		sampleEn:
			"Your intuition is 100% spot on! Since the raw means differ by 15%, no further control test is needed—ship it immediately!",
		refProb: 0.14,
		trueReward: -0.25,
		hackSensitivity: 1.85,
	},
	{
		id: "degenerate",
		labelZh: "D · 退化重复回答 (Degenerate)",
		labelEn: "D · Degenerate / Repetitive",
		sampleZh: "因果效应就是因果，只要有相关就有因果，因果效应显著……",
		sampleEn:
			"The causal effect is causal because correlation implies the effect is significant...",
		refProb: 0.09,
		trueReward: -0.9,
		hackSensitivity: -0.2,
	},
];

export function sigmoid(x: number): number {
	if (x >= 0) {
		const z = Math.exp(-x);
		return 1 / (1 + z);
	}
	const z = Math.exp(x);
	return z / (1 + z);
}

export function evaluatePolicyAtBeta(
	beta: number,
	hackBias: number,
): {
	candidates: AlignedCandidate[];
	klDivergence: number;
	expectedTrueReward: number;
	expectedProxyReward: number;
} {
	const safeBeta = Math.max(0.02, Math.min(2.0, beta));
	const safeBias = Math.max(0, Math.min(1.5, hackBias));

	const rawScores = RESPONSE_ARCHETYPES.map((item) => {
		const proxyReward = item.trueReward + safeBias * item.hackSensitivity;
		// Log unnormalized probability: ln(π_ref) + proxyReward / β
		const logUnnorm = Math.log(item.refProb) + proxyReward / safeBeta;
		return { item, proxyReward, logUnnorm };
	});

	const maxLog = Math.max(...rawScores.map((s) => s.logUnnorm));
	const exps = rawScores.map((s) => Math.exp(s.logUnnorm - maxLog));
	const sumExp = exps.reduce((a, b) => a + b, 0);

	let klDivergence = 0;
	let expectedTrueReward = 0;
	let expectedProxyReward = 0;

	const candidates: AlignedCandidate[] = rawScores.map((s, idx) => {
		const policyProb = exps[idx] / sumExp;
		const logRatio = Math.log(Math.max(1e-12, policyProb) / s.item.refProb);
		const implicitReward = safeBeta * logRatio;

		if (policyProb > 1e-12) {
			klDivergence += policyProb * logRatio;
		}
		expectedTrueReward += policyProb * s.item.trueReward;
		expectedProxyReward += policyProb * s.proxyReward;

		return {
			...s.item,
			policyProb,
			proxyReward: s.proxyReward,
			implicitReward,
			logRatio,
		};
	});

	return {
		candidates,
		klDivergence: Math.max(0, klDivergence),
		expectedTrueReward,
		expectedProxyReward,
	};
}

export function computeAlignmentSnapshot(
	beta: number,
	hackBias: number,
): AlignmentSnapshot {
	const current = evaluatePolicyAtBeta(beta, hackBias);

	const calibrated =
		current.candidates.find((c) => c.id === "calibrated") ??
		current.candidates[0];
	const sycophantic =
		current.candidates.find((c) => c.id === "sycophantic") ??
		current.candidates[0];

	// Bradley-Terry preference probability σ(r_hat(calibrated) - r_hat(sycophantic))
	const dpoMarginProb = sigmoid(
		calibrated.implicitReward - sycophantic.implicitReward,
	);

	const betas = [
		1.2, 0.8, 0.55, 0.4, 0.3, 0.22, 0.16, 0.12, 0.09, 0.065, 0.045, 0.03, 0.02,
	];
	const frontier: FrontierPoint[] = betas.map((b) => {
		const res = evaluatePolicyAtBeta(b, hackBias);
		return {
			beta: b,
			klDivergence: res.klDivergence,
			expectedTrueReward: res.expectedTrueReward,
			expectedProxyReward: res.expectedProxyReward,
		};
	});

	// Detect Goodhart overoptimization: when lowering beta further drops true reward relative to peak
	const peakTrue = Math.max(...frontier.map((p) => p.expectedTrueReward));
	const overoptimized =
		hackBias > 0.25 &&
		current.expectedTrueReward < peakTrue - 0.05 &&
		sycophantic.policyProb > calibrated.policyProb;

	return {
		beta,
		hackBias,
		klDivergence: current.klDivergence,
		expectedTrueReward: current.expectedTrueReward,
		expectedProxyReward: current.expectedProxyReward,
		dpoMarginProb,
		overoptimized,
		candidates: current.candidates,
		frontier,
	};
}
