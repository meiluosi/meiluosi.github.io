/**
 * Mechanistic Interpretability: Attention Lens & Induction Head Circuits.
 *
 * Implements canonical attention patterns analyzed in Anthropic's Transformer Circuits:
 * - Induction Head: recognizes [A][B] ... [A] -> attends to [B] (in-context copying)
 * - Previous-Token Head: attends to i - 1 (bigram syntax)
 * - Duplicate-Token Head: attends to past identical tokens t_j = t_i
 * - Local-Decay Head: local sliding-window attention
 * - Attention Sink / BOS Head: dumps unneeded attention to token 0
 */

export type AttentionHeadType =
	| "induction"
	| "previous"
	| "duplicate"
	| "local"
	| "sink";

export interface AttentionHeadMeta {
	id: AttentionHeadType;
	layer: number;
	headIndex: number;
	nameZh: string;
	nameEn: string;
	roleZh: string;
	roleEn: string;
}

export interface AttentionMatrixResult {
	tokens: string[];
	matrix: number[][]; // N x N, each row sums to 1.0 (causal upper triangle = 0)
	entropies: number[]; // entropy per query position
	headMeta: AttentionHeadMeta;
	samplePromptKey: string;
}

export const CANONICAL_HEADS: AttentionHeadMeta[] = [
	{
		id: "induction",
		layer: 2,
		headIndex: 4,
		nameZh: "L2H4 · 归纳头 (Induction Head)",
		nameEn: "L2H4 · Induction Head",
		roleZh:
			"识别上下文中的前缀重复：看到 [A] 便精准注意先前 [A] 后面的 [B]（上下文学习核心机制）",
		roleEn:
			"Recognizes in-context prefix repetition: seeing [A] attends to the [B] that followed it earlier",
	},
	{
		id: "previous",
		layer: 1,
		headIndex: 1,
		nameZh: "L1H1 · 前驱词头 (Previous-Token Head)",
		nameEn: "L1H1 · Previous-Token Head",
		roleZh: "严格关注前一个位置 i-1，负责局部二元语法与词组搭配绑定",
		roleEn:
			"Attends strictly to the preceding token i-1, binding local bigram syntax",
	},
	{
		id: "duplicate",
		layer: 1,
		headIndex: 6,
		nameZh: "L1H6 · 重复词头 (Duplicate-Token Head)",
		nameEn: "L1H6 · Duplicate-Token Head",
		roleZh: "回溯并高亮文本中已出现过的完全同名实体或符号",
		roleEn:
			"Looks back to attend to identical earlier occurrences of the current token",
	},
	{
		id: "local",
		layer: 1,
		headIndex: 3,
		nameZh: "L1H3 · 局部滑动衰减头 (Local Window Head)",
		nameEn: "L1H3 · Local Window Head",
		roleZh: "注意力集中在当前词附近 1~3 个位置，呈高斯衰减",
		roleEn:
			"Concentrates attention on immediately neighboring tokens with distance decay",
	},
	{
		id: "sink",
		layer: 3,
		headIndex: 0,
		nameZh: "L3H0 · 注意力水槽/BOS头 (Attention Sink Head)",
		nameEn: "L3H0 · Attention Sink Head",
		roleZh:
			"当本轮无显式特征需要提取时，将大部分注意力归约汇聚到起始首词 [BOS]",
		roleEn: "Dumps excess unneeded attention mass onto the initial [BOS] token",
	},
];

export const SAMPLE_PROMPTS: { key: string; text: string; label: string }[] = [
	{
		key: "induction-test",
		text: "Alice visited Paris . Later Alice visited Rome",
		label: "Induction Copy: Alice visited [Paris] ... Alice visited -> [Rome]",
	},
	{
		key: "repeated-code",
		text: "def compute_loss ( pred , target ) : diff = pred - target",
		label: "Code Identifiers: def compute_loss ... pred - target",
	},
	{
		key: "reasoning-steps",
		text: "Because hypothesis A implies B , and B implies C , therefore A implies C",
		label: "Causal Reasoning: A implies B ... therefore A implies C",
	},
];

export function tokenizePrompt(text: string): string[] {
	const cleaned = text.trim();
	if (!cleaned) return ["[BOS]"];
	const words = cleaned.split(/\s+/).filter(Boolean);
	return ["[BOS]", ...words];
}

export function computeAttentionMatrix(
	tokens: string[],
	headType: AttentionHeadType,
): AttentionMatrixResult {
	const N = tokens.length;
	const meta =
		CANONICAL_HEADS.find((h) => h.id === headType) ?? CANONICAL_HEADS[0];
	const matrix: number[][] = [];
	const entropies: number[] = [];

	for (let i = 0; i < N; i++) {
		const row: number[] = new Array(N).fill(0);
		const rawScores: number[] = new Array(i + 1).fill(0);

		if (i === 0) {
			rawScores[0] = 1.0;
		} else if (headType === "induction") {
			// Query token is tokens[i]
			// Look for previous position j < i-1 where tokens[j] === tokens[i-1]
			// If found, attend to j + 1!
			const prevTok = tokens[i - 1].toLowerCase();
			let foundInductionTarget = -1;
			for (let j = 0; j < i - 1; j++) {
				if (tokens[j].toLowerCase() === prevTok) {
					foundInductionTarget = j + 1;
					break;
				}
			}

			for (let k = 0; k <= i; k++) {
				if (k === foundInductionTarget) {
					rawScores[k] = 4.8; // Sharp attention to the token following earlier prefix
				} else if (k === 0) {
					rawScores[k] = 0.4; // Small baseline to BOS
				} else if (k === i - 1) {
					rawScores[k] = 1.2;
				} else {
					rawScores[k] = 0.1;
				}
			}
		} else if (headType === "previous") {
			for (let k = 0; k <= i; k++) {
				rawScores[k] = k === i - 1 ? 4.5 : k === 0 ? 0.3 : 0.05;
			}
		} else if (headType === "duplicate") {
			const cur = tokens[i].toLowerCase();
			for (let k = 0; k <= i; k++) {
				if (k < i && tokens[k].toLowerCase() === cur) {
					rawScores[k] = 4.2;
				} else if (k === 0) {
					rawScores[k] = 0.5;
				} else {
					rawScores[k] = 0.1;
				}
			}
		} else if (headType === "local") {
			for (let k = 0; k <= i; k++) {
				const dist = i - k;
				rawScores[k] = Math.max(0.01, 3.8 * Math.exp(-0.75 * dist));
			}
		} else {
			// Sink
			for (let k = 0; k <= i; k++) {
				rawScores[k] = k === 0 ? 4.6 : 0.15;
			}
		}

		// Softmax over causal mask [0...i]
		const maxScore = Math.max(...rawScores);
		const exps = rawScores.map((s) => Math.exp(s - maxScore));
		const sumExp = exps.reduce((a, b) => a + b, 0);

		let ent = 0;
		for (let k = 0; k <= i; k++) {
			const prob = exps[k] / sumExp;
			row[k] = prob;
			if (prob > 1e-12) {
				ent -= prob * Math.log(prob);
			}
		}

		matrix.push(row);
		entropies.push(ent);
	}

	return {
		tokens,
		matrix,
		entropies,
		headMeta: meta,
		samplePromptKey: "",
	};
}
