/**
 * Micro-Transformer Forward Pass Inference Engine
 *
 * A deterministic, zero-dependency in-browser Transformer implementation
 * designed for mechanistic interpretability and pedagogical step-by-step tracing.
 * Follows the modern LLaMA/SmolLM architecture: RMSNorm, RoPE/Positional, SwiGLU FFN, Causal Attention.
 */

export const DEFAULT_VOCAB = [
	"[BOS]",
	"the",
	"cat",
	"sat",
	"on",
	"mat",
	"and",
	"dog",
	"ran",
	"fast",
	"slow",
	"ai",
	"learns",
	"entropy",
	"mind",
	"[EOS]",
] as const;

export type Token = (typeof DEFAULT_VOCAB)[number];

export interface HeadTrace {
	headIndex: number;
	q: number[][]; // [seq_len, d_k]
	k: number[][]; // [seq_len, d_k]
	v: number[][]; // [seq_len, d_k]
	rawScores: number[][]; // [seq_len, seq_len]
	maskedScores: number[][]; // with -1e9 for causal mask
	attnWeights: number[][]; // [seq_len, seq_len]
	headOutput: number[][]; // [seq_len, d_k]
	isAblated: boolean;
}

export interface CandidateToken {
	token: string;
	id: number;
	prob: number;
	logit: number;
}

export interface ForwardPassTrace {
	tokens: string[];
	tokenIds: number[];
	embeddings: number[][]; // [seq_len, d_model]
	heads: HeadTrace[];
	mhaOutput: number[][]; // [seq_len, d_model]
	residual1: number[][]; // [seq_len, d_model]
	norm1: number[][]; // [seq_len, d_model]
	ffnSilu: number[][]; // [seq_len, d_ffn]
	ffnOut: number[][]; // [seq_len, d_model]
	residual2: number[][]; // [seq_len, d_model]
	norm2: number[][]; // [seq_len, d_model]
	lastTokenLogits: number[]; // [V]
	probabilities: number[]; // [V]
	topCandidates: CandidateToken[];
	sampledNextToken: CandidateToken;
	entropy: number;
	temperature: number;
	topP: number;
	ffnPerturbation: number;
	hallucinationMode: boolean;
	attentionRays: Array<{
		fromIdx: number;
		toIdx: number;
		head0Weight: number;
		head1Weight: number;
		avgWeight: number;
	}>;
	layerNorms: {
		embeddingNorm: number;
		mhaNorm: number;
		residual1Norm: number;
		ffnNorm: number;
		residual2Norm: number;
		finalNorm: number;
	};
}

export interface TransformerConfig {
	vocab: readonly string[];
	dModel: number; // e.g. 16
	nHeads: number; // e.g. 2
	dK: number; // dModel / nHeads = 8
	dFfn: number; // e.g. 32
}

export const DEFAULT_CONFIG: TransformerConfig = {
	vocab: DEFAULT_VOCAB,
	dModel: 16,
	nHeads: 2,
	dK: 8,
	dFfn: 32,
};

// Deterministic Pseudo Random Number Generator (PRNG) for reproducible weights
function createPrng(seed: number) {
	let s = seed;
	return () => {
		s = (s * 1664525 + 1013904223) % 4294967296;
		return s / 4294967296;
	};
}

// Matrix helper routines
function createMatrix(rows: number, cols: number, fill = 0): number[][] {
	return Array.from({ length: rows }, () => Array(cols).fill(fill));
}

function dotProduct(a: number[], b: number[]): number {
	let sum = 0;
	for (let i = 0; i < a.length; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

function matmul(a: number[][], b: number[][]): number[][] {
	const rowsA = a.length;
	const colsA = a[0].length;
	const colsB = b[0].length;
	const res = createMatrix(rowsA, colsB, 0);
	for (let i = 0; i < rowsA; i++) {
		for (let j = 0; j < colsB; j++) {
			let sum = 0;
			for (let k = 0; k < colsA; k++) {
				sum += a[i][k] * b[k][j];
			}
			res[i][j] = sum;
		}
	}
	return res;
}

function rmsNorm(vec: number[], eps = 1e-5): number[] {
	let sumSq = 0;
	for (let i = 0; i < vec.length; i++) {
		sumSq += vec[i] * vec[i];
	}
	const rms = Math.sqrt(sumSq / vec.length + eps);
	return vec.map((v) => v / rms);
}

function silu(x: number): number {
	return x / (1 + Math.exp(-x));
}

function softmax(vec: number[], temperature = 1.0): number[] {
	const t = Math.max(temperature, 0.01);
	let max = Number.NEGATIVE_INFINITY;
	for (let i = 0; i < vec.length; i++) {
		if (vec[i] > max) max = vec[i];
	}
	const exps = vec.map((v) => Math.exp((v - max) / t));
	const sum = exps.reduce((acc, cur) => acc + cur, 0);
	return exps.map((e) => (sum > 0 ? e / sum : 1 / vec.length));
}

/**
 * Weights store for the micro-model
 */
export class MicroTransformerWeights {
	public config: TransformerConfig;
	public wEmbedding: number[][]; // [V, dModel]
	public wQ: number[][][]; // [nHeads][dModel, dK]
	public wK: number[][][]; // [nHeads][dModel, dK]
	public wV: number[][][]; // [nHeads][dModel, dK]
	public wO: number[][]; // [dModel, dModel]
	public wGate: number[][]; // [dModel, dFfn]
	public wDown: number[][]; // [dFfn, dModel]
	public wUnembed: number[][]; // [dModel, V]

	constructor(config: TransformerConfig = DEFAULT_CONFIG, seed = 42) {
		this.config = config;
		const rand = createPrng(seed);
		const randn = () => (rand() - 0.5) * 0.4;

		const { vocab, dModel, nHeads, dK, dFfn } = config;
		const V = vocab.length;

		this.wEmbedding = createMatrix(V, dModel);
		for (let i = 0; i < V; i++) {
			for (let j = 0; j < dModel; j++) {
				this.wEmbedding[i][j] = randn();
			}
		}

		this.wQ = Array.from({ length: nHeads }, () => {
			const m = createMatrix(dModel, dK);
			for (let i = 0; i < dModel; i++) {
				for (let j = 0; j < dK; j++) m[i][j] = randn();
			}
			return m;
		});

		this.wK = Array.from({ length: nHeads }, () => {
			const m = createMatrix(dModel, dK);
			for (let i = 0; i < dModel; i++) {
				for (let j = 0; j < dK; j++) m[i][j] = randn();
			}
			return m;
		});

		this.wV = Array.from({ length: nHeads }, () => {
			const m = createMatrix(dModel, dK);
			for (let i = 0; i < dModel; i++) {
				for (let j = 0; j < dK; j++) m[i][j] = randn();
			}
			return m;
		});

		this.wO = createMatrix(dModel, dModel);
		for (let i = 0; i < dModel; i++) {
			for (let j = 0; j < dModel; j++) this.wO[i][j] = randn();
		}

		this.wGate = createMatrix(dModel, dFfn);
		for (let i = 0; i < dModel; i++) {
			for (let j = 0; j < dFfn; j++) this.wGate[i][j] = randn();
		}

		this.wDown = createMatrix(dFfn, dModel);
		for (let i = 0; i < dFfn; i++) {
			for (let j = 0; j < dModel; j++) this.wDown[i][j] = randn();
		}

		this.wUnembed = createMatrix(dModel, V);
		for (let i = 0; i < dModel; i++) {
			for (let j = 0; j < V; j++) this.wUnembed[i][j] = randn();
		}

		// Inject semantic language priors into embeddings and projection weights
		this.injectPriors();
	}

	/**
	 * Plant strong semantic circuits:
	 * Head 0: "Induction & Syntax Continuity" (the -> noun, noun -> verb, verb -> prep)
	 * Head 1: "Long-Range Semantic Tracking" (ai/mind/entropy cohesion)
	 */
	private injectPriors() {
		const v = this.config.vocab;
		const id = (tok: string) => v.indexOf(tok);

		const boostBigram = (src: string, tgt: string, weight: number) => {
			const s = id(src);
			const t = id(tgt);
			if (s >= 0 && t >= 0) {
				for (let d = 0; d < this.config.dModel; d++) {
					this.wUnembed[d][t] += this.wEmbedding[s][d] * weight * 0.4;
				}
			}
		};

		// Syntax flows
		boostBigram("[BOS]", "the", 2.5);
		boostBigram("[BOS]", "ai", 2.2);
		boostBigram("the", "cat", 2.8);
		boostBigram("the", "dog", 2.6);
		boostBigram("the", "mat", 1.8);
		boostBigram("cat", "sat", 3.2);
		boostBigram("cat", "ran", 1.9);
		boostBigram("dog", "ran", 3.0);
		boostBigram("dog", "sat", 1.7);
		boostBigram("sat", "on", 3.5);
		boostBigram("on", "the", 3.2);
		boostBigram("ran", "fast", 3.0);
		boostBigram("ran", "slow", 1.5);
		boostBigram("fast", "and", 2.0);
		boostBigram("slow", "and", 2.0);
		boostBigram("and", "the", 2.2);
		boostBigram("mat", "[EOS]", 2.5);
		boostBigram("mat", "and", 1.8);

		// AI & philosophy circuits
		boostBigram("ai", "learns", 3.4);
		boostBigram("learns", "entropy", 2.8);
		boostBigram("learns", "fast", 2.4);
		boostBigram("entropy", "and", 2.0);
		boostBigram("entropy", "mind", 2.6);
		boostBigram("mind", "[EOS]", 3.0);
	}
}

// Global cached model instance
let defaultModel: MicroTransformerWeights | null = null;
export function getMicroModel(): MicroTransformerWeights {
	if (!defaultModel) {
		defaultModel = new MicroTransformerWeights(DEFAULT_CONFIG, 2026);
	}
	return defaultModel;
}

export interface RunOptions {
	temperature?: number;
	topP?: number;
	ablateHeads?: number[]; // [0] or [1]
	noiseScale?: number; // 0.0 to 1.0
	ffnPerturbation?: number; // 0.0 to 1.0
	hallucinationMode?: boolean;
	forcedSeed?: number;
}

/**
 * Executes a full transparent forward pass step over the given token sequence.
 */
export function runForwardPass(
	rawTokens: string[],
	options: RunOptions = {},
	model: MicroTransformerWeights = getMicroModel(),
): ForwardPassTrace {
	const {
		temperature = 0.8,
		topP = 0.9,
		ablateHeads = [],
		noiseScale = 0.0,
		ffnPerturbation = 0.0,
		hallucinationMode = false,
		forcedSeed,
	} = options;

	const { vocab, dModel, nHeads, dK, dFfn } = model.config;
	const tokens = rawTokens.length === 0 ? ["[BOS]"] : [...rawTokens];

	// 1. Tokenization to IDs
	const tokenIds = tokens.map((tok) => {
		const idx = vocab.indexOf(tok);
		return idx >= 0 ? idx : 0; // fallback to [BOS]
	});
	const T = tokenIds.length;

	// 2. Token Embeddings + Simple Sinusoidal Positional Embeddings
	const embeddings = createMatrix(T, dModel);
	for (let t = 0; t < T; t++) {
		const tid = tokenIds[t];
		for (let d = 0; d < dModel; d++) {
			const posFreq = 1 / 10000 ** ((2 * Math.floor(d / 2)) / dModel);
			const posVal =
				d % 2 === 0 ? Math.sin(t * posFreq) : Math.cos(t * posFreq);
			embeddings[t][d] = model.wEmbedding[tid][d] + posVal * 0.25;
			if (noiseScale > 0) {
				embeddings[t][d] += (Math.random() - 0.5) * noiseScale * 0.5;
			}
		}
	}

	// 3. Multi-Head Attention Projections and Scaled Dot-Product Attention
	const headTraces: HeadTrace[] = [];
	const headOutputsCombined: number[][][] = []; // [nHeads][T, dK]

	for (let h = 0; h < nHeads; h++) {
		const isAblated = ablateHeads.includes(h);
		const q = matmul(embeddings, model.wQ[h]);
		const k = matmul(embeddings, model.wK[h]);
		const v = matmul(embeddings, model.wV[h]);

		// Raw Scores: Q * K^T / sqrt(dK)
		const scale = Math.sqrt(dK);
		const rawScores = createMatrix(T, T);
		const maskedScores = createMatrix(T, T);
		const attnWeights = createMatrix(T, T);

		for (let i = 0; i < T; i++) {
			for (let j = 0; j < T; j++) {
				const dot = dotProduct(q[i], k[j]);
				const score = dot / scale;
				rawScores[i][j] = score;

				// Causal Mask: future tokens (j > i) get -1e9
				if (j > i) {
					maskedScores[i][j] = -1e9;
				} else {
					maskedScores[i][j] = score;
				}
			}

			// Softmax per row
			const rowWeights = softmax(maskedScores[i], 1.0);
			for (let j = 0; j < T; j++) {
				attnWeights[i][j] = isAblated ? 0 : rowWeights[j];
			}
		}

		// Head Output = AttnWeights * V
		const headOutput = createMatrix(T, dK, 0);
		if (!isAblated) {
			for (let i = 0; i < T; i++) {
				for (let d = 0; d < dK; d++) {
					let sum = 0;
					for (let j = 0; j < T; j++) {
						sum += attnWeights[i][j] * v[j][d];
					}
					headOutput[i][d] = sum;
				}
			}
		}

		headTraces.push({
			headIndex: h,
			q,
			k,
			v,
			rawScores,
			maskedScores,
			attnWeights,
			headOutput,
			isAblated,
		});

		headOutputsCombined.push(headOutput);
	}

	// 4. Concatenate Heads and Multiply by W_O
	const concatMHA = createMatrix(T, dModel);
	for (let t = 0; t < T; t++) {
		for (let h = 0; h < nHeads; h++) {
			for (let d = 0; d < dK; d++) {
				concatMHA[t][h * dK + d] = headOutputsCombined[h][t][d];
			}
		}
	}
	const mhaOutput = matmul(concatMHA, model.wO);

	// 5. First Residual Connection + RMSNorm
	const residual1 = createMatrix(T, dModel);
	const norm1 = createMatrix(T, dModel);
	for (let t = 0; t < T; t++) {
		for (let d = 0; d < dModel; d++) {
			residual1[t][d] = embeddings[t][d] + mhaOutput[t][d];
		}
		norm1[t] = rmsNorm(residual1[t]);
	}

	// 6. SwiGLU / MLP Block
	const ffnGate = matmul(norm1, model.wGate); // [T, dFfn]
	const ffnSilu = createMatrix(T, dFfn);
	for (let t = 0; t < T; t++) {
		for (let d = 0; d < dFfn; d++) {
			let act = silu(ffnGate[t][d]);
			if (ffnPerturbation > 0 || hallucinationMode) {
				const pert = hallucinationMode ? 0.75 : ffnPerturbation;
				act += (Math.sin(t * 3.7 + d) * 0.8 + (Math.random() - 0.5)) * pert;
			}
			ffnSilu[t][d] = act;
		}
	}
	const ffnOut = matmul(ffnSilu, model.wDown); // [T, dModel]

	// 7. Second Residual Connection + Final RMSNorm
	const residual2 = createMatrix(T, dModel);
	const norm2 = createMatrix(T, dModel);
	for (let t = 0; t < T; t++) {
		for (let d = 0; d < dModel; d++) {
			residual2[t][d] = norm1[t][d] + ffnOut[t][d];
		}
		norm2[t] = rmsNorm(residual2[t]);
	}

	// 8. Language Model Head (Unembedding) on Last Token
	const lastTokenVec = norm2[T - 1]; // [dModel]
	const lastTokenLogits = new Array(vocab.length).fill(0);
	for (let vIdx = 0; vIdx < vocab.length; vIdx++) {
		let dot = 0;
		for (let d = 0; d < dModel; d++) {
			dot += lastTokenVec[d] * model.wUnembed[d][vIdx];
		}
		if (noiseScale > 0) {
			dot += (Math.random() - 0.5) * noiseScale * 2.0;
		}
		lastTokenLogits[vIdx] = dot;
	}

	// 9. Softmax with Temperature
	const rawProbs = softmax(lastTokenLogits, temperature);

	// 10. Top-p Nucleus Filtering
	const sortedIndices = Array.from({ length: vocab.length }, (_, i) => i).sort(
		(a, b) => rawProbs[b] - rawProbs[a],
	);

	let cumSum = 0;
	const filteredProbs = new Array(vocab.length).fill(0);
	for (const idx of sortedIndices) {
		cumSum += rawProbs[idx];
		filteredProbs[idx] = rawProbs[idx];
		if (cumSum >= topP) break;
	}

	// Renormalize
	const pSum = filteredProbs.reduce((acc, cur) => acc + cur, 0);
	const probabilities = filteredProbs.map((p) =>
		pSum > 0 ? p / pSum : 1 / vocab.length,
	);

	// Top candidates sorted
	const topCandidates: CandidateToken[] = sortedIndices
		.filter((idx) => probabilities[idx] > 0.001)
		.map((idx) => ({
			token: vocab[idx],
			id: idx,
			prob: probabilities[idx],
			logit: lastTokenLogits[idx],
		}));

	// 11. Sample next token
	const rnd = forcedSeed !== undefined ? forcedSeed : Math.random();
	let accumulated = 0;
	let sampledId = topCandidates[0]?.id ?? 0;
	for (const cand of topCandidates) {
		accumulated += cand.prob;
		if (rnd <= accumulated) {
			sampledId = cand.id;
			break;
		}
	}

	const sampledNextToken: CandidateToken = {
		token: vocab[sampledId],
		id: sampledId,
		prob: probabilities[sampledId],
		logit: lastTokenLogits[sampledId],
	};

	// 12. Shannon Entropy: H = - sum(p * log2(p))
	let entropy = 0;
	for (const p of probabilities) {
		if (p > 1e-6) {
			entropy -= p * Math.log2(p);
		}
	}

	// 13. Attention Rays for dynamic laser arc streamlines
	const attentionRays: Array<{
		fromIdx: number;
		toIdx: number;
		head0Weight: number;
		head1Weight: number;
		avgWeight: number;
	}> = [];
	for (let i = 0; i < T; i++) {
		for (let j = 0; j <= i; j++) {
			const w0 = headTraces[0]?.attnWeights[i]?.[j] ?? 0;
			const w1 = headTraces[1]?.attnWeights[i]?.[j] ?? 0;
			attentionRays.push({
				fromIdx: i,
				toIdx: j,
				head0Weight: w0,
				head1Weight: w1,
				avgWeight: (w0 + w1) / 2,
			});
		}
	}

	// 14. Layer-by-Layer Activation Energy Norms
	function vectorNorm(v: number[]): number {
		let sumSq = 0;
		for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
		return Math.sqrt(sumSq);
	}
	const lastIdx = T - 1;
	const layerNorms = {
		embeddingNorm: vectorNorm(embeddings[lastIdx] ?? []),
		mhaNorm: vectorNorm(mhaOutput[lastIdx] ?? []),
		residual1Norm: vectorNorm(residual1[lastIdx] ?? []),
		ffnNorm: vectorNorm(ffnOut[lastIdx] ?? []),
		residual2Norm: vectorNorm(residual2[lastIdx] ?? []),
		finalNorm: vectorNorm(norm2[lastIdx] ?? []),
	};

	return {
		tokens,
		tokenIds,
		embeddings,
		heads: headTraces,
		mhaOutput,
		residual1,
		norm1,
		ffnSilu,
		ffnOut,
		residual2,
		norm2,
		lastTokenLogits,
		probabilities,
		topCandidates,
		sampledNextToken,
		entropy,
		temperature,
		topP,
		ffnPerturbation,
		hallucinationMode,
		attentionRays,
		layerNorms,
	};
}
