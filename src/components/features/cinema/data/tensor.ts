/**
 * tensor.ts — 代码电影的极简张量原语（纯函数，零依赖）。
 *
 * 所有影片中出现的数字都必须由这些函数从源权重计算得出，
 * 不允许在场景或数据文件中手填派生量（见 SKILL.md「数值闭环」）。
 */

export type Vec = number[];
export type Mat = number[][];

/** 四舍五入到指定小数位（仅用于展示，计算链路保持全精度）。 */
export const round = (v: number, digits = 2): number => {
	const f = 10 ** digits;
	return Math.round(v * f) / f;
};

export const dot = (a: Vec, b: Vec): number =>
	a.reduce((acc, v, i) => acc + v * b[i], 0);

export const add = (a: Vec, b: Vec): Vec => a.map((v, i) => v + b[i]);

export const scale = (a: Vec, s: number): Vec => a.map((v) => v * s);

/** 行向量 × 矩阵：x (n) · W (n × m) -> (m) */
export const vecMat = (x: Vec, W: Mat): Vec => {
	const m = W[0].length;
	const out: Vec = new Array(m).fill(0);
	for (let i = 0; i < x.length; i++) {
		for (let j = 0; j < m; j++) out[j] += x[i] * W[i][j];
	}
	return out;
};

/** 矩阵 × 矩阵：A (k × n) · W (n × m) -> (k × m) */
export const matMul = (A: Mat, W: Mat): Mat => A.map((row) => vecMat(row, W));

/** 数值稳定的 softmax；-Infinity 项输出严格为 0。 */
export const softmax = (z: Vec, temperature = 1): Vec => {
	const t = Math.max(1e-6, temperature);
	const finite = z.filter((v) => Number.isFinite(v));
	const max = finite.length ? Math.max(...finite) : 0;
	const exps = z.map((v) => (Number.isFinite(v) ? Math.exp((v - max) / t) : 0));
	const sum = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / sum);
};

/** 香农熵（bit）。 */
export const entropyBits = (p: Vec): number =>
	-p.reduce((acc, v) => (v > 1e-12 ? acc + v * Math.log2(v) : acc), 0);

export const argmax = (v: Vec): number =>
	v.reduce((best, x, i, arr) => (x > arr[best] ? i : best), 0);

/** LayerNorm（无可学习仿射参数，eps = 1e-5）。 */
export const layerNorm = (x: Vec, eps = 1e-5): Vec => {
	const mean = x.reduce((a, b) => a + b, 0) / x.length;
	const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length;
	return x.map((v) => (v - mean) / Math.sqrt(variance + eps));
};

/** GELU 的 tanh 近似（Hendrycks & Gimpel, 2016）。 */
export const gelu = (x: number): number =>
	0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

/** 核采样（Top-p）：返回被保留的候选下标（按概率降序累积到 >= p 为止）。 */
export const nucleusKeep = (probs: Vec, p: number): number[] => {
	const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
	const keep: number[] = [];
	let cum = 0;
	for (const i of order) {
		keep.push(i);
		cum += probs[i];
		if (cum >= p) break;
	}
	return keep;
};

/** 确定性伪随机数（mulberry32），用于可复现的方差演示采样。 */
export const mulberry32 = (seed: number): (() => number) => {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
};

/** Box–Muller 标准正态采样。 */
export const gaussian = (rand: () => number): number => {
	const u = Math.max(1e-12, rand());
	const v = rand();
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
