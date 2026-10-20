/**
 * Rotary Position Embedding (RoPE) & Relative Attention Astrolabe math utilities.
 *
 * For each 2D subspace k in {0, ..., K-1}:
 *   θ_k = (base_eff)^(-2k / d)
 * Rotating q_k by m·θ_k and k_k by n·θ_k yields inner product:
 *   <R(m·θ_k) q_k, R(n·θ_k) k_k> = |q_k||k_k| cos((m - n)·θ_k)
 * which depends only on the relative token distance Δ = m - n.
 */

export type RopeScalingMode = "standard" | "linear-pi" | "ntk-aware";

export interface RopePlaneState {
	index: number;
	labelZh: string;
	labelEn: string;
	theta: number;
	wavelength: number;
	queryAngleRad: number;
	keyAngleRad: number;
	phaseDiffRad: number;
	cosineContribution: number;
}

export interface RopeDecayPoint {
	distance: number;
	kernelScore: number;
}

export interface RopeSnapshot {
	queryPos: number;
	keyPos: number;
	relativeDistance: number;
	base: number;
	scaleFactor: number;
	mode: RopeScalingMode;
	aggregateKernel: number;
	planes: RopePlaneState[];
	decayCurve: RopeDecayPoint[];
}

const PLANE_LABELS: { zh: string; en: string }[] = [
	{ zh: "通道 0 · 极高频 (局部邻近)", en: "Plane 0 · Ultra-High Freq (Local)" },
	{ zh: "通道 1 · 高频 (短语语法)", en: "Plane 1 · High Freq (Phrase)" },
	{ zh: "通道 2 · 中高频 (子句边界)", en: "Plane 2 · Mid-High Freq (Clause)" },
	{
		zh: "通道 3 · 中低频 (段落结构)",
		en: "Plane 3 · Mid-Low Freq (Paragraph)",
	},
	{ zh: "通道 4 · 低频 (跨段指代)", en: "Plane 4 · Low Freq (Cross-Section)" },
	{ zh: "通道 5 · 极低频 (全局长程)", en: "Plane 5 · Ultra-Low Freq (Global)" },
];

export function computeEffectiveTheta(
	planeIndex: number,
	numPlanes: number,
	base: number,
	scaleFactor: number,
	mode: RopeScalingMode,
): number {
	const d = numPlanes * 2;
	const safeScale = Math.max(1, scaleFactor);
	if (mode === "linear-pi") {
		const rawTheta = base ** ((-2 * planeIndex) / d);
		return rawTheta / safeScale;
	}
	if (mode === "ntk-aware") {
		// NTK-aware scaling scales the base so high frequencies are preserved while low frequencies stretch
		const scaledBase = base * safeScale ** (d / (d - 2));
		return scaledBase ** ((-2 * planeIndex) / d);
	}
	return base ** ((-2 * planeIndex) / d);
}

export function computeRopeKernelAtDistance(
	distance: number,
	numPlanes: number,
	base: number,
	scaleFactor: number,
	mode: RopeScalingMode,
): number {
	let sum = 0;
	for (let k = 0; k < numPlanes; k++) {
		const theta = computeEffectiveTheta(k, numPlanes, base, scaleFactor, mode);
		sum += Math.cos(distance * theta);
	}
	return sum / numPlanes;
}

export function computeRopeSnapshot(
	queryPos: number,
	keyPos: number,
	base = 100,
	scaleFactor = 2,
	mode: RopeScalingMode = "standard",
): RopeSnapshot {
	const numPlanes = PLANE_LABELS.length;
	const relativeDistance = queryPos - keyPos;

	const planes: RopePlaneState[] = PLANE_LABELS.map((meta, k) => {
		const theta = computeEffectiveTheta(k, numPlanes, base, scaleFactor, mode);
		const queryAngleRad = queryPos * theta;
		const keyAngleRad = keyPos * theta;
		const phaseDiffRad = relativeDistance * theta;
		const cosineContribution = Math.cos(phaseDiffRad);
		const wavelength = (2 * Math.PI) / Math.max(1e-9, theta);

		return {
			index: k,
			labelZh: meta.zh,
			labelEn: meta.en,
			theta,
			wavelength,
			queryAngleRad,
			keyAngleRad,
			phaseDiffRad,
			cosineContribution,
		};
	});

	const aggregateKernel =
		planes.reduce((acc, p) => acc + p.cosineContribution, 0) / numPlanes;

	const decayCurve: RopeDecayPoint[] = [];
	for (let dist = 0; dist <= 64; dist += 1) {
		decayCurve.push({
			distance: dist,
			kernelScore: computeRopeKernelAtDistance(
				dist,
				numPlanes,
				base,
				scaleFactor,
				mode,
			),
		});
	}

	return {
		queryPos,
		keyPos,
		relativeDistance,
		base,
		scaleFactor,
		mode,
		aggregateKernel,
		planes,
		decayCurve,
	};
}
