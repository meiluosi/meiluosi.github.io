/** Ideal planar linkage, redrawn from the project's Jansen-style geometry study.
 * Lengths are dimensionless model inputs, not measurements of a physical beast.
 * The assembly branch stays fixed; impossible circle intersections return null.
 */
export interface LinkPoint {
	x: number;
	y: number;
}
export type JointName = "G" | "O" | "C" | "K" | "L" | "M" | "N" | "F";
export type LinkId =
	| "GK"
	| "CK"
	| "GL"
	| "KL"
	| "GM"
	| "CM"
	| "LN"
	| "MN"
	| "MF"
	| "NF";

export interface LinkSpec {
	id: LinkId;
	from: JointName;
	to: JointName;
	length: number;
}
export type CustomLinkLengths = Partial<Record<LinkId, number>>;

export interface LegPose {
	points: Partial<Record<JointName, LinkPoint>>;
	valid: boolean;
	failedJoint: JointName | null;
}
export interface FootTrace {
	path: string;
	stancePath: string;
	swingPath: string;
	validSamples: number;
	samples: number;
	width: number | null;
	height: number | null;
	minY: number | null;
	maxY: number | null;
	flatness: number | null;
	dutyFactor: number | null;
}
export interface GaitMetrics {
	validSamples: number;
	samples: number;
	strideWidth: number | null;
	liftHeight: number | null;
	groundY: number | null;
	flatnessScore: number | null;
	dutyFactor: number | null;
}
export interface BeastPreset {
	id: string;
	nameZh: string;
	nameEn: string;
	noteZh: string;
	noteEn: string;
	crank: number;
	lengths: CustomLinkLengths;
}

export const DEFAULT_CRANK: number = 15;
export const STRANDBEEST_VIEWBOX: string = "-90 -64 172 172";
export const STRANDBEEST_LINKS: readonly LinkSpec[] = [
	{ id: "GK", from: "G", to: "K", length: 41.5 },
	{ id: "CK", from: "C", to: "K", length: 50 },
	{ id: "GL", from: "G", to: "L", length: 40.1 },
	{ id: "KL", from: "K", to: "L", length: 55.8 },
	{ id: "GM", from: "G", to: "M", length: 39.3 },
	{ id: "CM", from: "C", to: "M", length: 61.9 },
	{ id: "LN", from: "L", to: "N", length: 39.4 },
	{ id: "MN", from: "M", to: "N", length: 36.7 },
	{ id: "MF", from: "M", to: "F", length: 49 },
	{ id: "NF", from: "N", to: "F", length: 65.7 },
];

export const DEFAULT_LINK_MAP: Record<LinkId, number> = {
	GK: 41.5,
	CK: 50,
	GL: 40.1,
	KL: 55.8,
	GM: 39.3,
	CM: 61.9,
	LN: 39.4,
	MN: 36.7,
	MF: 49,
	NF: 65.7,
};

export const STRANDBEEST_PRESETS: readonly BeastPreset[] = [
	{
		id: "jansen",
		nameZh: "Jansen 神圣数字",
		nameEn: "Jansen Holy Numbers",
		noteZh:
			"Atari 遗传算法筛出的经典比例：触地段平直，抬脚轻盈，360° 全程闭合。",
		noteEn:
			"Classic proportions found via Atari GA: flat stance phase, clean lift, and 360° closure.",
		crank: 15,
		lengths: {},
	},
	{
		id: "early-proto",
		nameZh: "早期未调优原型",
		nameEn: "Untuned Early Prototype",
		noteZh: "模拟演化初期的粗糙比例：轨迹呈椭圆起伏，机身随迈步剧烈颠簸。",
		noteEn:
			"Untuned early-generation ratios: oval foot trajectory causing heavy vertical chassis bounce.",
		crank: 13.5,
		lengths: {
			GK: 44.0,
			CK: 52.0,
			GL: 42.5,
			KL: 53.0,
			GM: 41.0,
			CM: 59.5,
			LN: 41.0,
			MN: 38.5,
			MF: 46.0,
			NF: 63.0,
		},
	},
	{
		id: "dune-high",
		nameZh: "高抬腿沙丘步态",
		nameEn: "High-Lift Dune Stepper",
		noteZh: "增大下三角与曲柄行程，抬腿高度显著提升，适合跨越松软沙脊。",
		noteEn:
			"Amplified lower triangle and crank throw for higher foot clearance over soft sand ridges.",
		crank: 17.5,
		lengths: {
			GK: 42.0,
			CK: 51.0,
			GL: 40.1,
			KL: 55.8,
			GM: 39.8,
			CM: 62.5,
			LN: 39.0,
			MN: 37.2,
			MF: 52.5,
			NF: 67.5,
		},
	},
	{
		id: "wide-stride",
		nameZh: "大步幅疾行变体",
		nameEn: "Wide-Stride Sprinter",
		noteZh: "拉长水平摆幅以换取更大单步跨度，但触地段平直度有所牺牲。",
		noteEn:
			"Extended horizontal sweep for a longer stride, trading off stance flatness.",
		crank: 16.5,
		lengths: {
			GK: 40.2,
			CK: 49.2,
			GL: 39.5,
			KL: 54.8,
			GM: 38.5,
			CM: 61.2,
			LN: 40.2,
			MN: 36.2,
			MF: 51.0,
			NF: 67.0,
		},
	},
	{
		id: "singularity",
		nameZh: "死点奇异态 (曲柄=30)",
		nameEn: "Locked Singularity (Crank=30)",
		noteZh: "曲柄过长导致部分角度两圆无法相交，展示几何约束的刚性边界。",
		noteEn:
			"Oversized crank causes circle intersections to fail across parts of the cycle.",
		crank: 30,
		lengths: {},
	},
];

export function resolveLinkSpecs(
	custom?: CustomLinkLengths,
): readonly LinkSpec[] {
	if (!custom || Object.keys(custom).length === 0) return STRANDBEEST_LINKS;
	return STRANDBEEST_LINKS.map((spec) => ({
		...spec,
		length:
			custom[spec.id] !== undefined && Number.isFinite(custom[spec.id])
				? (custom[spec.id] as number)
				: spec.length,
	}));
}

export function circleIntersection(
	left: LinkPoint,
	leftRadius: number,
	right: LinkPoint,
	rightRadius: number,
	side: 1 | -1,
): LinkPoint | null {
	if (
		![left.x, left.y, right.x, right.y, leftRadius, rightRadius].every(
			Number.isFinite,
		) ||
		leftRadius <= 0 ||
		rightRadius <= 0
	)
		return null;
	const dx = right.x - left.x;
	const dy = right.y - left.y;
	const distance = Math.hypot(dx, dy);
	const tolerance = 1e-9 * Math.max(1, distance, leftRadius, rightRadius);
	if (
		distance <= tolerance ||
		distance > leftRadius + rightRadius + tolerance ||
		distance < Math.abs(leftRadius - rightRadius) - tolerance
	)
		return null;
	const along =
		(leftRadius ** 2 - rightRadius ** 2 + distance ** 2) / (2 * distance);
	const heightSquared = leftRadius ** 2 - along ** 2;
	if (heightSquared < -tolerance * Math.max(1, leftRadius)) return null;
	const height = Math.sqrt(Math.max(0, heightSquared));
	return {
		x: left.x + (along * dx) / distance - (side * height * dy) / distance,
		y: left.y + (along * dy) / distance + (side * height * dx) / distance,
	};
}

export function solveStrandbeest(
	degrees: number,
	crank: number = DEFAULT_CRANK,
	customLengths?: CustomLinkLengths,
	mirrorX = false,
): LegPose {
	if (!Number.isFinite(degrees) || !Number.isFinite(crank) || crank <= 0) {
		return { points: {}, valid: false, failedJoint: "C" };
	}
	const L = customLengths
		? { ...DEFAULT_LINK_MAP, ...customLengths }
		: DEFAULT_LINK_MAP;
	const effectiveDegrees = mirrorX ? 180 - degrees : degrees;
	const radians = (effectiveDegrees * Math.PI) / 180;
	const points: LegPose["points"] = {
		G: { x: 0, y: 0 },
		O: { x: 38, y: 7.8 },
		C: {
			x: 38 + crank * Math.cos(radians),
			y: 7.8 + crank * Math.sin(radians),
		},
	};
	const joints: readonly [
		JointName,
		JointName,
		number,
		JointName,
		number,
		1 | -1,
	][] = [
		["K", "G", L.GK, "C", L.CK, 1],
		["L", "G", L.GL, "K", L.KL, 1],
		["M", "G", L.GM, "C", L.CM, -1],
		["N", "L", L.LN, "M", L.MN, -1],
		["F", "M", L.MF, "N", L.NF, 1],
	];
	for (const [id, first, firstRadius, second, secondRadius, side] of joints) {
		const a = points[first];
		const b = points[second];
		const point =
			a && b ? circleIntersection(a, firstRadius, b, secondRadius, side) : null;
		if (!point) {
			if (mirrorX) {
				for (const key of Object.keys(points) as JointName[]) {
					const pt = points[key];
					if (pt) points[key] = { x: 76 - pt.x, y: pt.y };
				}
			}
			return { points, valid: false, failedJoint: id };
		}
		points[id] = point;
	}
	if (mirrorX) {
		for (const key of Object.keys(points) as JointName[]) {
			const pt = points[key];
			if (pt) points[key] = { x: 76 - pt.x, y: pt.y };
		}
	}
	return { points, valid: true, failedJoint: null };
}

/** Computes instantaneous foot velocity vector via central finite difference. */
export function computeFootVelocity(
	degrees: number,
	crank: number = DEFAULT_CRANK,
	customLengths?: CustomLinkLengths,
): { vx: number; vy: number; speed: number } | null {
	const delta = 1.5;
	const prev = solveStrandbeest(degrees - delta, crank, customLengths);
	const next = solveStrandbeest(degrees + delta, crank, customLengths);
	if (!prev.valid || !next.valid || !prev.points.F || !next.points.F) {
		return null;
	}
	const vx = (next.points.F.x - prev.points.F.x) / (2 * delta);
	const vy = (next.points.F.y - prev.points.F.y) / (2 * delta);
	return { vx, vy, speed: Math.hypot(vx, vy) };
}

/** SVG coordinates reverse y so mathematical positive y remains upward. */
export function svgPoint(point: LinkPoint): string {
	return `${point.x.toFixed(3)},${(-point.y).toFixed(3)}`;
}

export function rodPath(
	pose: LegPose,
	links: readonly Pick<LinkSpec, "from" | "to">[] = STRANDBEEST_LINKS,
): string {
	return links
		.map(({ from, to }) => {
			const a = pose.points[from];
			const b = pose.points[to];
			return a && b ? `M${svgPoint(a)}L${svgPoint(b)}` : "";
		})
		.join(" ");
}

export function polygonPath(
	pose: LegPose,
	names: readonly JointName[],
): string {
	const points = names.map((name) => pose.points[name]);
	return points.every((point) => point !== undefined)
		? `M${points.map((point) => svgPoint(point as LinkPoint)).join("L")}Z`
		: "";
}

export function traceStrandbeest(
	crank: number = DEFAULT_CRANK,
	samples = 360,
	customLengths?: CustomLinkLengths,
	mirrorX = false,
): FootTrace {
	if (!Number.isInteger(samples) || samples < 4 || samples > 1440)
		throw new RangeError("samples must be an integer from 4 to 1440");
	let path = "";
	let connected = false;
	const feet: LinkPoint[] = [];
	const sampledFeet: (LinkPoint | null)[] = [];
	for (let i = 0; i <= samples; i++) {
		const pose = solveStrandbeest(
			(i * 360) / samples,
			crank,
			customLengths,
			mirrorX,
		);
		const foot = pose.points.F;
		if (!pose.valid || !foot) {
			connected = false;
			if (i < samples) sampledFeet.push(null);
			continue;
		}
		path += `${connected ? "L" : "M"}${svgPoint(foot)} `;
		connected = true;
		if (i < samples) {
			feet.push(foot);
			sampledFeet.push(foot);
		}
	}
	const complete = feet.length === samples;
	const minY = feet.length > 0 ? Math.min(...feet.map((p) => p.y)) : null;
	const maxY = feet.length > 0 ? Math.max(...feet.map((p) => p.y)) : null;
	const width = complete
		? Math.max(...feet.map((p) => p.x)) - Math.min(...feet.map((p) => p.x))
		: null;
	const height =
		complete && minY !== null && maxY !== null ? maxY - minY : null;

	// Partition into ground-contact stance path (bottom 15% of vertical range) and swing path
	let stancePath = "";
	let swingPath = "";
	let stanceConnected = false;
	let swingConnected = false;
	const stanceThreshold =
		minY !== null && maxY !== null
			? minY + (maxY - minY) * 0.15
			: Number.NEGATIVE_INFINITY;

	const stancePoints: LinkPoint[] = [];
	for (let i = 0; i <= samples; i++) {
		const pt = sampledFeet[i % samples];
		if (!pt) {
			stanceConnected = false;
			swingConnected = false;
			continue;
		}
		const isStance = pt.y <= stanceThreshold;
		if (isStance) {
			stancePath += `${stanceConnected ? "L" : "M"}${svgPoint(pt)} `;
			stanceConnected = true;
			swingConnected = false;
			if (i < samples) stancePoints.push(pt);
		} else {
			swingPath += `${swingConnected ? "L" : "M"}${svgPoint(pt)} `;
			swingConnected = true;
			stanceConnected = false;
		}
	}

	let flatness: number | null = null;
	let dutyFactor: number | null = null;
	if (complete && height && height > 0 && stancePoints.length > 4) {
		dutyFactor = Math.round((stancePoints.length / samples) * 100);
		const meanY =
			stancePoints.reduce((acc, p) => acc + p.y, 0) / stancePoints.length;
		const variance =
			stancePoints.reduce((acc, p) => acc + (p.y - meanY) ** 2, 0) /
			stancePoints.length;
		const stdDev = Math.sqrt(variance);
		// Normalize stdDev relative to total height into a 0-100 flatness score
		flatness = Math.max(
			0,
			Math.min(100, Math.round(100 * (1 - (stdDev / height) * 4.5))),
		);
	}

	return {
		path: path.trim(),
		stancePath: stancePath.trim(),
		swingPath: swingPath.trim(),
		validSamples: feet.length,
		samples,
		width,
		height,
		minY,
		maxY,
		flatness,
		dutyFactor,
	};
}

export function computeGaitMetrics(
	crank: number = DEFAULT_CRANK,
	customLengths?: CustomLinkLengths,
): GaitMetrics {
	const trace = traceStrandbeest(crank, 360, customLengths);
	return {
		validSamples: trace.validSamples,
		samples: trace.samples,
		strideWidth: trace.width,
		liftHeight: trace.height,
		groundY: trace.minY,
		flatnessScore: trace.flatness,
		dutyFactor: trace.dutyFactor,
	};
}
