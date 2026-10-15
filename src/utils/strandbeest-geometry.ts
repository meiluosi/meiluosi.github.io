/** Ideal planar linkage, redrawn from the project's Jansen-style geometry study.
 * Lengths are dimensionless model inputs, not measurements of a physical beast.
 * The assembly branch stays fixed; impossible circle intersections return null.
 */
export interface LinkPoint {
	x: number;
	y: number;
}
export type JointName = "G" | "O" | "C" | "K" | "L" | "M" | "N" | "F";
export interface LinkSpec {
	from: JointName;
	to: JointName;
	length: number;
}
export interface LegPose {
	points: Partial<Record<JointName, LinkPoint>>;
	valid: boolean;
	failedJoint: JointName | null;
}
export interface FootTrace {
	path: string;
	validSamples: number;
	samples: number;
	width: number | null;
	height: number | null;
}

export const DEFAULT_CRANK: number = 15;
export const STRANDBEEST_VIEWBOX: string = "-90 -64 172 172";
export const STRANDBEEST_LINKS: readonly LinkSpec[] = [
	{ from: "G", to: "K", length: 41.5 },
	{ from: "C", to: "K", length: 50 },
	{ from: "G", to: "L", length: 40.1 },
	{ from: "K", to: "L", length: 55.8 },
	{ from: "G", to: "M", length: 39.3 },
	{ from: "C", to: "M", length: 61.9 },
	{ from: "L", to: "N", length: 39.4 },
	{ from: "M", to: "N", length: 36.7 },
	{ from: "M", to: "F", length: 49 },
	{ from: "N", to: "F", length: 65.7 },
];

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
): LegPose {
	if (!Number.isFinite(degrees) || !Number.isFinite(crank) || crank <= 0) {
		return { points: {}, valid: false, failedJoint: "C" };
	}
	const radians = (degrees * Math.PI) / 180;
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
		["K", "G", 41.5, "C", 50, 1],
		["L", "G", 40.1, "K", 55.8, 1],
		["M", "G", 39.3, "C", 61.9, -1],
		["N", "L", 39.4, "M", 36.7, -1],
		["F", "M", 49, "N", 65.7, 1],
	];
	for (const [id, first, firstRadius, second, secondRadius, side] of joints) {
		const a = points[first];
		const b = points[second];
		const point =
			a && b ? circleIntersection(a, firstRadius, b, secondRadius, side) : null;
		if (!point) return { points, valid: false, failedJoint: id };
		points[id] = point;
	}
	return { points, valid: true, failedJoint: null };
}

/** SVG coordinates reverse y so mathematical positive y remains upward. */
export function svgPoint(point: LinkPoint): string {
	return `${point.x.toFixed(3)},${(-point.y).toFixed(3)}`;
}

export function rodPath(
	pose: LegPose,
	links: readonly LinkSpec[] = STRANDBEEST_LINKS,
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
	samples: number = 360,
): FootTrace {
	if (!Number.isInteger(samples) || samples < 4 || samples > 1440)
		throw new RangeError("samples must be an integer from 4 to 1440");
	let path = "";
	let connected = false;
	const feet: LinkPoint[] = [];
	for (let i = 0; i <= samples; i++) {
		const pose = solveStrandbeest((i * 360) / samples, crank);
		const foot = pose.points.F;
		if (!pose.valid || !foot) {
			connected = false;
			continue;
		}
		path += `${connected ? "L" : "M"}${svgPoint(foot)} `;
		connected = true;
		if (i < samples) feet.push(foot);
	}
	const complete = feet.length === samples;
	return {
		path: path.trim(),
		validSamples: feet.length,
		samples,
		width: complete
			? Math.max(...feet.map((p) => p.x)) - Math.min(...feet.map((p) => p.x))
			: null,
		height: complete
			? Math.max(...feet.map((p) => p.y)) - Math.min(...feet.map((p) => p.y))
			: null,
	};
}
