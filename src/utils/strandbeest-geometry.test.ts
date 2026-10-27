import assert from "node:assert/strict";
import test from "node:test";
import {
	circleIntersection,
	computeFootVelocity,
	computeGaitMetrics,
	STRANDBEEST_LINKS,
	STRANDBEEST_PRESETS,
	solveStrandbeest,
	traceStrandbeest,
} from "./strandbeest-geometry";

const close = (actual: number, expected: number, epsilon = 1e-7) =>
	assert.ok(
		Math.abs(actual - expected) < epsilon,
		`${actual} should equal ${expected}`,
	);

test("circle intersections preserve both radii and assembly side", () => {
	for (const side of [-1, 1] as const) {
		const p = circleIntersection({ x: 0, y: 0 }, 5, { x: 6, y: 0 }, 5, side);
		assert.ok(p);
		close(Math.hypot(p.x, p.y), 5);
		close(Math.hypot(p.x - 6, p.y), 5);
		assert.equal(Math.sign(p.y), side);
	}
	assert.deepEqual(
		circleIntersection({ x: 0, y: 0 }, 3, { x: 6, y: 0 }, 3, 1),
		{ x: 3, y: 0 },
	);
});

test("separated, contained, coincident and invalid circles are not invented", () => {
	assert.equal(
		circleIntersection({ x: 0, y: 0 }, 2, { x: 6, y: 0 }, 2, 1),
		null,
	);
	assert.equal(
		circleIntersection({ x: 0, y: 0 }, 5, { x: 1, y: 0 }, 2, 1),
		null,
	);
	assert.equal(
		circleIntersection({ x: 0, y: 0 }, 5, { x: 0, y: 0 }, 5, 1),
		null,
	);
	assert.equal(
		circleIntersection({ x: Number.NaN, y: 0 }, 5, { x: 6, y: 0 }, 5, 1),
		null,
	);
	assert.equal(solveStrandbeest(0, -1).valid, false);
	assert.equal(solveStrandbeest(Number.POSITIVE_INFINITY).valid, false);
});

test("one sampled revolution preserves every rod and crank length", () => {
	for (let i = 0; i <= 1440; i++) {
		const pose = solveStrandbeest(i / 4);
		assert.ok(pose.valid, `unreachable default pose at ${i / 4} degrees`);
		for (const link of [
			...STRANDBEEST_LINKS,
			{ from: "O", to: "C", length: 15 },
		] as const) {
			const a = pose.points[link.from];
			const b = pose.points[link.to];
			assert.ok(a && b);
			close(Math.hypot(a.x - b.x, a.y - b.y), link.length);
		}
	}
	const start = solveStrandbeest(0).points;
	const end = solveStrandbeest(360).points;
	for (const name of Object.keys(start) as (keyof typeof start)[]) {
		close(start[name]!.x, end[name]!.x);
		close(start[name]!.y, end[name]!.y);
	}
	assert.equal(traceStrandbeest().validSamples, 360);
});

test("changed geometry leaves gaps instead of bridging unreachable phases", () => {
	const trace = traceStrandbeest(30);
	assert.ok(trace.validSamples > 0 && trace.validSamples < 360);
	assert.equal(trace.width, null);
	assert.equal(trace.height, null);
	assert.ok((trace.path.match(/M/g) ?? []).length >= 2);
	assert.ok(!/NaN|Infinity/.test(trace.path));
	const unreachable = Array.from({ length: 360 }, (_, angle) =>
		solveStrandbeest(angle, 30),
	).find((pose) => !pose.valid);
	assert.ok(unreachable);
	assert.equal(unreachable.points.F, undefined);
	assert.notEqual(unreachable.failedJoint, null);
	assert.throws(() => traceStrandbeest(15, 0), RangeError);
});

test("gait metrics, velocity vector, mirrored leg, and presets compute accurately", () => {
	const jansenMetrics = computeGaitMetrics();
	assert.equal(jansenMetrics.validSamples, 360);
	assert.ok(jansenMetrics.strideWidth && jansenMetrics.strideWidth > 40);
	assert.ok(jansenMetrics.liftHeight && jansenMetrics.liftHeight > 15);
	assert.ok(
		jansenMetrics.flatnessScore !== null && jansenMetrics.flatnessScore >= 70,
	);
	assert.ok(
		jansenMetrics.dutyFactor !== null &&
			jansenMetrics.dutyFactor >= 25 &&
			jansenMetrics.dutyFactor <= 75,
	);

	const vel = computeFootVelocity(205);
	assert.ok(vel && vel.speed > 0);

	// Mirrored leg shares the exact same crank pin C(x, y) and mirrors G to x=76
	const left = solveStrandbeest(205, 15, undefined, false);
	const right = solveStrandbeest(205, 15, undefined, true);
	assert.ok(left.valid && right.valid);
	assert.ok(
		right.points.G && right.points.O && left.points.C && right.points.C,
	);
	close(right.points.G.x, 76);
	close(right.points.O.x, 38);
	close(left.points.C.x, right.points.C.x);
	close(left.points.C.y, right.points.C.y);

	for (const preset of STRANDBEEST_PRESETS) {
		const m = computeGaitMetrics(preset.crank, preset.lengths);
		assert.ok(m.validSamples > 0);
	}
});
