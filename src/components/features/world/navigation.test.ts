import assert from "node:assert/strict";
import test from "node:test";
import { researchThreads } from "../../../data/researchThreads";
import { planWorldRoute, WORLD_ROVER_CLEARANCE } from "./navigation";

interface Point {
	x: number;
	z: number;
}

const tolerance = 1e-8;
const stations = researchThreads;
const insideIsland = (point: Point): boolean =>
	Math.hypot(point.x / 5.84, point.z / 4.25) <= 1 + tolerance;
const separation = (a: Point, b: Point): number =>
	Math.hypot(a.x - b.x, a.z - b.z);

// Check the continuous line segment, not only the grid vertices: a smooth route
// can clip a circular plinth even when every one of its waypoints is valid.
function distanceToSegment(point: Point, from: Point, to: Point): number {
	const dx = to.x - from.x;
	const dz = to.z - from.z;
	const lengthSquared = dx * dx + dz * dz;
	const projection = lengthSquared
		? ((point.x - from.x) * dx + (point.z - from.z) * dz) / lengthSquared
		: 0;
	const fraction = Math.max(0, Math.min(1, projection));
	return Math.hypot(
		point.x - from.x - fraction * dx,
		point.z - from.z - fraction * dz,
	);
}

test("station routes reach their destination without crossing a plinth or the coast", (context) => {
	const starts: Point[] = [];
	for (let x = -5.5; x <= 5.5; x += 0.5) {
		for (let z = -4; z <= 4; z += 0.5) starts.push({ x, z });
	}
	// Include manual-driving contact points, especially the narrow coast behind
	// the feedback station, which is missed by a coarse navigation grid.
	for (const station of stations) {
		for (let direction = 0; direction < 36; direction++) {
			const angle = (direction * Math.PI) / 18;
			starts.push({
				x: station.x + WORLD_ROVER_CLEARANCE * Math.cos(angle),
				z: station.z + WORLD_ROVER_CLEARANCE * Math.sin(angle),
			});
		}
	}
	const goals = stations.map((station) => {
		const radius = Math.hypot(station.x, station.z);
		return {
			x: station.x - (station.x / radius) * 1.64,
			z: station.z - (station.z / radius) * 1.64,
		};
	});
	let checked = 0;
	for (const start of starts) {
		if (!insideIsland(start) || stations.some((station) =>
			separation(start, station) < WORLD_ROVER_CLEARANCE - tolerance,
		)) continue;
		for (const goal of goals) {
			const description = `${JSON.stringify(start)} → ${JSON.stringify(goal)}`;
			const route = planWorldRoute(start, goal, stations);
			assert.ok(route.length > 0, `No route: ${description}`);
			assert.deepEqual(route.at(-1), goal, `Wrong destination: ${description}`);
			let previous = start;
			for (const point of route) {
				// The walkable ellipse is convex, so its straight segments are also inside.
				assert.ok(insideIsland(point), `Route leaves the coast: ${description}`);
				for (const station of stations) {
					assert.ok(
						distanceToSegment(station, previous, point) >= WORLD_ROVER_CLEARANCE - tolerance,
						`Route clips ${station.id}: ${description}`,
					);
				}
				previous = point;
			}
			checked++;
		}
	}
	context.diagnostic(`${checked} routes checked, including exact obstacle contact and coastal starts.`);
});
