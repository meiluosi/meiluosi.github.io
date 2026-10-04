interface Point { x: number; z: number; }

export const WORLD_ROVER_CLEARANCE = 1.46;
const ROUTE_CLEARANCE = 1.54;

/** Small visibility-smoothed A* route around plinths; computed only on station selection. */
function searchRoute(start: Point, goal: Point, obstacles: readonly Point[], step: number, requiredClearance: number): Point[] {
	const columns = Math.round(11.52 / step) + 1;
	const rows = Math.round(8.64 / step) + 1;
	const points: Point[] = Array.from({ length: columns * rows }, (_, index) => ({
		x: (index % columns - (columns - 1) / 2) * step,
		z: (Math.floor(index / columns) - (rows - 1) / 2) * step,
	}));
	const valid = points.map((point) => Math.hypot(point.x / 5.75, point.z / 4.16) < 1 && obstacles.every((obstacle) => Math.hypot(point.x - obstacle.x, point.z - obstacle.z) > requiredClearance));
	const distance = (a: Point, b: Point): number => Math.hypot(a.x - b.x, a.z - b.z);
	function clearSegment(a: Point, b: Point): boolean {
		const dx = b.x - a.x;
		const dz = b.z - a.z;
		const lengthSquared = dx * dx + dz * dz;
		return obstacles.every((obstacle) => {
			const fraction = lengthSquared ? Math.max(0, Math.min(1, ((obstacle.x - a.x) * dx + (obstacle.z - a.z) * dz) / lengthSquared)) : 0;
			const clearance = Math.hypot(obstacle.x - (a.x + fraction * dx), obstacle.z - (a.z + fraction * dz));
			// Keep room for steering. A manually parked rover may already touch a plinth;
			// allow its first segment to head outwards, never through that plinth.
			return clearance >= requiredClearance || (fraction === 0 && clearance >= WORLD_ROVER_CLEARANCE - 1e-8 && distance(b, obstacle) >= requiredClearance);
		});
	}
	if (clearSegment(start, goal)) return [goal];
	function nearest(point: Point): number {
		let nearestIndex = -1;
		let nearestDistance = Number.POSITIVE_INFINITY;
		points.forEach((candidate, index) => {
			const candidateDistance = distance(point, candidate);
			if (valid[index] && candidateDistance < nearestDistance && clearSegment(point, candidate)) {
				nearestIndex = index;
				nearestDistance = candidateDistance;
			}
		});
		return nearestIndex;
	}
	const from = nearest(start);
	const to = nearest(goal);
	if (from < 0 || to < 0) return [];
	const costs = new Float64Array(points.length).fill(Number.POSITIVE_INFINITY);
	const previous = new Int32Array(points.length).fill(-1);
	const open = new Set([from]);
	costs[from] = 0;
	while (open.size) {
		let current = -1;
		let best = Number.POSITIVE_INFINITY;
		for (const index of open) {
			const score = costs[index] + distance(points[index], points[to]);
			if (score < best) { current = index; best = score; }
		}
		if (current === to) {
			const route: Point[] = [goal];
			for (let index = to; index >= 0; index = previous[index]) route.unshift(points[index]);
			const smooth: Point[] = [];
			let anchor = start;
			for (let index = 0; index < route.length;) {
				let farthest = index;
				while (farthest + 1 < route.length && clearSegment(anchor, route[farthest + 1])) farthest++;
				anchor = route[farthest];
				smooth.push(anchor);
				index = farthest + 1;
			}
			return smooth;
		}
		open.delete(current);
		const x = current % columns;
		const z = Math.floor(current / columns);
		for (let dz = -1; dz <= 1; dz++) {
			for (let dx = -1; dx <= 1; dx++) {
				if ((!dx && !dz) || x + dx < 0 || x + dx >= columns || z + dz < 0 || z + dz >= rows) continue;
				const next = (z + dz) * columns + x + dx;
				if (!valid[next] || !clearSegment(points[current], points[next])) continue;
				const cost = costs[current] + distance(points[current], points[next]);
				if (cost < costs[next]) { costs[next] = cost; previous[next] = current; open.add(next); }
			}
		}
	}
	return [];
}

export function planWorldRoute(start: Point, goal: Point, obstacles: readonly Point[]): Point[] {
	const route = searchRoute(start, goal, obstacles, 0.36, ROUTE_CLEARANCE);
	// The coarse grid can miss the narrow strip between a plinth and the coast.
	// Use a finer grid there, keeping the physical collision radius as a hard limit.
	return route.length ? route : searchRoute(start, goal, obstacles, 0.18, WORLD_ROVER_CLEARANCE + 0.002);
}
