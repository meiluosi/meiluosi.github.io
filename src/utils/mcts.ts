export interface SearchNode {
	id: number;
	label: string;
	parent: number | null;
	children: number[];
	x: number;
	y: number;
	reward: number | null;
	expanded: boolean;
	visits: number;
	total: number;
}
export interface SearchState {
	nodes: SearchNode[];
	phase: number;
	iterations: number;
	path: number[];
	rollout: number[];
	reward: number | null;
	random: number;
	expandedId: number | null;
}
export function createSearch(seed = 7): SearchState {
	const nodes: SearchNode[] = [];
	const add = (
		label: string,
		parent: number | null,
		x: number,
		y: number,
		reward: number | null = null,
	): number => {
		const id = nodes.length;
		nodes.push({
			id,
			label,
			parent,
			x,
			y,
			reward,
			children: [],
			expanded: id === 0,
			visits: 0,
			total: 0,
		});
		if (parent !== null) nodes[parent].children.push(id);
		return id;
	};
	add("START", null, 480, 40);
	const rewards = [
		[0.5, 0.45, 0.6, 0.4],
		[0.05, 0.1, 0.15, 1],
		[0.55, 0.65, 0.7, 0.6],
	];
	for (let branch = 0; branch < 3; branch++) {
		const label = String.fromCharCode(65 + branch);
		const parent = add(label, 0, 160 + branch * 320, 140);
		for (let group = 0; group < 2; group++) {
			const inner = add(
				`${label}${group + 1}`,
				parent,
				80 + branch * 320 + group * 160,
				245,
			);
			for (let leaf = 0; leaf < 2; leaf++) {
				const offset = group * 2 + leaf;
				add(
					`${label}${offset + 1}′`,
					inner,
					40 + branch * 320 + offset * 80,
					355,
					rewards[branch][offset],
				);
			}
		}
	}
	return {
		nodes,
		phase: 0,
		iterations: 0,
		path: [],
		rollout: [],
		reward: null,
		random: seed >>> 0,
		expandedId: null,
	};
}

export function uctScore(
	node: SearchNode,
	parentVisits: number,
	exploration: number,
): number {
	return node.visits === 0
		? Number.POSITIVE_INFINITY
		: node.total / node.visits +
				exploration *
					Math.sqrt(Math.log(Math.max(1, parentVisits)) / node.visits);
}

// One call advances one visible phase. Simulation-only states are not added to the tree.
export function advanceSearch(
	previous: SearchState,
	exploration: number,
): SearchState {
	const state: SearchState = {
		...previous,
		nodes: previous.nodes.map((node) => ({ ...node })),
		path: [...previous.path],
		rollout: [...previous.rollout],
	};
	const choose = (ids: number[]): number => {
		state.random = (Math.imul(1664525, state.random) + 1013904223) >>> 0;
		return ids[Math.floor((state.random / 4294967296) * ids.length)];
	};
	state.phase = (previous.phase % 4) + 1;
	if (state.phase === 1) {
		state.path = [0];
		state.rollout = [];
		state.reward = null;
		state.expandedId = null;
		let node = state.nodes[0];
		while (
			node.children.length &&
			node.children.every((id) => state.nodes[id].expanded)
		) {
			const scores = node.children.map((id) =>
				uctScore(state.nodes[id], node.visits, exploration),
			);
			const best = Math.max(...scores);
			const id = choose(node.children.filter((_, i) => scores[i] === best));
			state.path.push(id);
			node = state.nodes[id];
		}
	} else if (state.phase === 2) {
		const node = state.nodes[state.path[state.path.length - 1]];
		const untried = node.children.filter((id) => !state.nodes[id].expanded);
		if (untried.length) {
			const id = choose(untried);
			state.nodes[id].expanded = true;
			state.expandedId = id;
			state.path.push(id);
		}
	} else if (state.phase === 3) {
		let node = state.nodes[state.path[state.path.length - 1]];
		state.rollout = [node.id];
		while (node.children.length) {
			node = state.nodes[choose(node.children)];
			state.rollout.push(node.id);
		}
		state.reward = node.reward;
	} else {
		for (const id of state.path) {
			state.nodes[id].visits++;
			state.nodes[id].total += state.reward ?? 0;
		}
		state.iterations++;
	}
	return state;
}

export function recommendedAction(state: SearchState): SearchNode | null {
	const visited = state.nodes[0].children
		.map((id) => state.nodes[id])
		.filter((node) => node.visits > 0);
	return (
		visited.sort(
			(a, b) =>
				b.visits - a.visits ||
				b.total / b.visits - a.total / a.visits ||
				a.id - b.id,
		)[0] ?? null
	);
}
