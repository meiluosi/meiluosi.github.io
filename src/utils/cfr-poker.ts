/**
 * Exact Counterfactual Regret Minimization (CFR) for Kuhn Poker.
 *
 * Deck: J (0), Q (1), K (2). Each player antes 1 chip; a bet/call costs 1 chip.
 * Actions: index 0 = Pass (Check / Fold), index 1 = Bet (Bet / Call).
 *
 * Theoretical Nash Equilibrium properties for Kuhn Poker:
 * - Game value for Player 1 is -1/18 ≈ -0.05556.
 * - For Player 1 opening: P(Bet | J) = α ∈ [0, 1/3], P(Bet | Q) = 0, P(Bet | K) = 3α.
 * - Facing Pass -> Bet: P(Call | J) = 0, P(Call | Q) = α + 1/3, P(Call | K) = 1.
 * - Player 2 facing Pass: P(Bet | J) = 1/3, P(Bet | Q) = 0, P(Bet | K) = 1.
 * - Player 2 facing Bet: P(Call | J) = 0, P(Call | Q) = 1/3, P(Call | K) = 1.
 */

export type KuhnCard = "J" | "Q" | "K";

export interface InfoSetState {
	key: string;
	player: 1 | 2;
	card: KuhnCard;
	history: string;
	regretSum: [number, number];
	strategySum: [number, number];
	currentStrategy: [number, number];
	averageStrategy: [number, number];
}

export interface CfrSnapshot {
	iterations: number;
	expectedValueP1: number;
	nashValueP1: number;
	infoSets: InfoSetState[];
}

const CARDS: KuhnCard[] = ["J", "Q", "K"];

const INFO_SET_ORDER: {
	key: string;
	player: 1 | 2;
	card: KuhnCard;
	history: string;
}[] = [
	{ key: "J", player: 1, card: "J", history: "" },
	{ key: "Q", player: 1, card: "Q", history: "" },
	{ key: "K", player: 1, card: "K", history: "" },
	{ key: "Jp", player: 2, card: "J", history: "p" },
	{ key: "Qp", player: 2, card: "Q", history: "p" },
	{ key: "Kp", player: 2, card: "K", history: "p" },
	{ key: "Jb", player: 2, card: "J", history: "b" },
	{ key: "Qb", player: 2, card: "Q", history: "b" },
	{ key: "Kb", player: 2, card: "K", history: "b" },
	{ key: "Jpb", player: 1, card: "J", history: "pb" },
	{ key: "Qpb", player: 1, card: "Q", history: "pb" },
	{ key: "Kpb", player: 1, card: "K", history: "pb" },
];

export function regretMatching(regretSum: [number, number]): [number, number] {
	const r0 = Math.max(0, regretSum[0]);
	const r1 = Math.max(0, regretSum[1]);
	const total = r0 + r1;
	if (total > 0) {
		return [r0 / total, r1 / total];
	}
	return [0.5, 0.5];
}

export function normalizeStrategySum(
	strategySum: [number, number],
): [number, number] {
	const total = strategySum[0] + strategySum[1];
	if (total > 0) {
		return [strategySum[0] / total, strategySum[1] / total];
	}
	return [0.5, 0.5];
}

export function isTerminalHistory(history: string): boolean {
	return (
		history === "pp" ||
		history === "pbp" ||
		history === "pbb" ||
		history === "bp" ||
		history === "bb"
	);
}

/**
 * Returns payoff for the active player at a terminal history.
 */
export function terminalUtility(
	history: string,
	c1: number,
	c2: number,
): number {
	const plays = history.length;
	const activePlayer = plays % 2 === 0 ? 1 : 2;
	const playerCard = activePlayer === 1 ? c1 : c2;
	const opponentCard = activePlayer === 1 ? c2 : c1;
	const higher = playerCard > opponentCard;

	if (history === "bp" || history === "pbp") {
		// Opponent folded to a bet; active player at this terminal node is the one who bet and won +1.
		return 1;
	}
	if (history === "pp") {
		return higher ? 1 : -1;
	}
	if (history === "bb" || history === "pbb") {
		return higher ? 2 : -2;
	}
	return 0;
}

/**
 * Evaluates Player 1's expected value under a given strategy map (key -> [P(Pass), P(Bet)]).
 */
export function evaluateExpectedValueP1(
	strategies: Map<string, [number, number]>,
): number {
	function evalNode(history: string, c1: number, c2: number): number {
		if (isTerminalHistory(history)) {
			const uActive = terminalUtility(history, c1, c2);
			const activePlayer = history.length % 2 === 0 ? 1 : 2;
			return activePlayer === 1 ? uActive : -uActive;
		}
		const activePlayer = history.length % 2 === 0 ? 1 : 2;
		const card = CARDS[activePlayer === 1 ? c1 : c2];
		const key = `${card}${history}`;
		const strat = strategies.get(key) ?? [0.5, 0.5];
		return (
			strat[0] * evalNode(`${history}p`, c1, c2) +
			strat[1] * evalNode(`${history}b`, c1, c2)
		);
	}

	let total = 0;
	let deals = 0;
	for (let c1 = 0; c1 < 3; c1++) {
		for (let c2 = 0; c2 < 3; c2++) {
			if (c1 === c2) continue;
			total += evalNode("", c1, c2);
			deals++;
		}
	}
	return total / deals;
}

export function runKuhnCfr(iterations: number): CfrSnapshot {
	const safeIterations = Math.max(0, Math.min(20000, Math.floor(iterations)));
	const nodes = new Map<
		string,
		{ regretSum: [number, number]; strategySum: [number, number] }
	>();

	for (const meta of INFO_SET_ORDER) {
		nodes.set(meta.key, {
			regretSum: [0, 0],
			strategySum: [0, 0],
		});
	}

	function cfr(
		c1: number,
		c2: number,
		history: string,
		p0: number,
		p1: number,
	): number {
		if (isTerminalHistory(history)) {
			return terminalUtility(history, c1, c2);
		}
		const activePlayer = history.length % 2 === 0 ? 1 : 2;
		const card = CARDS[activePlayer === 1 ? c1 : c2];
		const key = `${card}${history}`;
		const node = nodes.get(key);
		if (!node) return 0;

		const reachSelf = activePlayer === 1 ? p0 : p1;
		const reachOpp = activePlayer === 1 ? p1 : p0;
		const strategy = regretMatching(node.regretSum);

		node.strategySum[0] += reachSelf * strategy[0];
		node.strategySum[1] += reachSelf * strategy[1];

		const util: [number, number] = [0, 0];
		let nodeUtil = 0;

		for (let a = 0; a < 2; a++) {
			const nextHistory = `${history}${a === 0 ? "p" : "b"}`;
			util[a] =
				activePlayer === 1
					? -cfr(c1, c2, nextHistory, p0 * strategy[a], p1)
					: -cfr(c1, c2, nextHistory, p0, p1 * strategy[a]);
			nodeUtil += strategy[a] * util[a];
		}

		for (let a = 0; a < 2; a++) {
			const regret = util[a] - nodeUtil;
			node.regretSum[a] += reachOpp * regret;
		}

		return nodeUtil;
	}

	for (let t = 0; t < safeIterations; t++) {
		for (let c1 = 0; c1 < 3; c1++) {
			for (let c2 = 0; c2 < 3; c2++) {
				if (c1 === c2) continue;
				cfr(c1, c2, "", 1 / 6, 1 / 6);
			}
		}
	}

	const avgMap = new Map<string, [number, number]>();
	const infoSets: InfoSetState[] = INFO_SET_ORDER.map((meta) => {
		const raw = nodes.get(meta.key) ?? {
			regretSum: [0, 0] as [number, number],
			strategySum: [0, 0] as [number, number],
		};
		const currentStrategy = regretMatching(raw.regretSum);
		const averageStrategy = normalizeStrategySum(raw.strategySum);
		avgMap.set(meta.key, averageStrategy);
		return {
			key: meta.key,
			player: meta.player,
			card: meta.card,
			history: meta.history,
			regretSum: [raw.regretSum[0], raw.regretSum[1]],
			strategySum: [raw.strategySum[0], raw.strategySum[1]],
			currentStrategy,
			averageStrategy,
		};
	});

	return {
		iterations: safeIterations,
		expectedValueP1: evaluateExpectedValueP1(avgMap),
		nashValueP1: -1 / 18,
		infoSets,
	};
}
