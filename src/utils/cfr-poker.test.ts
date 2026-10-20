import assert from "node:assert/strict";
import test from "node:test";
import { regretMatching, runKuhnCfr } from "./cfr-poker";

test("regretMatching returns uniform strategy when regrets are non-positive", () => {
	assert.deepEqual(regretMatching([0, 0]), [0.5, 0.5]);
	assert.deepEqual(regretMatching([-3, -1]), [0.5, 0.5]);
	const matched = regretMatching([1, 3]);
	assert.ok(Math.abs(matched[0] - 0.25) < 1e-9);
	assert.ok(Math.abs(matched[1] - 0.75) < 1e-9);
});

test("Kuhn Poker CFR converges toward Nash equilibrium game value (-1/18) and dominance invariants", () => {
	const snapshot = runKuhnCfr(2500);
	assert.equal(snapshot.infoSets.length, 12);
	// Expected value of Player 1 converges to -1/18 ≈ -0.05556
	assert.ok(
		Math.abs(snapshot.expectedValueP1 - snapshot.nashValueP1) < 0.004,
		`Expected EV near -1/18, got ${snapshot.expectedValueP1}`,
	);

	const byKey = new Map(snapshot.infoSets.map((s) => [s.key, s]));
	// Dominance invariants:
	// 1. P2 facing Bet with K always calls (Kb -> Bet/Call ≈ 1)
	assert.ok((byKey.get("Kb")?.averageStrategy[1] ?? 0) > 0.98);
	// 2. P2 facing Bet with J always folds (Jb -> Bet/Call ≈ 0)
	assert.ok((byKey.get("Jb")?.averageStrategy[1] ?? 1) < 0.02);
	// 3. P2 facing Pass with J bluffs ~ 1/3 of the time (Jp -> Bet ≈ 1/3)
	const jpBet = byKey.get("Jp")?.averageStrategy[1] ?? 0;
	assert.ok(
		Math.abs(jpBet - 1 / 3) < 0.04,
		`Expected Jp bet ~ 1/3, got ${jpBet}`,
	);
	// 4. P1 value-bet with K is ~ 3x P1 bluff with J (K = 3 * α)
	const jBet = byKey.get("J")?.averageStrategy[1] ?? 0;
	const kBet = byKey.get("K")?.averageStrategy[1] ?? 0;
	assert.ok(
		Math.abs(kBet - 3 * jBet) < 0.05,
		`Expected K bet ≈ 3*J bet (${kBet} vs ${3 * jBet})`,
	);
});
