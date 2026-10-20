import assert from "node:assert/strict";
import test from "node:test";
import { computeAlignmentSnapshot } from "./preference-alignment";

test("DPO policy probabilities sum to 1 and KL divergence is non-negative", () => {
	const snap = computeAlignmentSnapshot(0.2, 0.3);
	const sum = snap.candidates.reduce((acc, c) => acc + c.policyProb, 0);
	assert.ok(Math.abs(sum - 1) < 1e-9);
	assert.ok(snap.klDivergence >= 0);
});

test("smaller beta increases KL divergence from reference policy", () => {
	const highBeta = computeAlignmentSnapshot(0.8, 0.2);
	const lowBeta = computeAlignmentSnapshot(0.05, 0.2);
	assert.ok(
		lowBeta.klDivergence > highBeta.klDivergence,
		`Expected KL at beta=0.05 (${lowBeta.klDivergence}) > KL at beta=0.8 (${highBeta.klDivergence})`,
	);
});

test("high reward-hacking bias + small beta triggers Goodhart overoptimization", () => {
	const clean = computeAlignmentSnapshot(0.04, 0.0);
	const hacked = computeAlignmentSnapshot(0.04, 0.9);
	assert.equal(clean.overoptimized, false);
	assert.equal(hacked.overoptimized, true);
	assert.ok(hacked.expectedTrueReward < clean.expectedTrueReward);
});
