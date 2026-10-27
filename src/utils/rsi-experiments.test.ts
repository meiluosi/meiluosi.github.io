import assert from "node:assert/strict";
import test from "node:test";
import { evaluateRepair, selectCards } from "./rsi-experiments";

test("familiar examples cannot distinguish the three selection rules", () => {
	for (const rule of ["color", "position", "shape"] as const) {
		assert.equal(
			selectCards(rule, "familiar").filter((card) => card.correct).length,
			6,
		);
	}
});

test("changing one nuisance feature exposes its corresponding shortcut", () => {
	assert.equal(
		selectCards("color", "color").filter((card) => card.correct).length,
		0,
	);
	assert.equal(
		selectCards("position", "position").filter((card) => card.correct).length,
		2,
	);
	for (const scene of ["color", "position"] as const) {
		assert.equal(
			selectCards("shape", scene).filter((card) => card.correct).length,
			6,
		);
	}
});

test("a new target makes every unchanged rule fail on these cards", () => {
	for (const rule of ["color", "position", "shape"] as const) {
		const cards = selectCards(rule, "task");
		assert.equal(cards.filter((card) => card.correct).length, 0);
		assert.ok(cards.every((card) => card.expected === !card.triangle));
	}
});

test("repair ranking changes with declared costs, while outcomes stay fixed", () => {
	assert.deepEqual(evaluateRepair(4, 1), {
		inspected: 6,
		extraChecks: 3,
		missedRepairs: 0,
		cost: 3,
	});
	assert.deepEqual(evaluateRepair(8, 1), {
		inspected: 2,
		extraChecks: 0,
		missedRepairs: 1,
		cost: 1,
	});
	assert.equal(evaluateRepair(4, 3).cost, evaluateRepair(8, 3).cost);
	assert.ok(evaluateRepair(4, 5).cost < evaluateRepair(8, 5).cost);
});
