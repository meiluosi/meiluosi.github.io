import assert from "node:assert/strict";
import { test } from "node:test";
import {
	dpoPairLoss,
	encodeToyBpe,
	maskedNll,
	trainingBudget,
	trainToyBpe,
} from "./model-pipeline";

test("BPE counts repeated training words, merges non-overlapping pairs, and freezes the vocabulary", () => {
	const tokenizer = trainToyBpe(["low", "low", "lower", "lowest"], 2);
	assert.deepEqual(tokenizer.merges, [
		{ left: "l", right: "o", count: 4 },
		{ left: "lo", right: "w", count: 4 },
	]);
	assert.deepEqual(encodeToyBpe("lower", tokenizer), [["low", "e", "r"]]);
	assert.deepEqual(encodeToyBpe("snow", tokenizer), [["s", "<unk>", "o", "w"]]);
	assert.equal(tokenizer.alphabet.includes("n"), false);
	assert.deepEqual(trainToyBpe(["aaa"], 1).words, [["aa", "a"]]);
	assert.deepEqual(trainToyBpe([], 8).merges, []);
});

test("NLL ignores prompt and padding positions and distinguishes an empty objective", () => {
	const loss = maskedNll(
		[0, 0.5, 0.25, Number.NaN],
		[false, true, true, false],
	);
	assert.equal(loss.count, 2);
	assert.ok(Math.abs((loss.perplexity ?? 0) - Math.sqrt(8)) < 1e-12);
	assert.equal(maskedNll([0.1], [false]).mean, null);
	assert.throws(() => maskedNll([0.5], []));
	assert.throws(() => maskedNll([0], [true]));
});

test("DPO has log(2) loss at the reference and decreases with relative preference", () => {
	assert.equal(dpoPairLoss(1, 1, 0.1), Math.log(2));
	assert.ok(dpoPairLoss(3, 1, 0.1) < dpoPairLoss(1, 1, 0.1));
	assert.ok(dpoPairLoss(-3, 1, 0.1) > dpoPairLoss(1, 1, 0.1));
	assert.ok(Number.isFinite(dpoPairLoss(-1e6, 0, 1)));
});

test("budget uses binary GiB and counts accumulation only in tokens, not live parameter states", () => {
	const budget = trainingBudget(500_000_000, 512, 1, 16, 1000);
	assert.equal(budget.tokens, 8_192_000);
	assert.equal(budget.stateGiB, 8_000_000_000 / 2 ** 30);
	assert.equal(
		trainingBudget(500_000_000, 512, 1, 32, 1000).stateGiB,
		budget.stateGiB,
	);
	assert.throws(() => trainingBudget(1, 0, 1, 1, 1));
});

test("per-document means can reverse model ranking on a fixed token set", () => {
	const a = [0.8, 0.8, ...Array<number>(8).fill(0.2)];
	const b = [0.5, 0.5, ...Array<number>(8).fill(0.3)];
	const score = (values: number[]): number =>
		maskedNll(
			values,
			values.map(() => true),
		).mean ?? 0;
	assert.ok(score(a) > score(b));
	assert.ok(
		(score(a.slice(0, 2)) + score(a.slice(2))) / 2 <
			(score(b.slice(0, 2)) + score(b.slice(2))) / 2,
	);
});
