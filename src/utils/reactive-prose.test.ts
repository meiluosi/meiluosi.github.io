import assert from "node:assert/strict";
import test from "node:test";
import {
	computeBpeCost,
	computeDpoDynamics,
	computeReactiveProse,
} from "./reactive-prose";

test("reactive prose normalizes active nucleus probabilities to 1.0", () => {
	const state = computeReactiveProse(0.7, 0.9, 2048);
	const sum = state.tokens
		.filter((t) => t.inNucleus)
		.reduce((acc, t) => acc + t.prob, 0);
	assert.ok(
		Math.abs(sum - 1.0) < 1e-9,
		`Expected sum of active probs to be 1, got ${sum}`,
	);
});

test("low temperature sharply reduces entropy compared to high temperature", () => {
	const cold = computeReactiveProse(0.15, 1.0, 2048);
	const hot = computeReactiveProse(1.8, 1.0, 2048);
	assert.ok(
		cold.entropy < hot.entropy,
		`Cold entropy (${cold.entropy}) should be strictly less than hot entropy (${hot.entropy})`,
	);
	assert.ok(cold.tokens[0].prob > hot.tokens[0].prob);
});

test("KV cache calculation scales linearly with context length L", () => {
	const k4 = computeReactiveProse(0.8, 0.9, 4096);
	const k16 = computeReactiveProse(0.8, 0.9, 16384);
	assert.ok(Math.abs(k16.kvCacheGiB / k4.kvCacheGiB - 4) < 1e-4);
});

test("BPE cost model yields higher compression ratio with larger vocab", () => {
	const v8k = computeBpeCost(8000, 10000, 2048);
	const v64k = computeBpeCost(64000, 10000, 2048);
	assert.ok(v64k.compressionRatio > v8k.compressionRatio);
	assert.ok(v64k.tokensCount < v8k.tokensCount);
	assert.ok(v64k.embedMemoryMiB > v8k.embedMemoryMiB);
});

test("DPO dynamics increases implicit reward with larger positive margin", () => {
	const neutral = computeDpoDynamics(0.1, 0, 0);
	const positive = computeDpoDynamics(0.1, 2.0, -1.0);
	assert.ok(positive.implicitRewardMargin > neutral.implicitRewardMargin);
	assert.ok(positive.loss < neutral.loss);
	assert.ok(positive.gradientWeight < neutral.gradientWeight);
});
