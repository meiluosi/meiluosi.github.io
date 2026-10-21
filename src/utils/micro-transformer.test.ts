import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_VOCAB, runForwardPass } from "./micro-transformer";

test("MicroTransformer runs a forward pass over valid tokens", () => {
	const trace = runForwardPass(["[BOS]", "the", "cat"]);
	assert.equal(trace.tokens.length, 3);
	assert.equal(trace.tokenIds.length, 3);
	assert.equal(trace.embeddings.length, 3);
	assert.equal(trace.embeddings[0].length, 16); // dModel

	// Heads
	assert.equal(trace.heads.length, 2);
	assert.equal(trace.heads[0].attnWeights.length, 3);
	assert.equal(trace.heads[0].attnWeights[0].length, 3);

	// Causal mask: future tokens should be 0 probability
	assert.equal(trace.heads[0].attnWeights[0][1], 0);
	assert.equal(trace.heads[0].attnWeights[0][2], 0);
	assert.equal(trace.heads[0].attnWeights[1][2], 0);

	// First token can only attend to itself, so weight must be 1.0
	assert.ok(Math.abs(trace.heads[0].attnWeights[0][0] - 1.0) < 1e-4);

	// Top candidates
	assert.ok(trace.topCandidates.length > 0);
	assert.ok(trace.probabilities.length === DEFAULT_VOCAB.length);

	// Sum of probabilities should be close to 1
	const sumProb = trace.probabilities.reduce((a, b) => a + b, 0);
	assert.ok(Math.abs(sumProb - 1.0) < 1e-3);
});

test("MicroTransformer respects head ablation", () => {
	const normalTrace = runForwardPass(["[BOS]", "the", "cat"]);
	const ablatedTrace = runForwardPass(["[BOS]", "the", "cat"], {
		ablateHeads: [0],
	});

	assert.equal(normalTrace.heads[0].isAblated, false);
	assert.equal(ablatedTrace.heads[0].isAblated, true);

	// Ablated head output should be zeroes
	for (const row of ablatedTrace.heads[0].headOutput) {
		for (const val of row) {
			assert.equal(val, 0);
		}
	}

	// Probabilities should differ due to missing head circuit
	const diff = Math.abs(
		normalTrace.probabilities[0] - ablatedTrace.probabilities[0],
	);
	assert.ok(diff >= 0);
});

test("MicroTransformer generates higher entropy with higher temperature", () => {
	const lowTemp = runForwardPass(["[BOS]", "the"], { temperature: 0.2 });
	const highTemp = runForwardPass(["[BOS]", "the"], { temperature: 2.0 });

	assert.ok(
		highTemp.entropy > lowTemp.entropy,
		`Expected highTemp entropy (${highTemp.entropy}) > lowTemp entropy (${lowTemp.entropy})`,
	);
});

test("MicroTransformer computes attention rays and supports hallucination perturbation", () => {
	const trace = runForwardPass(["[BOS]", "the", "cat"], {
		hallucinationMode: true,
	});
	assert.ok(trace.hallucinationMode === true);
	assert.ok(trace.attentionRays.length > 0);

	// For 3 tokens (indices 0, 1, 2), causal pairs: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2) = 6 rays
	assert.equal(trace.attentionRays.length, 6);

	const lastRay = trace.attentionRays[trace.attentionRays.length - 1];
	assert.equal(lastRay.fromIdx, 2);
	assert.equal(lastRay.toIdx, 2);
	assert.ok(lastRay.avgWeight >= 0);
});
