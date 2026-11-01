import assert from "node:assert/strict";
import test from "node:test";
import {
	CANDIDATES,
	computeAttention,
	DEFAULT_SAMPLING,
	empiricalDotVariance,
	FFN_ACTIVATIONS,
	HEAD1,
	HEAD2,
	HERO,
	KV_CACHE_TABLE,
	LN_1,
	LOGITS,
	RESIDUAL_1,
	SQRT_DK,
	sample,
	TOKENS,
	X_MATRIX,
	Z_CONCAT,
} from "./modelData";
import { dot, softmax } from "./tensor";

const close = (a: number, b: number, eps = 1e-9): void => {
	assert.ok(Math.abs(a - b) < eps, `${a} should equal ${b}`);
};

test("scores are q·k / √d_k computed from Q and K", () => {
	for (const h of [HEAD1, HEAD2]) {
		h.K.forEach((k, j) => {
			close(h.rawScores[j], dot(h.Q, k));
			close(h.scores[j], dot(h.Q, k) / SQRT_DK);
		});
	}
});

test("weights are softmax(scores) and sum to 1", () => {
	for (const h of [HEAD1, HEAD2]) {
		const w = softmax(h.scores);
		h.weights.forEach((v, j) => {
			close(v, w[j]);
		});
		close(
			h.weights.reduce((a, b) => a + b, 0),
			1,
		);
	}
});

test("narrative: head 1 focuses 能够, head 2 focuses 智能", () => {
	assert.equal(HEAD1.focusTarget, "能够");
	assert.equal(HEAD2.focusTarget, "智能");
});

test("Z = Σ α_j V_j and concat has 4 dims", () => {
	for (const h of [HEAD1, HEAD2]) {
		for (let d = 0; d < 2; d++) {
			close(
				h.Z[d],
				h.V.reduce((acc, v, j) => acc + h.weights[j] * v[d], 0),
			);
		}
	}
	assert.equal(Z_CONCAT.length, 4);
});

test("causal mask zeroes every future position exactly", () => {
	for (let q = 0; q < TOKENS.length; q++) {
		for (const h of [0, 1]) {
			const { weights } = computeAttention(h, q);
			for (let j = q + 1; j < TOKENS.length; j++) assert.equal(weights[j], 0);
			close(
				weights.reduce((a, b) => a + b, 0),
				1,
			);
		}
	}
});

test("LayerNorm output has zero mean and unit variance", () => {
	const mean = LN_1.reduce((a, b) => a + b, 0) / 4;
	const variance = LN_1.reduce((a, b) => a + (b - mean) ** 2, 0) / 4;
	close(mean, 0, 1e-9);
	close(variance, 1, 1e-3);
	assert.equal(RESIDUAL_1.length, 4);
	assert.equal(X_MATRIX[HERO].length, 4);
});

test("FFN has 8 neurons and logits rank 考 first", () => {
	assert.equal(FFN_ACTIVATIONS.length, 8);
	assert.equal(CANDIDATES[0].word, "考");
	assert.ok(LOGITS[0] > LOGITS[1]);
});

test("sampling probabilities are consistent", () => {
	close(
		DEFAULT_SAMPLING.probs.reduce((a, b) => a + b, 0),
		1,
	);
	close(
		DEFAULT_SAMPLING.renormalized.reduce((a, b) => a + b, 0),
		1,
	);
	// 低温趋向 argmax，高温熵更大
	assert.ok(sample(0.1, 1).probs[0] > 0.99);
	assert.ok(sample(2, 1).entropy > sample(0.7, 1).entropy);
});

test("KV cache rows equal the head-1 K/V of each prefix token", () => {
	TOKENS.forEach((_, i) => {
		assert.deepEqual(KV_CACHE_TABLE[i].k, HEAD1.K[i]);
		assert.deepEqual(KV_CACHE_TABLE[i].v, HEAD1.V[i]);
	});
	assert.equal(KV_CACHE_TABLE.at(-1)?.isNew, true);
});

test("√d_k scaling restores unit variance (empirically)", () => {
	const raw = empiricalDotVariance(64, false, 2000);
	const scaled = empiricalDotVariance(64, true, 2000);
	assert.ok(raw.variance > 40 && raw.variance < 90);
	assert.ok(scaled.variance > 0.7 && scaled.variance < 1.4);
});
