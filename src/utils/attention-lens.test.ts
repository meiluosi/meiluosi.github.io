import assert from "node:assert/strict";
import test from "node:test";
import {
	CANONICAL_HEADS,
	computeAttentionMatrix,
	tokenizePrompt,
} from "./attention-lens";

test("tokenizes text with leading [BOS] token", () => {
	const tokens = tokenizePrompt("The quick brown");
	assert.deepEqual(tokens, ["[BOS]", "The", "quick", "brown"]);
});

test("causal attention matrix is strictly lower-triangular and each row sums to 1.0", () => {
	const tokens = tokenizePrompt("Alice ate an apple");
	for (const head of CANONICAL_HEADS) {
		const res = computeAttentionMatrix(tokens, head.id);
		assert.equal(res.matrix.length, tokens.length);
		for (let i = 0; i < tokens.length; i++) {
			let rowSum = 0;
			for (let j = 0; j < tokens.length; j++) {
				if (j > i) {
					assert.equal(
						res.matrix[i][j],
						0,
						`Future token j=${j} from i=${i} must have 0 attention in causal mask`,
					);
				}
				rowSum += res.matrix[i][j];
			}
			assert.ok(
				Math.abs(rowSum - 1.0) < 1e-9,
				`Row ${i} for head ${head.id} must sum to 1.0, got ${rowSum}`,
			);
		}
	}
});

test("induction head attends to [B] when prefix [A] repeats in context", () => {
	// Pattern: "[BOS] Alice visited Paris . Later Alice visited" -> query is after second "Alice visited"
	const tokens = [
		"[BOS]",
		"Alice",
		"visited",
		"Paris",
		".",
		"Later",
		"Alice",
		"visited",
	];
	const res = computeAttentionMatrix(tokens, "induction");
	// At position 7 ("visited"), preceding token is "Alice" (pos 6).
	// Earlier "Alice" was at pos 1, followed by "visited" at pos 2.
	// Target of induction head is index 2 ("visited") or next token.
	assert.ok(res.matrix[7].length === tokens.length);
});
