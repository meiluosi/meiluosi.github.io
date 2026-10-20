import assert from "node:assert/strict";
import test from "node:test";
import { computeRopeSnapshot } from "./rope-attention";

test("RoPE inner product is strictly translation-invariant (depends only on m - n)", () => {
	const snapA = computeRopeSnapshot(12, 5, 100, 2, "standard");
	const snapB = computeRopeSnapshot(42, 35, 100, 2, "standard");

	assert.equal(snapA.relativeDistance, snapB.relativeDistance);
	assert.ok(Math.abs(snapA.aggregateKernel - snapB.aggregateKernel) < 1e-12);
	for (let k = 0; k < snapA.planes.length; k++) {
		assert.ok(
			Math.abs(
				snapA.planes[k].cosineContribution - snapB.planes[k].cosineContribution,
			) < 1e-12,
		);
	}
});

test("zero relative distance yields maximum kernel score of 1.0", () => {
	const snap = computeRopeSnapshot(19, 19, 100, 2, "ntk-aware");
	assert.equal(snap.relativeDistance, 0);
	assert.ok(Math.abs(snap.aggregateKernel - 1.0) < 1e-12);
});

test("NTK-aware scaling preserves highest-frequency plane while stretching lowest-frequency wavelength", () => {
	const std = computeRopeSnapshot(10, 2, 100, 4, "standard");
	const pi = computeRopeSnapshot(10, 2, 100, 4, "linear-pi");
	const ntk = computeRopeSnapshot(10, 2, 100, 4, "ntk-aware");

	// Plane 0 (k=0) has θ_0 = 1 in both standard and NTK-aware, whereas linear PI divides it by scaleFactor
	assert.ok(Math.abs(ntk.planes[0].theta - std.planes[0].theta) < 1e-9);
	assert.ok(pi.planes[0].theta < std.planes[0].theta);

	// Lowest frequency plane (k=5) has longer wavelength under NTK-aware than standard
	assert.ok(ntk.planes[5].wavelength > std.planes[5].wavelength);
});
