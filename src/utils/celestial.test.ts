import assert from "node:assert/strict";
import test from "node:test";
import { computeCelestialState } from "./celestial";

test("computes correct celestial period across 24 hours", () => {
	assert.equal(computeCelestialState(6.5).period, "dawn");
	assert.equal(computeCelestialState(12.0).period, "noon");
	assert.equal(computeCelestialState(18.0).period, "sunset");
	assert.equal(computeCelestialState(23.5).period, "night");
	assert.equal(computeCelestialState(2.0).period, "night");
});

test("noon solar intensity is strictly higher than midnight intensity", () => {
	const noon = computeCelestialState(12.0);
	const night = computeCelestialState(0.0);
	assert.ok(noon.sunIntensity > night.sunIntensity);
	assert.ok(noon.sunPosition[1] > night.sunPosition[1]);
});

test("hours wrap seamlessly around 24-hour clock", () => {
	const h1 = computeCelestialState(25.5);
	const h2 = computeCelestialState(1.5);
	assert.equal(h1.period, h2.period);
	assert.ok(Math.abs(h1.hour - h2.hour) < 1e-9);
});
