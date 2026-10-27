import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// Constructed hypotheses, not trained networks or a reproduction of SCAN.
const cases = ["walk", "run", "jump"].flatMap((verb) =>
	[1, 2, 3].map((count) => ({ verb, count })),
);
const training = cases.filter((x) => x.verb !== "jump" || x.count === 1);
const heldout = cases.filter((x) => !training.includes(x));
const compositional = ({ verb, count }) => Array(count).fill(verb).join(" ");
const restricted = (x) =>
	compositional({ ...x, count: x.verb === "jump" ? 1 : x.count });
const accuracy = (h, rows) =>
	rows.filter((x) => h(x) === compositional(x)).length / rows.length;
assert.equal(accuracy(compositional, training), 1);
assert.equal(accuracy(restricted, training), 1);
assert.equal(accuracy(compositional, heldout), 1);
assert.equal(accuracy(restricted, heldout), 0);
// Exact deterministic A -> B -> terminal, gamma=1, reward=1 on B -> terminal.
// This episodic teaching case is not the discounted Q-learning convergence theorem.
const values = { A: 0, B: 0 };
const trace = [];
for (let episode = 1; episode <= 30; episode++) {
	values.A += 0.5 * (values.B - values.A);
	values.B += 0.5 * (1 - values.B);
	if ([1, 2, 5, 10, 30].includes(episode)) trace.push({ episode, ...values });
}
assert.ok(Math.abs(values.A - 1) < 1e-6);
assert.ok(Math.abs(values.B - 1) < 1e-6);
// Entropy from exact joint tables, not Monte Carlo estimates or causal interventions.
const entropy = (rows, keys) => {
	const counts = new Map();
	for (const row of rows) {
		const key = JSON.stringify(keys.map((k) => row[k]));
		counts.set(key, (counts.get(key) ?? 0) + 1);
	}
	return [...counts.values()].reduce((h, n) => {
		const p = n / rows.length;
		return h - p * Math.log2(p);
	}, 0);
};
const mi = (rows, a, b) =>
	entropy(rows, a) + entropy(rows, b) - entropy(rows, [...a, ...b]);
const cmi = (rows, x, y, z) =>
	entropy(rows, [...x, ...z]) +
	entropy(rows, [...y, ...z]) -
	entropy(rows, z) -
	entropy(rows, [...x, ...y, ...z]);
const information = ["world", "action", "xor"].map((mode) => {
	const rows = [0, 1].flatMap((W) =>
		[0, 1].map((A) => ({
			W,
			A,
			next: mode === "world" ? W : mode === "action" ? A : W ^ A,
		})),
	);
	return {
		mode,
		worldInformation: mi(rows, ["next"], ["W"]),
		actionInformation: mi(rows, ["next"], ["A"]),
		jointInformation: mi(rows, ["next"], ["W", "A"]),
		conditionalWorldInformation: cmi(rows, ["next"], ["W"], ["A"]),
	};
});
assert.equal(information[2].worldInformation, 0);
assert.equal(information[2].actionInformation, 0);
assert.equal(information[2].jointInformation, 1);
assert.equal(information[2].conditionalWorldInformation, 1);
const result = {
	basis: "verified-by-local-demo",
	scope: "本站构造的教学算例，非论文复现，非真实系统实验。",
	generalization: {
		trainingCount: training.length,
		heldoutCount: heldout.length,
		compositional: { training: 1, heldout: 1 },
		restricted: { training: 1, heldout: 0 },
		limitation:
			"两个规则由我们显式指定，没有训练模型；演示训练样例不能唯一决定外推规则。",
	},
	temporalDifference: {
		alpha: 0.5,
		gamma: 1,
		trace,
		analyticTarget: { A: 1, B: 1 },
		limitation: "确定性两状态预测例，不检验函数逼近、随机采样或策略学习。",
	},
	information: {
		units: "bits",
		rows: information,
		limitation:
			"独立公平二元变量的精确枚举；条件互信息不等于世界独有贡献，也不是干预因果量。",
	},
};
const target = path.join(
	root,
	"docs/intelligence-history/round-two/evidence/demos.json",
);
const output = JSON.stringify(result, null, 2) + "\n";
if (process.argv.includes("--check"))
	assert.equal(fs.readFileSync(target, "utf8"), output, "算例输出需要重新生成");
else fs.writeFileSync(target, output);
console.log("三个教学算例的独立预期检查通过。");
