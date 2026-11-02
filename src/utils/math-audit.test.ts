import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

const audit = path.resolve("scripts/audit-math-notation.js");

test("math audit allows only the recorded occurrences, independent of line numbers", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "math-audit-"));
	try {
		mkdirSync(path.join(cwd, "scripts"), { recursive: true });
		mkdirSync(path.join(cwd, "src/content/posts"), { recursive: true });
		const file = "src/content/posts/example.md";
		const text = "旧问题 θ=1";
		writeFileSync(
			path.join(cwd, "scripts/math-audit-baseline.json"),
			JSON.stringify({
				version: 1,
				entries: [{ file, text, count: 1, category: "待修正的数学记法" }],
			}),
		);
		const run = () =>
			spawnSync(process.execPath, [audit, "--baseline"], {
				cwd,
				encoding: "utf8",
			});
		writeFileSync(path.join(cwd, file), `# Title\n\n${text}\n`);
		assert.equal(run().status, 0);
		writeFileSync(path.join(cwd, file), `# Title\n\nNew paragraph.\n${text}\n`);
		assert.equal(run().status, 0);
		writeFileSync(path.join(cwd, file), `${text}\n${text}\n`);
		assert.equal(run().status, 1);
		writeFileSync(path.join(cwd, file), `${text}\n新增 λ=1\n`);
		assert.equal(run().status, 1);
	} finally {
		rmSync(cwd, { recursive: true, force: true });
	}
});
