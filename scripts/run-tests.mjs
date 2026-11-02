// Discover all source tests and run them with Node's test runner and tsx.
import { spawnSync } from "node:child_process";
import { globSync } from "glob";

const files = globSync("src/**/*.test.{js,mjs,cjs,ts,mts,cts}").sort();
if (files.length === 0) {
	console.error("No source tests found.");
	process.exit(1);
}

const result = spawnSync(process.execPath, ["--import", "tsx", "--test", ...files], {
	stdio: "inherit",
});
if (result.error) console.error(result.error);
process.exit(result.status ?? 1);
