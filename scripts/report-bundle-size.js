/**
 * 构建产物包体基线脚本
 *
 * 输入：构建产物目录（默认 dist/），需先 pnpm build。
 * 输出：stdout 的体积摘要（总量、按扩展名、最大的 15 个 JS/CSS/HTML/图片文件）；
 *       加 --json 输出 JSON，加 --out <file> 同时写入文件（不会写入 docs/，除非你显式指定）。
 * 只读、只记录：退出码恒为 0，不拦截构建。
 *
 * 用法：
 *   node scripts/report-bundle-size.js [distDir] [--json] [--out file]
 */

import fs from "node:fs/promises";
import path from "node:path";
import zlib from "node:zlib";

const args = process.argv.slice(2);
const asJson = args.includes("--json");
const outIdx = args.indexOf("--out");
const outFile = outIdx >= 0 ? args[outIdx + 1] : undefined;
const distArg = args.find(
	(a, i) => !a.startsWith("--") && args[i - 1] !== "--out",
);
const distDir = path.resolve(distArg ?? "dist");

async function* walk(dir) {
	for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) yield* walk(full);
		else yield full;
	}
}

const COMPRESSIBLE = new Set([".js", ".css", ".html", ".json", ".svg"]);
const files = [];
for await (const file of walk(distDir)) {
	const ext = path.extname(file).toLowerCase() || "(none)";
	const buf = await fs.readFile(file);
	files.push({
		file: path.relative(distDir, file),
		ext,
		bytes: buf.length,
		gzip: COMPRESSIBLE.has(ext) ? zlib.gzipSync(buf).length : buf.length,
	});
}

const byExt = {};
for (const f of files) {
	byExt[f.ext] ??= { count: 0, bytes: 0, gzip: 0 };
	byExt[f.ext].count += 1;
	byExt[f.ext].bytes += f.bytes;
	byExt[f.ext].gzip += f.gzip;
}

const largest = (exts) =>
	files
		.filter((f) => exts.includes(f.ext))
		.sort((a, b) => b.bytes - a.bytes)
		.slice(0, 15);

const report = {
	distDir,
	fileCount: files.length,
	totalBytes: files.reduce((n, f) => n + f.bytes, 0),
	byExt,
	largestJs: largest([".js"]),
	largestCss: largest([".css"]),
	largestHtml: largest([".html"]),
	largestImages: largest([".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".svg"]),
};

const kb = (n) => `${(n / 1024).toFixed(1)} KB`;
if (outFile) await fs.writeFile(outFile, `${JSON.stringify(report, null, 2)}\n`);
if (asJson) {
	console.log(JSON.stringify(report, null, 2));
} else {
	console.log(`${report.fileCount} 个文件 · 共 ${kb(report.totalBytes)}`);
	console.log("\n按扩展名：");
	for (const [ext, v] of Object.entries(byExt).sort((a, b) => b[1].bytes - a[1].bytes).slice(0, 10)) {
		console.log(`  ${ext.padEnd(8)} ${String(v.count).padStart(5)} 个  ${kb(v.bytes).padStart(12)}  gzip ${kb(v.gzip)}`);
	}
	for (const [title, list] of [["最大的 JS", report.largestJs], ["最大的 CSS", report.largestCss], ["最大的 HTML", report.largestHtml], ["最大的图片", report.largestImages]]) {
		console.log(`\n${title}：`);
		for (const f of list.slice(0, 8)) console.log(`  ${kb(f.bytes).padStart(12)}  gzip ${kb(f.gzip).padStart(10)}  ${f.file}`);
	}
}
