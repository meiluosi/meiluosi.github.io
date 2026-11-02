/**
 * 缺英文版文章统计脚本
 *
 * 输入：src/content/posts（中文）与 src/content/en-posts（英文）的 frontmatter。
 * 配对规则：英文文章的 sourceId 等于中文文件名（不含扩展名）；Astro 的条目 id 会转成小写，
 *          所以两边都按小写比较。
 * 输出：stdout 的统计与缺口清单；加 --json 输出机器可读 JSON。
 * 是否写入 docs/：否。只读，不改任何文件，退出码恒为 0（只记录，不拦截）。
 *
 * 用法：
 *   node scripts/report-missing-translations.js
 *   node scripts/report-missing-translations.js --json
 *   node scripts/report-missing-translations.js --by-category
 */

import fs from "node:fs/promises";
import path from "node:path";

const ZH_DIR = "src/content/posts";
const EN_DIR = "src/content/en-posts";
const asJson = process.argv.includes("--json");
const byCategory = process.argv.includes("--by-category");

/** 读取 frontmatter 的单行标量字段；不做完整 YAML 解析。 */
function field(front, key) {
	const m = front.match(new RegExp(`^${key}:\\s*(.+?)\\s*$`, "m"));
	return m ? m[1].replace(/^["']|["']$/g, "") : undefined;
}

async function readDir(dir) {
	const names = (await fs.readdir(dir)).filter((n) => /\.(md|mdx)$/.test(n));
	const items = [];
	for (const name of names) {
		const raw = await fs.readFile(path.join(dir, name), "utf-8");
		const end = raw.indexOf("\n---", 3);
		const front = raw.startsWith("---") && end !== -1 ? raw.slice(4, end) : "";
		items.push({
			file: path.join(dir, name),
			id: name.replace(/\.(md|mdx)$/, ""),
			title: field(front, "title"),
			category: field(front, "category") ?? "(未分类)",
			draft: field(front, "draft") === "true",
			sourceId: field(front, "sourceId"),
		});
	}
	return items;
}

const zh = (await readDir(ZH_DIR)).filter((p) => !p.draft);
const en = (await readDir(EN_DIR)).filter((p) => !p.draft);

const norm = (id) => id?.normalize("NFC").toLowerCase().replace(/\s+/g, "-");
const zhIds = new Set(zh.map((p) => norm(p.id)));
const translated = new Set(en.map((p) => norm(p.sourceId)).filter(Boolean));
const missing = zh.filter((p) => !translated.has(norm(p.id)));
const enOnly = en.filter((p) => !p.sourceId || !zhIds.has(norm(p.sourceId)));

const report = {
	zhCount: zh.length,
	enCount: en.length,
	pairedCount: zh.length - missing.length,
	missingCount: missing.length,
	coverage: Number(((zh.length - missing.length) / zh.length).toFixed(3)),
	enWithoutZhSource: enOnly.map((p) => p.file),
	missing: missing.map((p) => ({
		id: p.id,
		title: p.title,
		category: p.category,
	})),
};

if (asJson) {
	console.log(JSON.stringify(report, null, 2));
} else {
	console.log(
		`中文 ${report.zhCount} 篇 · 英文 ${report.enCount} 篇 · 已配对 ${report.pairedCount} · 缺英文版 ${report.missingCount} · 覆盖率 ${(report.coverage * 100).toFixed(1)}%`,
	);
	if (enOnly.length) {
		console.log(`\n英文文章无对应中文来源（sourceId 缺失或不匹配）：${enOnly.length}`);
		for (const p of enOnly) console.log(`  - ${p.file}`);
	}
	if (byCategory) {
		const groups = Map.groupBy(missing, (p) => p.category);
		console.log("\n按分类的缺口：");
		for (const [cat, list] of [...groups].sort((a, b) => b[1].length - a[1].length)) {
			console.log(`  ${cat}: ${list.length}`);
		}
	} else {
		console.log("\n缺英文版清单：");
		for (const p of missing) console.log(`  - ${p.id}  ${p.title ?? ""}`);
	}
}
