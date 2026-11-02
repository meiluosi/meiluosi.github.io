/**
 * 数学公式与符号排版审计脚本
 *
 * 扫描 Markdown 正文中未经 LaTeX 包裹的裸露希腊字母、数学算子与伪下标公式。
 * 遵循 AGENTS.md「数学符号与公式规范」：数学变量与推导必须使用 LaTeX ($ ... $ 或 $$ ... $$)，
 * 以激活站内 InteractiveMathEnhancer 跨公式变量联动高亮并确保专业排版质感。
 *
 * 用法：
 *   node scripts/audit-math-notation.js
 *   node scripts/audit-math-notation.js --strict
 *   node scripts/audit-math-notation.js --baseline
 */

import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";

const CONTENT_DIRS = [
	"src/content/posts/**/*.md",
	"src/content/posts/**/*.mdx",
	"src/content/en-posts/**/*.md",
	"src/content/en-posts/**/*.mdx",
];

// 裸露的希腊字母与数学符号
const GREEK_MATH_REGEX = /[αβγδεθλμπστφψω]/u;
const MATH_OPERATOR_REGEX = /[∑∏√≠≈≤≥∇]/u;
const PSEUDO_SUBSCRIPT_REGEX = /\b([a-zA-Zπ]_[0-9a-zA-Zθ]+)\b/u;

async function auditFile(filePath) {
	const raw = await fs.readFile(filePath, "utf-8");
	const lines = raw.split("\n");
	const findings = [];

	let inFrontmatter = false;
	let frontmatterDone = false;
	let inCodeBlock = false;
	let inDisplayMath = false;

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const trimmed = line.trim();

		// Handle Frontmatter
		if (!frontmatterDone) {
			if (i === 0 && trimmed.startsWith("---")) {
				inFrontmatter = true;
				continue;
			}
			if (inFrontmatter && trimmed.startsWith("---")) {
				inFrontmatter = false;
				frontmatterDone = true;
				continue;
			}
			if (inFrontmatter) continue;
		}

		// Handle Fenced Code Blocks (``` or ~~~)
		if (trimmed.startsWith("```") || trimmed.startsWith("~~~")) {
			inCodeBlock = !inCodeBlock;
			continue;
		}
		if (inCodeBlock) continue;

		// Handle Display Math ($$)
		if (trimmed.startsWith("$$")) {
			// Check if single-line display math $$ ... $$
			if (trimmed.length > 2 && trimmed.endsWith("$$")) {
				continue;
			}
			inDisplayMath = !inDisplayMath;
			continue;
		}
		if (inDisplayMath) continue;

		// Clean line: strip inline code blocks `...` and inline math $...$
		// We replace them with spaces so line numbers and context remain clear
		let prose = line
			.replace(/`[^`]+`/g, " ") // strip inline code
			.replace(/\$[^$]+\$/g, " ") // strip inline math
			.replace(/<!--[\s\S]*?-->/g, " "); // strip HTML comments

		// Skip tables or markdown link URLs
		prose = prose.replace(/\]\([^\)]+\)/g, "]");

		// Check for naked Greek/math symbols
		const greekMatch = prose.match(GREEK_MATH_REGEX);
		const opMatch = prose.match(MATH_OPERATOR_REGEX);
		const subMatch = prose.match(PSEUDO_SUBSCRIPT_REGEX);

		if (greekMatch || opMatch || subMatch) {
			findings.push({
				line: i + 1,
				raw: line.trim(),
				char: greekMatch?.[0] || opMatch?.[0] || subMatch?.[0],
				type: greekMatch ? "希腊变量" : opMatch ? "数学算子" : "伪下标",
			});
		}
	}

	return findings;
}

async function main() {
	const strict = process.argv.includes("--strict");
	const useBaseline = process.argv.includes("--baseline");
	const baselinePath = "scripts/math-audit-baseline.json";
	const baseline = useBaseline ? JSON.parse(await fs.readFile(baselinePath, "utf8")) : null;
	const allowance = new Map();
	if (baseline) {
		for (const entry of baseline.entries) {
			const key = JSON.stringify([entry.file, entry.text]);
			if (allowance.has(key)) throw new Error(`重复基线项: ${entry.file}`);
			allowance.set(key, entry.count);
		}
	}
	const unapproved = [];
	const files = (
		await Promise.all(CONTENT_DIRS.map((pattern) => glob(pattern)))
	).flat();

	let totalIssues = 0;
	let fileCountWithIssues = 0;

	console.log(`🔍 正在审计 ${files.length} 篇正文的数学符号规范...`);

	for (const file of files) {
		const issues = await auditFile(file);
		if (baseline) {
			for (const issue of issues) {
				const key = JSON.stringify([file, issue.raw]);
				const remaining = allowance.get(key) || 0;
				if (remaining > 0) allowance.set(key, remaining - 1);
				else unapproved.push({ file, ...issue });
			}
		}
		if (issues.length > 0) {
			fileCountWithIssues++;
			totalIssues += issues.length;
			const rel = path.relative(process.cwd(), file);
			console.log(`\n📄 ${rel} (${issues.length} 处符号需优化):`);
			for (const issue of issues.slice(0, 5)) {
				console.log(
					`   第 ${issue.line} 行 [${issue.type}: ${issue.char}]: ${issue.raw.slice(0, 80)}`,
				);
			}
			if (issues.length > 5) {
				console.log(`   ... 及其余 ${issues.length - 5} 处`);
			}
		}
	}

	console.log("\n--------------------------------------------------");
	if (totalIssues === 0) {
		console.log("✅ 全站正文数学符号与公式规范审计通过！零裸露伪公式。");
	} else {
		console.log(
			`⚠️  发现 ${totalIssues} 处正文使用了裸露字符或伪公式，分布在 ${fileCountWithIssues} 篇文章中。`,
		);
		console.log(
			"💡 提示：使用标准行内 LaTeX ($ ... $) 可激活站内 InteractiveMathEnhancer 跨公式变量高亮能力。",
		);
		if (strict && !baseline) process.exitCode = 1;
	}
	if (baseline) {
		if (unapproved.length) {
			console.error(`\n❌ 发现 ${unapproved.length} 处未在基线中的疑似问题：`);
			for (const issue of unapproved) console.error(`${issue.file}:${issue.line} [${issue.type}] ${issue.raw}`);
			process.exitCode = 1;
		} else console.log("✅ 没有新增基线外疑似问题。");
		const stale = [...allowance.values()].reduce((sum, count) => sum + count, 0);
		if (stale) console.log(`ℹ️ ${stale} 条基线已不再出现，可审查后移除。`);
	}
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
