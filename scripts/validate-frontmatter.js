/**
 * 内容 frontmatter 校验脚本
 *
 * 在 pnpm build 流程中、astro build 之前运行。目的不是替代 Astro 的 schema
 * 校验，而是把「YAML 解析失败」这类错误变成一条可读的提示。
 *
 * 背景：未加引号的 title / description 里如果含有第二个冒号，YAML 会把它
 * 当成嵌套映射来解析，astro 随后报 "bad indentation of a mapping entry"，
 * 位置只指向某一列，很难看出是哪一篇。批量写作英文概览时这个问题出现过多次，
 * 所以在构建前主动拦下来。
 *
 * 用法：
 *   node scripts/validate-frontmatter.js
 *
 * 退出码非 0 表示校验失败，构建应当终止。
 */

import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";

const CONTENT_DIRS = [
	"src/content/posts",
	"src/content/en-posts",
	"src/content/spec",
];

/** 取出一篇内容文件的 frontmatter 原文；不是 Markdown 文档则返回 null。 */
async function readFrontmatter(file) {
	const raw = await fs.readFile(file, "utf-8");
	if (!raw.startsWith("---")) return null;
	const end = raw.indexOf("\n---", 3);
	if (end === -1) return null;
	return raw.slice(raw.indexOf("\n") + 1, end);
}

/**
 * 极简 YAML 标量检查：找出那些「未加引号却含有第二个冒号」的行。
 * 这不是完整 YAML 解析器，只负责定位最常见的一类错误。
 */
function findUnquotedColon(line) {
	const match = /^([A-Za-z_][\w-]*):\s+(.*)$/.exec(line);
	if (!match) return false;
	const value = match[2].trim();
	if (!value) return false;
	if (value.startsWith('"') || value.startsWith("'")) return false;
	if (value.startsWith("[") || value.startsWith("{")) return false;
	// 只有一个冒号说明这是普通标量；出现第二个冒号才会被 YAML 误读。
	return value.includes(": ");
}

async function main() {
	const problems = [];
	let checked = 0;

	for (const dir of CONTENT_DIRS) {
		const files = await glob(`${dir}/**/*.{md,mdx}`, { nodir: true });
		for (const file of files.sort()) {
			const frontmatter = await readFrontmatter(file);
			if (frontmatter === null) continue;
			checked += 1;
			const lines = frontmatter.split("\n");
			for (let i = 0; i < lines.length; i += 1) {
				if (findUnquotedColon(lines[i])) {
					problems.push(
						`${path.relative(process.cwd(), file)}:${i + 2} 未加引号的值里含有冒号：${lines[i].trim()}`,
					);
				}
			}
		}
	}

	if (problems.length > 0) {
		console.error("✗ frontmatter 校验未通过：");
		for (const problem of problems) console.error(`  - ${problem}`);
		console.error(
			"\n提示：给 title / description 等标量字段加双引号即可（值内部的 \" 需要转义）。",
		);
		process.exitCode = 1;
		return;
	}

	console.log(`✓ frontmatter 校验通过（${checked} 个内容文件）`);
}

await main();
