/**
 * 站内链接审计脚本
 *
 * 扫描构建产物 dist/ 中的所有 HTML，收集站内链接（a[href]），
 * 检查每个目标是否真实存在，并把结果写成一份 JSON 报告。
 *
 * 用法：
 *   node scripts/audit-internal-links.js [distDir]
 *
 * 默认扫描 dist/。报告写到 stdout，同时把摘要打到 stderr，
 * 便于在流水线里直接查看或落盘。
 *
 * 检查规则：
 *   - 忽略外部链接（http/https/mailto/tel）与协议相对链接
 *   - 忽略 hash-only 链接（#anchor），它们在页面内部解析
 *   - 目录形式（/notes/）对应 dist/notes/index.html
 *   - 允许站点按既有约定重定向的 301 壳（读取 astro.config.mjs 的 redirects）
 *   - 目标带 hash 时，只检查文件是否存在，不校验锚点
 */

import fs from "node:fs/promises";
import path from "node:path";

const distDir = path.resolve(process.argv[2] ?? "dist");

/** 读取 astro.config.mjs 里声明的重定向路径，这些目标允许不存在。 */
async function readRedirects() {
	const configPath = path.resolve("astro.config.mjs");
	try {
		const source = await fs.readFile(configPath, "utf-8");
		const match = /redirects:\s*\{([\s\S]*?)\n\s*\}/.exec(source);
		if (!match) return new Set();
		const targets = new Set();
		for (const [, from, to] of match[1].matchAll(
			/["'](\/[^"']*)["']\s*:\s*["'](\/[^"']*)["']/g,
		)) {
			targets.add(from);
			void to;
		}
		return targets;
	} catch {
		return new Set();
	}
}

async function listHtmlFiles(dir) {
	const out = [];
	for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) out.push(...(await listHtmlFiles(full)));
		else if (entry.name.endsWith(".html")) out.push(full);
	}
	return out;
}

/** 把一个站内 URL 映射到 dist 里的文件路径。 */
function resolveTarget(urlPath) {
	// 产物里的中文路径通常是百分号编码的，先解码再落到文件路径上。
	const decoded = (() => {
		try {
			return decodeURIComponent(urlPath);
		} catch {
			return urlPath;
		}
	})();
	const clean = decoded.split("#")[0].split("?")[0];
	if (!clean || clean === "/") return path.join(distDir, "index.html");
	if (clean.endsWith("/")) return path.join(distDir, clean, "index.html");
	return path.join(distDir, clean);
}

const distExists = await fs
	.stat(distDir)
	.then(() => true)
	.catch(() => false);

if (!distExists) {
	console.error(`✗ 找不到构建产物目录：${distDir}（请先运行 pnpm build）`);
	process.exit(1);
}

const redirects = await readRedirects();
const htmlFiles = await listHtmlFiles(distDir);
const pagePaths = new Set(
	htmlFiles.map((file) => `/${path.relative(distDir, file).split(path.sep).join("/")}`),
);

const broken = [];
let checked = 0;

for (const file of htmlFiles) {
	const html = await fs.readFile(file, "utf-8");
	const from = `/${path.relative(distDir, file).split(path.sep).join("/")}`;
	for (const match of html.matchAll(/<a\b[^>]*\bhref=["']([^"']+)["']/gi)) {
		const href = match[1].trim();
		if (!href) continue;
		if (/^(https?:|mailto:|tel:|javascript:|\/\/)/i.test(href)) continue;
		if (href.startsWith("#")) continue;
		if (href.startsWith("/")) {
			checked += 1;
			const target = resolveTarget(href);
			const exists = await fs
				.stat(target)
				.then(() => true)
				.catch(() => false);
			const isRedirectShell = redirects.has(href);
			if (!exists && !isRedirectShell) {
				broken.push({ from, href, reason: "target missing" });
			}
			continue;
		}
		// 相对链接：相对于当前文件所在目录
		checked += 1;
		const relativeHref = (() => {
			try {
				return decodeURIComponent(href.split("#")[0]);
			} catch {
				return href.split("#")[0];
			}
		})();
		const target = path.resolve(path.dirname(file), relativeHref);
		const exists = await fs
			.stat(target)
			.then(() => true)
			.catch(() => false);
		if (!exists) broken.push({ from, href, reason: "relative target missing" });
	}
}

const report = {
	distDir,
	pages: htmlFiles.length,
	internalLinksChecked: checked,
	redirectShells: [...redirects],
	brokenCount: broken.length,
	broken,
	pagePathCount: pagePaths.size,
};

console.log(JSON.stringify(report, null, 2));

if (broken.length > 0) {
	console.error(
		`✗ 站内链接审计未通过：${broken.length} 个失效目标（检查 ${checked} 个站内链接）`,
	);
	process.exitCode = 1;
} else {
	console.error(`✓ 站内链接审计通过：${checked} 个站内链接全部有效（${htmlFiles.length} 个页面）`);
}
