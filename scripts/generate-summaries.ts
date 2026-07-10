/**
 * AI 文章摘要生成脚本
 *
 * 在 pnpm build 流程中运行，对每篇文章调用 LLM 生成一句话摘要，
 * 将摘要写入 frontmatter 的 aiSummary 字段。
 *
 * 用法：
 *   npx tsx scripts/generate-summaries.ts
 *
 * 环境变量：
 *   AI_SUMMARY_API_KEY — LLM API Key（必填）
 *   AI_SUMMARY_API_URL — API 地址（默认 https://api.deepseek.com/v1/chat/completions）
 *   AI_SUMMARY_MODEL   — 模型名称（默认 deepseek-chat）
 *
 * 特性：
 *   - 增量生成：跳过已有 aiSummary 的文章（除非设置 AI_SUMMARY_FORCE=true）
 *   - 按需生成：只处理最近 N 篇文章（默认 50 篇，通过 AI_SUMMARY_LIMIT 设置）
 *   - 错误重试：单篇失败不影响其他文章
 */

import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";

// ========== 配置 ==========

const POSTS_DIR = "src/content/posts";
const API_KEY = process.env.AI_SUMMARY_API_KEY;
const API_URL =
	process.env.AI_SUMMARY_API_URL ||
	"https://api.deepseek.com/v1/chat/completions";
const MODEL = process.env.AI_SUMMARY_MODEL || "deepseek-chat";
const FORCE = process.env.AI_SUMMARY_FORCE === "true";
const LIMIT = Number.parseInt(process.env.AI_SUMMARY_LIMIT || "50", 10);
const DRY_RUN = process.env.AI_SUMMARY_DRY_RUN === "true";

// ========== 类型 ==========

interface PostFrontmatter {
	title?: string;
	description?: string;
	aiSummary?: string;
	[key: string]: unknown;
}

interface ProcessResult {
	file: string;
	status: "skipped" | "generated" | "error";
	summary?: string;
	error?: string;
}

// ========== 辅助函数 ==========

/**
 * 解析 Markdown 文件的 frontmatter 和正文
 */
function parseFrontmatter(content: string): {
	frontmatter: string;
	body: string;
	data: PostFrontmatter;
} {
	const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
	if (!match) {
		return { frontmatter: "", body: content, data: {} };
	}

	const raw = match[1];
	const body = match[2];
	const data: PostFrontmatter = {};

	// 简易 YAML 解析（处理数组）
	const lines = raw.split("\n");
	let currentKey = "";
	let inArray = false;

	for (const line of lines) {
		const trimmed = line.trim();

		if (inArray) {
			if (trimmed.startsWith("- ")) {
				const arr = (data[currentKey] as string[]) || [];
				arr.push(trimmed.slice(2).trim());
				data[currentKey] = arr;
			} else if (trimmed === "") {
				continue;
			} else {
				inArray = false;
			}
		}

		if (!inArray) {
			const colonIdx = line.indexOf(":");
			if (colonIdx > 0) {
				const key = line.slice(0, colonIdx).trim();
				const value = line.slice(colonIdx + 1).trim();

				if (value === "") {
					currentKey = key;
					inArray = true;
				} else {
					// 去掉引号
					data[key] = value.replace(/^["']|["']$/g, "");
				}
			}
		}
	}

	return { frontmatter: raw, body, data };
}

/**
 * 截取文章前 N 个字符用于摘要生成
 */
function truncateContent(content: string, maxChars: number): string {
	// 移除 Markdown 语法噪音
	let cleaned = content
		.replace(/^#{1,6}\s+/gm, "") // 标题
		.replace(/\*\*([^*]+)\*\*/g, "$1") // 加粗
		.replace(/\*([^*]+)\*/g, "$1") // 斜体
		.replace(/`([^`]+)`/g, "$1") // 行内代码
		.replace(/```[\s\S]*?```/g, "[代码块]") // 代码块
		.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // 链接
		.replace(/!\[([^\]]*)\]\([^)]+\)/g, "[图片]") // 图片
		.replace(/>\s+/gm, "") // 引用
		.replace(/---+/g, "") // 分隔线
		.replace(/\n{3,}/g, "\n\n") // 多空行
		.trim();

	if (cleaned.length <= maxChars) return cleaned;
	return cleaned.slice(0, maxChars) + "...";
}

/**
 * 调用 LLM API 生成摘要
 */
async function generateSummary(
	title: string,
	description: string,
	content: string,
): Promise<string> {
	const prompt = `请为以下技术博客文章生成一句话中文摘要（30-50字），概括文章的核心内容和技术要点：

标题：${title}
简介：${description || "无"}

正文（节选）：
${truncateContent(content, 2000)}

请直接输出摘要，不要包含任何前缀或解释。`;

	const response = await fetch(API_URL, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Authorization: `Bearer ${API_KEY}`,
		},
		body: JSON.stringify({
			model: MODEL,
			messages: [
				{
					role: "system",
					content:
						"你是一个技术博客编辑，擅长用简洁的语言概括技术文章的核心内容。",
				},
				{ role: "user", content: prompt },
			],
			max_tokens: 200,
			temperature: 0.3,
		}),
	});

	if (!response.ok) {
		const err = await response.text().catch(() => "");
		throw new Error(`API 请求失败 (${response.status}): ${err.slice(0, 200)}`);
	}

	const data = (await response.json()) as {
		choices?: { message?: { content?: string } }[];
	};

	const summary = data.choices?.[0]?.message?.content?.trim() || "";
	if (!summary) {
		throw new Error("API 返回了空的摘要");
	}

	return summary;
}

// ========== 主流程 ==========

async function main() {
	// 检查 API Key
	if (!API_KEY) {
		console.log(
			"⏭ 跳过 AI 摘要生成：未设置 AI_SUMMARY_API_KEY 环境变量",
		);
		console.log("   如需启用，请设置环境变量后重新运行构建。");
		process.exit(0);
	}

	console.log("🤖 AI 文章摘要生成器");
	console.log(`   API: ${API_URL}`);
	console.log(`   模型: ${MODEL}`);
	console.log(`   强制重新生成: ${FORCE}`);
	console.log(`   最多处理: ${LIMIT} 篇`);
	console.log(`   试运行: ${DRY_RUN}\n`);

	// 查找所有文章
	const postFiles = await glob("**/*.md", { cwd: POSTS_DIR });
	const sortedFiles = postFiles.sort().reverse(); // 最新的在前面
	const filesToProcess = sortedFiles.slice(0, LIMIT);

	console.log(`   找到 ${postFiles.length} 篇文章，处理最近 ${filesToProcess.length} 篇\n`);

	const results: ProcessResult[] = [];

	for (const file of filesToProcess) {
		const filePath = path.join(POSTS_DIR, file);
		const content = await fs.readFile(filePath, "utf-8");
		const { frontmatter, body, data } = parseFrontmatter(content);

		// 跳过已有摘要的文章
		if (data.aiSummary && !FORCE) {
			console.log(`   ⏭ ${file} — 已有摘要`);
			results.push({ file, status: "skipped" });
			continue;
		}

		// 跳过没有标题的文章
		if (!data.title) {
			console.log(`   ⚠ ${file} — 缺少标题`);
			results.push({ file, status: "error", error: "缺少标题" });
			continue;
		}

		console.log(`   🔄 ${file} — 生成中...`);

		try {
			const summary = await generateSummary(
				data.title,
				data.description || "",
				body,
			);

			console.log(`   ✅ ${file} — ${summary}`);

			if (!DRY_RUN) {
				// 更新 frontmatter
				let newContent: string;
				if (frontmatter) {
					// 在 frontmatter 末尾添加 aiSummary
					const newFrontmatter = frontmatter.includes("aiSummary:")
						? frontmatter.replace(
								/^aiSummary:.*$/m,
								`aiSummary: "${summary.replace(/"/g, '\\"')}"`,
							)
						: `${frontmatter}\naiSummary: "${summary.replace(/"/g, '\\"')}"`;
					newContent = content.replace(frontmatter, newFrontmatter);
				} else {
					newContent = `---\naiSummary: "${summary.replace(/"/g, '\\"')}"\n---\n${body}`;
				}

				await fs.writeFile(filePath, newContent, "utf-8");
			}

			results.push({ file, status: "generated", summary });
		} catch (error: any) {
			console.error(`   ❌ ${file} — ${error.message}`);
			results.push({
				file,
				status: "error",
				error: error.message,
			});
		}

		// API 请求间隔
		await new Promise((r) => setTimeout(r, 500));
	}

	// 汇总
	const generated = results.filter((r) => r.status === "generated").length;
	const skipped = results.filter((r) => r.status === "skipped").length;
	const errors = results.filter((r) => r.status === "error").length;

	console.log(`\n📊 汇总：`);
	console.log(`   ✅ 新生成: ${generated} 篇`);
	console.log(`   ⏭ 跳过: ${skipped} 篇`);
	console.log(`   ❌ 失败: ${errors} 篇`);

	if (DRY_RUN) {
		console.log(`\n💡 这是试运行，未实际修改文件。`);
		console.log(`   去掉 AI_SUMMARY_DRY_RUN 环境变量即可正式运行。`);
	}
}

main().catch((err) => {
	console.error("致命错误:", err);
	process.exit(1);
});