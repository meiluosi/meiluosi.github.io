/**
 * 知识图谱数据生成脚本
 *
 * 在 pnpm build 流程中运行，基于文章标签/分类生成图谱 JSON 数据。
 * 输出到 src/constants/knowledge-graph.json，供前端组件使用。
 *
 * 用法：
 *   npx tsx scripts/generate-knowledge-graph.ts
 */

import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";

// ========== 配置 ==========

const POSTS_DIR = "src/content/posts";
const OUTPUT_FILE = "src/constants/knowledge-graph.json";

// ========== 类型 ==========

interface GraphNode {
	id: string;
	label: string;
	type: "tag" | "category" | "post";
	url?: string;
	count?: number;
}

interface GraphEdge {
	source: string;
	target: string;
}

interface PostData {
	id: string;
	title: string;
	tags: string[];
	category: string;
}

interface GraphData {
	nodes: GraphNode[];
	edges: GraphEdge[];
}

// ========== 辅助函数 ==========

function parseFrontmatter(content: string): {
	title: string;
	tags: string[];
	category: string;
} {
	const match = content.match(/^---\n([\s\S]*?)\n---/);
	if (!match) return { title: "", tags: [], category: "" };

	const raw = match[1];
	const result: { title: string; tags: string[]; category: string } = {
		title: "",
		tags: [],
		category: "",
	};

	let currentKey = "";
	let inArray = false;

	for (const line of raw.split("\n")) {
		const trimmed = line.trim();

		if (inArray) {
			if (trimmed.startsWith("- ")) {
				const arr = (result as any)[currentKey] || [];
				arr.push(trimmed.slice(2).trim().replace(/^["']|["']$/g, ""));
				(result as any)[currentKey] = arr;
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
				} else if (key === "title" || key === "category") {
					(result as any)[key] = value.replace(/^["']|["']$/g, "");
				}
			}
		}
	}

	return result;
}

function slugify(text: string): string {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9\u4e00-\u9fff]+/g, "-")
		.replace(/^-|-$/g, "");
}

// ========== 主流程 ==========

async function main() {
	console.log("🗺️  知识图谱数据生成器\n");

	const postFiles = await glob("**/*.md", { cwd: POSTS_DIR });
	console.log(`   找到 ${postFiles.length} 篇文章\n`);

	const posts: PostData[] = [];
	const tagCounts: Record<string, number> = {};
	const categoryCounts: Record<string, number> = {};

	for (const file of postFiles) {
		const filePath = path.join(POSTS_DIR, file);
		const content = await fs.readFile(filePath, "utf-8");
		const fm = parseFrontmatter(content);

		if (!fm.title) continue;

		const id = file.replace(/\.md$/, "");

		posts.push({
			id,
			title: fm.title,
			tags: fm.tags,
			category: fm.category,
		});

		// 统计标签
		for (const tag of fm.tags) {
			tagCounts[tag] = (tagCounts[tag] || 0) + 1;
		}

		// 统计分类
		if (fm.category) {
			categoryCounts[fm.category] = (categoryCounts[fm.category] || 0) + 1;
		}
	}

	// 构建节点
	const nodes: GraphNode[] = [];
	const categoryNodeIds = new Set<string>();
	const tagNodeIds = new Set<string>();
	const postNodeIds = new Set<string>();

	// 分类节点
	for (const [name, count] of Object.entries(categoryCounts)) {
		const id = `cat:${slugify(name)}`;
		nodes.push({
			id,
			label: name,
			type: "category",
			url: `/categories/${slugify(name)}/`,
			count,
		});
		categoryNodeIds.add(id);
	}

	// 标签节点（只保留出现次数 >= 2 的标签，避免图谱过于拥挤）
	for (const [name, count] of Object.entries(tagCounts)) {
		if (count < 2) continue;
		const id = `tag:${slugify(name)}`;
		nodes.push({
			id,
			label: name,
			type: "tag",
			url: `/tags/${slugify(name)}/`,
			count,
		});
		tagNodeIds.add(id);
	}

	// 文章节点（只保留有标签的文章）
	for (const post of posts) {
		if (post.tags.length === 0) continue;
		const id = `post:${post.id}`;
		nodes.push({
			id,
			label: post.title,
			type: "post",
			url: `/posts/${post.id}/`,
		});
		postNodeIds.add(id);
	}

	// 构建边
	const edges: GraphEdge[] = [];

	for (const post of posts) {
		const postId = `post:${post.id}`;
		if (!postNodeIds.has(postId)) continue;

		// 文章 → 分类
		if (post.category) {
			const catId = `cat:${slugify(post.category)}`;
			if (categoryNodeIds.has(catId)) {
				edges.push({ source: postId, target: catId });
			}
		}

		// 文章 → 标签
		for (const tag of post.tags) {
			const tagId = `tag:${slugify(tag)}`;
			if (tagNodeIds.has(tagId)) {
				edges.push({ source: postId, target: tagId });
			}
		}
	}

	// 限制节点数量（最多 200 个节点，避免性能问题）
	if (nodes.length > 200) {
		console.log(`   ⚠ 节点数量 ${nodes.length} 超过 200，将进行裁剪...`);
		// 保留所有分类和出现次数最多的标签
		const catNodes = nodes.filter((n) => n.type === "category");
		const tagNodes = nodes
			.filter((n) => n.type === "tag")
			.sort((a, b) => (b.count || 0) - (a.count || 0))
			.slice(0, 30);
		const postNodes = nodes
			.filter((n) => n.type === "post")
			.slice(0, 200 - catNodes.length - tagNodes.length);

		const keptIds = new Set([
			...catNodes.map((n) => n.id),
			...tagNodes.map((n) => n.id),
			...postNodes.map((n) => n.id),
		]);

		nodes.length = 0;
		nodes.push(...catNodes, ...tagNodes, ...postNodes);

		// 过滤边
		const filteredEdges = edges.filter(
			(e) => keptIds.has(e.source) && keptIds.has(e.target),
		);
		edges.length = 0;
		edges.push(...filteredEdges);
	}

	const graphData: GraphData = { nodes, edges };

	// 写入文件
	await fs.mkdir(path.dirname(OUTPUT_FILE), { recursive: true });
	await fs.writeFile(OUTPUT_FILE, JSON.stringify(graphData, null, 2), "utf-8");

	console.log(`   ✅ 生成完成！`);
	console.log(`   📊 节点: ${nodes.length} (分类: ${categoryNodeIds.size}, 标签: ${tagNodeIds.size}, 文章: ${postNodeIds.size})`);
	console.log(`   🔗 边: ${edges.length}`);
	console.log(`   📁 输出: ${OUTPUT_FILE}\n`);
}

main().catch((err) => {
	console.error("致命错误:", err);
	process.exit(1);
});