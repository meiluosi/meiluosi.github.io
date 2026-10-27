import assert from "node:assert/strict";
import { buildConceptGraph } from "./article-concept-graph";

const graph = buildConceptGraph(
	"测试",
	"## A\n**采样** 与 `token`。\n### B\n**采样**。\n```md\n## 假标题\n**不可见**\n```\n## C\n正文。",
	[
		{ depth: 2, text: "A", slug: "a" },
		{ depth: 3, text: "B", slug: "b" },
		{ depth: 2, text: "C", slug: "c" },
	],
);
assert.equal(graph.nodes.find((n) => n.label === "B")?.parent, "section-0");
assert.equal(graph.nodes.find((n) => n.label === "C")?.parent, "root");
assert.equal(graph.links.length, 1);
assert.ok(
	!graph.nodes.some((n) => n.label === "不可见" || n.label === "假标题"),
);
assert.ok(
	graph.nodes
		.filter((n) => n.label === "采样")
		.every((n) => n.slug && n.evidence.includes("采样")),
);
assert.equal(new Set(graph.nodes.map((n) => n.id)).size, graph.nodes.length);
assert.equal(buildConceptGraph("Empty", "", []).nodes.length, 1);
console.log(
	"Concept hierarchy, explicit evidence, cross-section recurrence and fenced code exclusion passed.",
);

const anchored = buildConceptGraph(
	"Heading anchor",
	"## 方法\nMinHash 与 Jaccard。\n## 比较\nMinHash 与 LSH。",
	[
		{ depth: 2, text: "方法#", slug: "方法" },
		{ depth: 2, text: "比较#", slug: "比较" },
	],
);
assert.equal(anchored.nodes.find((n) => n.label === "方法")?.parent, "root");
assert.equal(anchored.nodes.filter((n) => n.label === "MinHash").length, 2);
assert.equal(anchored.links.length, 1);
assert.ok(
	anchored.nodes
		.filter((n) => n.kind === "concept")
		.every((n) => n.parent !== "root"),
);

const distinct = buildConceptGraph(
	"术语边界",
	"## A\n无监督学习与 tokenizer。",
	[{ depth: 2, text: "A#", slug: "a" }],
);
assert.ok(distinct.nodes.some((n) => n.label === "无监督学习"));
assert.ok(
	!distinct.nodes.some((n) => n.label === "监督学习" || n.label === "token"),
);
