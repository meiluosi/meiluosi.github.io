import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { createResearchViews } from "./lib/intelligence-history-round-three-views.mjs";

const root = path.resolve(import.meta.dirname, "../docs/intelligence-history");
const dir = path.join(root, "round-three-a");
const data = JSON.parse(fs.readFileSync(path.join(dir, "research.json"), "utf8"));
const baseline = JSON.parse(fs.readFileSync(path.join(root, "atlas.json"), "utf8"));
const previous = JSON.parse(fs.readFileSync(path.join(root, "round-two/research.json"), "utf8"));
const hash = (value) => crypto.createHash("sha256").update(value).digest("hex");
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
const ids = (items) => new Set(items.map((item) => item.id));
for (const key of ["sources", "nodes", "relations", "routes", "questionCards"]) {
	requireValue(ids(data[key]).size === data[key].length, `Duplicate IDs: ${key}`);
}
requireValue(hash(fs.readFileSync(path.join(root, "atlas.json"))) === data.baseline.sha256, "First-round baseline changed");
requireValue(hash(fs.readFileSync(path.join(root, "round-two/research.json"))) === data.baseline.roundTwoSHA256, "Second-round baseline changed");
const sourceMap = new Map(data.sources.map((s) => [s.id, s]));
const routeMap = new Map(data.routes.map((s) => [s.id, s]));
const nodeMap = new Map(data.nodes.map((s) => [s.id, s]));
const knownNodes = new Set([...ids(data.nodes), ...ids(previous.nodes)]);
const allClaims = data.nodes.flatMap((n) => n.claims);
requireValue(ids(allClaims).size === allClaims.length, "Duplicate claim IDs");
for (const n of data.nodes) {
	requireValue(n.routeIds.length && n.sourceIds.length && n.claims.length, `Missing links: ${n.id}`);
	for (const id of n.routeIds) requireValue(routeMap.has(id), `Unknown route: ${id}`);
	for (const id of n.sourceIds) requireValue(sourceMap.has(id), `Unknown source: ${id}`);
	for (const c of n.claims) requireValue(n.sourceIds.includes(c.sourceId) && c.locator && c.basis, `Incomplete claim: ${c.id}`);
	if (n.baselineEventId) requireValue(n.id === n.baselineEventId && baseline.events.some((e) => e.id === n.baselineEventId), `Unknown or changed baseline ID: ${n.id}`);
	if (n.previousRoundNodeId) requireValue(n.id === n.previousRoundNodeId && previous.nodes.some((e) => e.id === n.previousRoundNodeId), `Unknown previous node: ${n.id}`);
}
for (const q of data.questionCards) {
	requireValue(q.status === "editorial-comparison" && q.evidence.length && q.definitions.length && q.comparison && q.limits, `Incomplete comparison: ${q.id}`);
	requireValue(q.distinguishingTest.status === "proposed-not-run", `Unreviewed experiment status: ${q.id}`);
	for (const d of q.definitions) requireValue(d.label && d.meaning && d.nodeIds.every((id) => knownNodes.has(id)), `Invalid definition: ${q.id}`);
	for (const e of q.evidence) requireValue(knownNodes.has(e.nodeId) && e.locator && e.comment, `Invalid comparison evidence: ${q.id}`);
}
for (const s of data.sources) requireValue(s.readScope && s.unreadScope && s.version && s.url && s.readingStatus, `Incomplete source: ${s.id}`);
for (const r of data.relations) {
	requireValue(knownNodes.has(r.fromId) && knownNodes.has(r.toId), `Unknown relation endpoint: ${r.id}`);
	requireValue(r.evidenceSourceIds.every((id) => sourceMap.has(id)) && r.locator && r.explanation, `Incomplete relation: ${r.id}`);
}
for (const v of data.versionComparisons ?? []) {
	requireValue(nodeMap.has(v.nodeId) && sourceMap.has(v.earlierSourceId) && sourceMap.has(v.laterSourceId), `Unknown version comparison reference: ${v.id}`);
	requireValue(v.status === "selected-comparison-reviewed" && v.rows.length && v.limits, `Incomplete version comparison: ${v.id}`);
	for (const row of v.rows) requireValue(["aspect", "earlier", "later", "earlierLocator", "laterLocator", "interpretation"].every((key) => row[key]), `Missing version comparison evidence: ${v.id}`);
}
for (const n of data.nodes) if (n.versionComparisonId) requireValue(data.versionComparisons?.some((v) => v.id === n.versionComparisonId && v.nodeId === n.id), `Missing node version comparison: ${n.id}`);
requireValue(["in-progress", "completed"].includes(data.status), "Unknown batch status");
const completed = data.status === "completed";
const statusLabel = completed ? "已验收" : "进行中";
const generated = `<!-- 由 research.json 生成；修改数据后重生成。A批${statusLabel}；不代表完整路线史。 -->\n\n`;
const files = new Map();
const emit = (name, text) => files.set(name, `${generated}${text.trim()}\n`);
const table = (text) => String(text).replaceAll("|", "／").replaceAll("\n", " ");
const sourceLink = (s) => `[${s.title}](${s.url})`;
const fingerprint = (s) => s.sha256 ? `\`${s.sha256}\`` : "无：未取得可校验的本地PDF；实际阅读载体与范围见本条来源记录。";
const nodeLink = (id, prefix = "") => nodeMap.has(id)
	? `[${nodeMap.get(id).title}](${prefix}notes/${id}.md)`
	: `[${previous.nodes.find((n) => n.id === id)?.title ?? id}](${prefix}../round-two/notes/${id}.md)`;
const basis = { "source-reported": "原文主张", derived: "本次推导或验证", "editorial-inference": "本站分析" };
for (const n of data.nodes) {
	const sources = n.sourceIds.map((id) => sourceMap.get(id));
	emit(`notes/${n.id}.md`, `# ${n.dateLabel} · ${n.title}

${n.routeIds.map((id) => `[${routeMap.get(id).name}](../routes/${id}.md)`).join(" · ")} · [文献夹](../literature.md)

**原始问题：** ${n.question}

**方法与机制：** ${n.mechanism}

**成立条件：** ${n.assumptions}

## 主张与原文定位

${n.claims.map((c) => `- **${basis[c.basis] ?? c.basis}**：${c.text}\n  - 依据：${sourceLink(sourceMap.get(c.sourceId))}；${c.locator}。独立验证：${c.independentlyReplicated ? "已完成所标范围，见复现报告" : "未做独立复现"}。`).join("\n\n")}

## 阅读记录

${sources.map((s) => `### ${s.title}\n\n${s.authors.join("；")}，${s.year}。${s.version}\n\n- 实读：${s.readScope}\n- 未覆盖：${s.unreadScope}\n- 状态：${s.readingStatus}；核对日期：${s.checkedAt}\n- 来源：${sourceLink(s)}${s.doi ? `；DOI：${s.doi}` : ""}${s.metadataUrl ? `；[书目核对](${s.metadataUrl})` : ""}\n- SHA-256：${fingerprint(s)}${s.carriedFrom ? `\n- 阅读记录延续自第二轮；本批复核范围另列。` : ""}`).join("\n\n")}

## 边界与下一问

${n.limits}

${n.nextQuestion}

${n.id === "e-baird-1995" ? "[正式复现报告、结果与图](../replication/README.md)" : ""}
${n.versionComparisonId ? `[版本对照与未覆盖范围](../versions/${n.versionComparisonId}.md)` : ""}`);
}
for (const v of data.versionComparisons ?? []) {
	emit(`versions/${v.id}.md`, `# ${v.title}

${nodeLink(v.nodeId, "../")} · [返回A批](../README.md)

较早版本：${sourceLink(sourceMap.get(v.earlierSourceId))}。

较晚版本：${sourceLink(sourceMap.get(v.laterSourceId))}。

下表是指定范围的实际正文对照；版本名称或文件名不作为内容相同／不同的证据。

| 核对对象 | 较早版本 | 较晚版本 | 两侧定位 | 对研究记录的影响 |
| --- | --- | --- | --- | --- |
${v.rows.map((r) => `| ${table(r.aspect)} | ${table(r.earlier)} | ${table(r.later)} | 早：${table(r.earlierLocator)}；晚：${table(r.laterLocator)} | ${table(r.interpretation)} |`).join("\n")}

## 核对范围

${v.limits}

两版的载体、阅读范围和指纹分别保存在[文献夹](../literature.md)。第一、二轮记录保留为历史快照；本批对照不回写当时阅读状态。`);
}
emit("literature.md", `# A批文献夹（${statusLabel}）

当前${data.sources.length}份来源形成明确阅读记录。下载数不计为阅读数；未覆盖范围随来源保存。

${data.sources.map((s) => `## ${s.title}\n\n${sourceLink(s)} · ${s.authors.join("；")} · ${s.year}\n\n**版本：** ${s.version}\n\n**已读：** ${s.readScope}\n\n**未读／未验：** ${s.unreadScope}\n\n**关联节点：** ${data.nodes.filter((n) => n.sourceIds.includes(s.id)).map((n) => nodeLink(n.id)).join("；")}\n\n状态：${s.readingStatus}；记录日期${s.checkedAt}；指纹：${fingerprint(s)}${s.carriedFrom ? "\n\n第二轮阅读记录延续；本批新增复核范围已单列。" : ""}`).join("\n\n")}

[访问成功与失败记录](evidence/access-log.json) · [A批计划](plan.md)`);
emit("claims.md", `# A批主张台账（${statusLabel}）

| 节点 | 主张 | 依据类型 | 定位 | 独立复现 |
| --- | --- | --- | --- | --- |
${data.nodes.flatMap((n) => n.claims.map((c) => `| ${nodeLink(n.id)} | ${table(c.text)} | ${basis[c.basis]} | ${sourceLink(sourceMap.get(c.sourceId))}；${table(c.locator)} | ${c.independentlyReplicated ? "是，限定所列范围" : "否"} |`)).join("\n")}`);
emit("relations.md", `# A批关系台账（${statusLabel}）

同年双向争论保留双向边；图形箭头表示所标关系，不自动表示时间顺序或思想继承。

| 起点 → 终点 | 类型与状态 | 原文位置 | 解释 |
| --- | --- | --- | --- |
${data.relations.map((r) => `| ${nodeLink(r.fromId)} → ${nodeLink(r.toId)} | ${r.type} / ${r.status} | ${r.evidenceSourceIds.map((id) => sourceLink(sourceMap.get(id))).join("；")}；${table(r.locator)} | ${table(r.explanation)} |`).join("\n")}`);
const mermaid = (t) => t.replaceAll('"', "＂").replaceAll("\n", " ");
for (const q of data.questionCards) {
	const experiment = q.distinguishingTest;
	const graph = `flowchart LR\n${q.definitions.map((d, i) => `    d${i}["${mermaid(d.label)}"] -. "本站比较" .-> q["${mermaid(q.title)}"]`).join("\n")}\n    q --> t["提出区分检验，尚未执行"]\n`;
	files.set(`questions/${q.id}.mmd`, graph);
	emit(`questions/${q.id}.md`, `# ${q.title}

**问题：** ${q.question}

这是基于所列阅读范围的本站比较；下述实验为待执行构想，不计入正式复现。

## 定义与比较对象

${q.definitions.map((d) => `### ${d.label}\n\n${d.meaning}\n\n${d.nodeIds.map((id) => nodeLink(id, "../")).join(" · ")}`).join("\n\n")}

## 比较结论

${q.comparison}

\`\`\`mermaid
${graph}\`\`\`

[编辑比较图](${q.id}.mmd)

## 原文定位

${q.evidence.map((e) => `- ${nodeLink(e.nodeId, "../")}：${e.locator}；${e.comment}。`).join("\n")}

## 可区分的检验（未执行）

- 操纵：${experiment.intervention}
- 测量：${experiment.measure}
- 控制：${experiment.controls}
- 解释边界：${experiment.interpretation}

${q.limits}

下一问：${q.nextQuestion}

[返回A批](../README.md)`);
}
const views = createResearchViews(data, previous);
for (const [name, content] of views.files) files.set(name, content);
for (const r of data.routes) {
	const nodes = views.routeNodes[r.id];
	const currentCount = nodes.filter((n) => !n.previousRound).length;
	const selected = ids(nodes);
	const relations = data.relations.filter((e) => selected.has(e.fromId) && selected.has(e.toId));
	const mmd = `timeline\n    title ${r.name}：A批增量，非完整历史\n${nodes.map((n) => `    ${n.dateLabel} : ${mermaid(n.title)}`).join("\n")}\n`;
	const branch = `flowchart TB\n${nodes.map((n, i) => `    n${i}["${n.dateLabel} ${mermaid(n.title)}${n.previousRound ? "（第二轮延续）" : ""}"]`).join("\n")}\n${relations.map((e) => `    n${nodes.findIndex((n) => n.id === e.fromId)} ${e.status === "editorial" ? "-." : "--"} ${mermaid(views.relationLabels[e.type] ?? e.type)} ${e.status === "editorial" ? ".->" : "-->"} n${nodes.findIndex((n) => n.id === e.toId)}`).join("\n")}\n`;
	if (nodes.length) { files.set(`routes/${r.id}-timeline.mmd`, mmd); files.set(`routes/${r.id}-relations.mmd`, branch); }
	emit(`routes/${r.id}.md`, `# ${r.name} · A批增量

**${completed ? "已完成本批指定范围验收，后续扩展与未覆盖材料另列。" : "进行中，尚未完成本路线的批次验收。"}** 当前${currentCount}个本批节点，图中另有${nodes.length - currentCount}个第二轮延续节点；前两轮记录仍独立保留。

${r.narrative.join("\n\n")}

![等比例时间线](${r.id}-timeline.svg)

![分支与关系图](${r.id}-relations.svg)

[交互阅读](../atlas.html?route=${r.id}) · [编辑SVG时间线](${r.id}-timeline.svg) · [编辑SVG关系图](${r.id}-relations.svg)

图中标为“第二轮延续”的节点用于呈现本批新增关系，不计入本批独立节点数。SVG关系编号与下表对应；完整原文定位见[关系台账](../relations.md)。

${views.relationTables.get(r.id)}

${nodes.length ? `\`\`\`mermaid\n${mmd}\`\`\`\n\n[编辑时间线](${r.id}-timeline.mmd) · [编辑关系图](${r.id}-relations.mmd)\n\n\`\`\`mermaid\n${branch}\`\`\`` : "关键节点仍待阅读与审校。"}

${nodes.map((n) => `- ${nodeLink(n.id, "../")}：${n.question}`).join("\n")}

${r.gaps.length ? `## 未覆盖范围与后续研究\n\n${r.gaps.map((g) => `- ${g}`).join("\n")}` : "范围与验收见[A批计划](../plan.md)。"}

[返回A批](../README.md)`);
}
emit("README.md", `# 第三轮 A 批：表征与认知机制（${statusLabel}）

目标：补三个样板的关键断层，展开四条新路线，完成三张跨路线问题卡和一项正式复现。**${completed ? "已完成执行计划列明的A批范围；第三轮B—E批仍待开展。" : "本批尚未完成。"} 节点数不构成整条路线已覆盖的声明。**

[交互研究工作台](atlas.html) · [完整范围与验收计划](plan.md) · [${completed ? "验收记录" : "当前检查"}](validation.md) · [文献夹](literature.md) · [主张台账](claims.md) · [关系台账](relations.md) · [研究数据](research.json)

${(data.versionComparisons ?? []).map((v) => `[${v.title}](versions/${v.id}.md)`).join(" · ")}

当前增量：${data.nodes.length}个节点、${data.sources.length}份有阅读范围的来源、${allClaims.length}条定位主张、${data.relations.length}条关系。

| 路线 | 当前增量节点 |
| --- | ---: |
${data.routes.map((r) => `| [${r.name}](routes/${r.id}.md) | ${data.nodes.filter((n) => n.routeIds.includes(r.id)).length} |`).join("\n")}

一个节点可以属于多条路线，上表不能相加作为独立节点数。

## 跨路线问题卡

${data.questionCards.map((q) => `- [${q.title}](questions/${q.id}.md)`).join("\n")}

问题卡区分原文结论与本站比较；所提区分实验尚未运行。时间线、关系图、问题卡和笔记从同一份数据重建。

## 正式复现

[Baird六状态反例](replication/README.md)：原文版本、预先协议、独立实现、数值结果、可编辑曲线、参数敏感性和原文差异。复现范围限于图1的六状态预测问题，图5速度实验与一般强化学习结论不在本项范围内。

## 维护

修改research.json后运行 \`node scripts/generate-intelligence-history-round-three-a.mjs\`；加 \`--check\` 检查基线指纹、引用和生成一致性。上述校验不能代替文献论证审校或完整范围验收。第一轮与第二轮数据不回写当时阅读状态。

[第一轮](../README.md) · [第二轮](../round-two/README.md)`);
const check = process.argv.includes("--check");
for (const [name, content] of files) {
	const target = path.join(dir, name);
	if (check) requireValue(fs.existsSync(target) && fs.readFileSync(target, "utf8") === content, `Stale output: ${name}`);
	else { fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, content); }
}
console.log(`${check ? "Verified" : "Generated"} ${files.size} files; ${data.nodes.length} nodes, ${data.sources.length} sources. Batch remains ${data.status}.`);
