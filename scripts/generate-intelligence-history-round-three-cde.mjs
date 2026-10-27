import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { createResearchViews } from "./lib/intelligence-history-round-three-cde-views.mjs";

const root = path.resolve(import.meta.dirname, "../docs/intelligence-history");
const batch = process.argv.find(a => /^[cde]$/.test(a));
if (!batch) throw new Error("Usage: node scripts/generate-intelligence-history-round-three-cde.mjs c|d|e [--check]");
const dir = path.join(root, `round-three-${batch}`);
const data = JSON.parse(fs.readFileSync(path.join(dir, "research.json"), "utf8"));
const baseline = JSON.parse(fs.readFileSync(path.join(root, "atlas.json"), "utf8"));
const previous = JSON.parse(fs.readFileSync(path.join(root, "round-two/research.json"), "utf8"));
const hash = (value) => crypto.createHash("sha256").update(value).digest("hex");
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
const ids = (items) => new Set(items.map((item) => item.id));
for (const key of ["sources", "nodes", "relations", "routes", "questionCards"]) {
	requireValue(ids(data[key]).size === data[key].length, `Duplicate IDs: ${key}`);
}
for (const [name, expected] of Object.entries(data.baselines)) requireValue(hash(fs.readFileSync(path.join(root, name))) === expected, `Baseline changed: ${name}`);
const evaluationLabels = { kind: "证据性质", unit: "分析单位", observation: "观察／论证依据", discriminator: "需要区分的解释" };
for (const n of data.nodes) requireValue(Object.keys(evaluationLabels).every(k => n.evidence?.[k]), `Missing evidence field: ${n.id}`);
for (const r of data.revisits) {
 const prior = JSON.parse(fs.readFileSync(path.join(root, r.batch, "research.json"), "utf8"));
 requireValue(prior.questionCards.some(q => q.id === r.questionCardId) && r.evidenceNodeIds.every(id => data.nodes.some(n => n.id === id)), `Invalid revisit: ${r.id}`);
}
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
const generated = `<!-- 由 research.json 生成；修改数据后重生成。${data.batchLabel}批${statusLabel}；不代表完整路线史。 -->\n\n`;
const files = new Map();
const emit = (name, text) => files.set(name, `${generated}${text.trim()}\n`);
const table = (text) => String(text).replaceAll("|", "／").replaceAll("\n", " ");
const sourceLink = (s) => `[${s.title}](${s.url})`;
const fingerprint = (s) => s.sha256 ? `\`${s.sha256}\`${s.fingerprintScope ? `（范围：${s.fingerprintScope}）` : ""}` : "无本地载体指纹；实际阅读范围见本条来源记录。";
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

## 证据标准

| 项目 | 条件与边界 |
| --- | --- |
${Object.entries(evaluationLabels).map(([key, label]) => `| ${label} | ${table(n.evidence[key])} |`).join("\n")}

## 阅读记录

${sources.map((s) => `### ${s.title}\n\n${s.authors.join("；")}，${s.year ?? "年代见版本说明"}。${s.version}\n\n- 实读：${s.readScope}\n- 未覆盖：${s.unreadScope}\n- 状态：${s.readingStatus}；核对日期：${s.checkedAt}\n- 来源：${sourceLink(s)}${s.doi ? `；DOI：${s.doi}` : ""}${s.metadataUrl ? `；[书目核对](${s.metadataUrl})` : ""}\n- SHA-256：${fingerprint(s)}${s.carriedFrom ? `\n- 阅读记录延续自${s.carriedFrom}；本批复核范围另列。` : ""}`).join("\n\n")}

## 边界与下一问

${n.limits}

${n.nextQuestion}

${n.carriedFrom ? `阅读延续：[此前笔记](../../${n.carriedFrom}/notes/${n.id}.md)。` : ""}
${n.versionComparisonId ? `[版本对照与未覆盖范围](../versions/${n.versionComparisonId}.md)` : ""}`);
}
for (const v of data.versionComparisons ?? []) {
	emit(`versions/${v.id}.md`, `# ${v.title}

${nodeLink(v.nodeId, "../")} · [返回${data.batchLabel}批](../README.md)

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
emit("literature.md", `# ${data.batchLabel}批文献夹（${statusLabel}）

当前${data.sources.length}份来源形成明确阅读记录。下载数不计为阅读数；未覆盖范围随来源保存。

${data.sources.map((s) => `## ${s.title}\n\n${sourceLink(s)} · ${s.authors.join("；")} · ${s.year ?? "年代见版本说明"}\n\n**版本：** ${s.version}\n\n**已读：** ${s.readScope}\n\n**未读／未验：** ${s.unreadScope}\n\n**关联节点：** ${data.nodes.filter((n) => n.sourceIds.includes(s.id)).map((n) => nodeLink(n.id)).join("；")}\n\n状态：${s.readingStatus}；记录日期${s.checkedAt}；指纹：${fingerprint(s)}${s.carriedFrom ? "\n\n既有批次阅读记录延续；本批范围与来源批次已单列。" : ""}`).join("\n\n")}

[访问成功与失败记录](evidence/access-log.json) · [${data.batchLabel}批计划](plan.md)`);
emit("claims.md", `# ${data.batchLabel}批主张台账（${statusLabel}）

| 节点 | 主张 | 依据类型 | 定位 | 独立复现 |
| --- | --- | --- | --- | --- |
${data.nodes.flatMap((n) => n.claims.map((c) => `| ${nodeLink(n.id)} | ${table(c.text)} | ${basis[c.basis]} | ${sourceLink(sourceMap.get(c.sourceId))}；${table(c.locator)} | ${c.independentlyReplicated ? "是，限定所列范围" : "否"} |`)).join("\n")}`);
emit("relations.md", `# ${data.batchLabel}批关系台账（${statusLabel}）

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

[返回${data.batchLabel}批](../README.md)`);
}
const views = createResearchViews(data, previous);
for (const [name, content] of views.files) files.set(name, content);
for (const r of data.routes) {
	const nodes = views.routeNodes[r.id];
	const currentCount = nodes.filter((n) => !n.previousRound).length;
	const selected = ids(nodes);
	const relations = data.relations.filter((e) => selected.has(e.fromId) && selected.has(e.toId));
	const mmd = `timeline\n    title ${r.name}：${data.batchLabel}批增量，非完整历史\n${nodes.map((n) => `    ${n.dateLabel} : ${mermaid(n.title)}`).join("\n")}\n`;
	const branch = `flowchart TB\n${nodes.map((n, i) => `    n${i}["${n.dateLabel} ${mermaid(n.title)}${n.previousRound ? "（第二轮延续）" : ""}"]`).join("\n")}\n${relations.map((e) => `    n${nodes.findIndex((n) => n.id === e.fromId)} ${e.status === "editorial-inference" ? "-." : "--"} ${mermaid(views.relationLabels[e.type] ?? e.type)} ${e.status === "editorial-inference" ? ".->" : "-->"} n${nodes.findIndex((n) => n.id === e.toId)}`).join("\n")}\n`;
	if (nodes.length) { files.set(`routes/${r.id}-timeline.mmd`, mmd); files.set(`routes/${r.id}-relations.mmd`, branch); }
	emit(`routes/${r.id}.md`, `# ${r.name} · ${data.batchLabel}批增量

**${completed ? "已完成本批指定范围验收，后续扩展与未覆盖材料另列。" : "进行中，尚未完成本路线的批次验收。"}** 当前${currentCount}个本批节点，图中另有${nodes.length - currentCount}个第二轮延续节点；此前各批记录独立保留。

${r.narrative.join("\n\n")}

![现代等比例轴与古典文本独立区](${r.id}-timeline.svg)

![分支与关系图](${r.id}-relations.svg)

[交互阅读](../atlas.html?route=${r.id}) · [编辑SVG时间线](${r.id}-timeline.svg) · [编辑SVG关系图](${r.id}-relations.svg)

图中标为“第二轮延续”的节点用于呈现本批新增关系，不计入本批记录数。SVG关系编号与下表对应；完整原文定位见[关系台账](../relations.md)。

${views.relationTables.get(r.id)}

${nodes.length ? `\`\`\`mermaid\n${mmd}\`\`\`\n\n[编辑时间线](${r.id}-timeline.mmd) · [编辑关系图](${r.id}-relations.mmd)\n\n\`\`\`mermaid\n${branch}\`\`\`` : "关键节点仍待阅读与审校。"}

${nodes.map((n) => `- ${nodeLink(n.id, "../")}：${n.question}`).join("\n")}

${r.gaps.length ? `## 未覆盖范围与后续研究\n\n${r.gaps.map((g) => `- ${g}`).join("\n")}` : "范围与验收见[本批计划](../plan.md)。"}

[返回${data.batchLabel}批](../README.md)`);
}
emit("evaluation.md", `# 证据条件横向索引

实验、形式模型、概念论证、规范判断与历史文本分别标识；同名术语不保证解释对象一致。

| 节点 | 性质 | 单位 | 依据 | 区分条件 |
| --- | --- | --- | --- | --- |
${data.nodes.map(n => `| ${nodeLink(n.id)} | ${table(n.evidence.kind)} | ${table(n.evidence.unit)} | ${table(n.evidence.observation)} | ${table(n.evidence.discriminator)} |`).join("\n")}

[返回工作台](atlas.html)`);
emit("revisits.md", `# ${data.batchLabel}批对前批的回访

前批数据和问题卡保留原验收快照，以下是${data.batchLabel}批新增的解释条件。

${data.revisits.map(r => `## ${r.title}\n\n[原问题卡](../${r.batch}/questions/${r.questionCardId}.md)\n\n原问题：${r.original}\n\n新增：${r.addition}\n\n依据：${r.evidenceNodeIds.map(id => nodeLink(id)).join("；")}\n\n下一步：${r.nextStep}`).join("\n\n")}

[合成核验](method-check/README.md) · [返回${data.batchLabel}批](README.md)`);
emit("README.md", `# 第三轮 ${data.batchLabel} 批：${data.title}（${statusLabel}）

${data.scope}

**${completed ? "已完成plan.md所列范围；第四轮另行开展。" : "正在完成本批范围与验收。"} 这是有明确选读范围的研究增量，不是完整路线史。**

[全局总览](../index.html) · [交互研究工作台](atlas.html) · [执行计划](plan.md) · [验收记录](validation.md) · [文献夹](literature.md) · [主张台账](claims.md) · [关系证据](relations.md) · [证据条件索引](evaluation.md) · [前批回访](revisits.md) · [研究数据](research.json)

当前：${data.nodes.length}条本批节点记录（含延续）、${data.sources.length}份来源记录、${allClaims.length}条定位主张、${data.relations.length}条有类型的关系。下载、选读与独立验证分别记账。

| 路线 | 本批节点 |
| --- | ---: |
${data.routes.map(r => `| [${r.name}](routes/${r.id}.md) | ${data.nodes.filter(n => n.routeIds.includes(r.id)).length} |`).join("\n")}

节点可属于多条路线，表中数量不可直接相加。

## 跨路线问题卡

${data.questionCards.map(q => `- [${q.title}](questions/${q.id}.md)`).join("\n")}

问题卡的后续干预实验均未运行；已经完成的数学例子另列如下。

## 方法核验

[${data.methodCheckTitle}](method-check/README.md)：预先协议、精确计算、独立解析对照与负控制。结果只覆盖协议定义的合成模型；不计原论文或生物实验复现。

## 维护

先修改research.json，再运行 \`node scripts/generate-intelligence-history-round-three-cde.mjs ${batch}\`，加 \`--check\` 检查生成一致性与引用。核验程序为 \`python3 scripts/check-intelligence-cde.py ${batch} --check\`。时间线SVG、关系SVG、Mermaid与笔记可重建；直接修改生成文件会被覆盖。

[访问记录与缺口](evidence/access-log.json) · [第一轮](../README.md) · [第二轮](../round-two/README.md) · [第三轮A批](../round-three-a/README.md)`);
const check = process.argv.includes("--check");
for (const [name, content] of files) {
	const target = path.join(dir, name);
	if (check) requireValue(fs.existsSync(target) && fs.readFileSync(target, "utf8") === content, `Stale output: ${name}`);
	else { fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, content); }
}
console.log(`${check ? "Verified" : "Generated"} ${files.size} files; ${data.nodes.length} nodes, ${data.sources.length} sources. Batch remains ${data.status}.`);
