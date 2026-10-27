import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dir = path.join(root, "docs/intelligence-history");
const atlas = JSON.parse(fs.readFileSync(path.join(dir, "atlas.json"), "utf8"));
const checkOnly = process.argv.includes("--check");
const reading = {
	"discovery-only": "仅定位，待阅读",
	metadata: "已核对书目",
	abstract: "已读摘要/内容提要",
	"selected-sections": "已读指定片段",
	"full-text": "已读正文",
};
const dateBasis = {
	publication: "原始出版年",
	"preprint-v1": "首版预印本年",
	"conference-preprint-unresolved": "会议与预印本年份待核",
	"volume-publication-unresolved": "卷年与刊年待核",
	"textual-tradition": "文本传统，成书年代待核",
	"traditional-dating": "传统约定年代，待原典研究核实",
};
const escape = (s) =>
	String(s).replace(
		/[&<>"']/g,
		(c) =>
			({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
				c
			],
	);
const table = (s) => String(s).replaceAll("|", "\\|").replaceAll("\n", " ");
const markdownPath = (p) =>
	encodeURI(p).replaceAll("(", "%28").replaceAll(")", "%29");
const local = (p) => `[${path.basename(p)}](../../${markdownPath(p)})`;
const byId = (items) => new Map(items.map((item) => [item.id, item]));
const sourceMap = byId(atlas.sources);
const eventMap = byId(atlas.events);
const routeMap = byId(atlas.routes);
const sortEvents = (a, b) =>
	(a.date.start ?? Infinity) - (b.date.start ?? Infinity) ||
	a.id.localeCompare(b.id);
const routeEvents = (r) =>
	atlas.events.filter((e) => e.routeIds.includes(r.id)).sort(sortEvents);
const sourceLink = (s) => `[${table(s.title)}](${s.url})`;
const sourceLine = (s) =>
	`${sourceLink(s)}；${s.authors}；书目年 ${s.year ?? "待核"}；${reading[s.readingStatus]}。`;
const modern = (e) => e.date.start !== null && e.date.start >= 1800;
const classic = atlas.events.filter((e) => !modern(e));
const stats = Object.fromEntries(
	Object.keys(reading).map((k) => [
		k,
		atlas.sources.filter((s) => s.readingStatus === k).length,
	]),
);

function validate() {
	assert.equal(atlas.schemaVersion, 1, "不支持的数据版本");
	for (const key of ["routes", "sources", "events", "relations"]) {
		assert.equal(
			new Set(atlas[key].map((x) => x.id)).size,
			atlas[key].length,
			`${key} ID 重复`,
		);
		for (const item of atlas[key])
			assert.match(item.id, /^[a-z][a-z0-9-]*$/, `${key} ID 不稳定或不安全`);
	}
	const routeCode = fs
		.readFileSync(path.join(root, "src/data/intelligenceRoutes.ts"), "utf8")
		.split("export const INTELLIGENCE_ROUTES")[1]
		.split("export const ROUTE_CLUSTERS")[0];
	const siteIds = [...routeCode.matchAll(/\n\t\tid: "([^"]+)"/g)].map(
		(m) => m[1],
	);
	assert.deepEqual(
		[...routeMap.keys()].sort(),
		siteIds.sort(),
		"研究路线与当前网站的路线 ID 不一致",
	);
	for (const r of atlas.routes) {
		assert.ok(
			r.question && r.searchTerms.length && r.gaps.length && r.branches.length,
			`${r.id} 缺研究信息`,
		);
		assert.ok(routeEvents(r).length >= 4, `${r.id} 候选覆盖不足`);
		for (const p of r.existingMaterials)
			assert.ok(fs.existsSync(path.join(root, p)), `材料不存在：${p}`);
	}
	for (const s of atlas.sources) {
		assert.ok(
			s.title && s.authors && s.readLocator && s.checkedAt && s.sourceType,
			`${s.id} 缺来源身份或阅读范围`,
		);
		assert.equal(
			new URL(s.url).protocol,
			"https:",
			`${s.id} 来源链接协议不安全`,
		);
		assert.ok(s.readingStatus in reading, `${s.id} 阅读状态未知`);
		assert.ok(
			atlas.events.some((e) => e.sourceIds.includes(s.id)),
			`${s.id} 来源未关联事件`,
		);
	}
	const checks = JSON.parse(
		fs.readFileSync(path.join(dir, "source-checks.json"), "utf8"),
	);
	for (const entry of checks.openAttempts) {
		assert.ok(
			sourceMap.has(entry.sourceId),
			`访问记录引用未知来源 ${entry.sourceId}`,
		);
		assert.ok(
			entry.attemptedAt && entry.note && entry.url,
			"访问记录缺日期、范围或入口",
		);
	}
	for (const s of atlas.sources.filter(
		(s) => s.discoveryStatus === "record-checked",
	)) {
		assert.ok(
			checks.openAttempts.some(
				(entry) =>
					entry.sourceId === s.id &&
					entry.outcome === "record-reviewed" &&
					entry.readingStatus === s.readingStatus,
			),
			`${s.id} 阅读状态没有访问记录支撑`,
		);
	}
	for (const e of atlas.events) {
		assert.ok(
			e.title && e.question && e.approach && e.openQuestion,
			`${e.id} 缺问题或待核边界`,
		);
		assert.ok(e.routeIds.length && e.sourceIds.length, `${e.id} 无归属或来源`);
		assert.equal(
			new Set(e.routeIds).size,
			e.routeIds.length,
			`${e.id} 重复归属`,
		);
		for (const id of e.routeIds)
			assert.ok(routeMap.has(id), `${e.id} 引用未知路线 ${id}`);
		for (const id of e.sourceIds)
			assert.ok(sourceMap.has(id), `${e.id} 引用未知来源 ${id}`);
		assert.equal(
			e.claimStatus,
			"candidate",
			"第一轮不能把候选自动升级为已验证结论",
		);
		const d = e.date;
		assert.ok(d.label && d.basis && d.precision, `${e.id} 日期无口径`);
		assert.ok(d.basis in dateBasis, `${e.id} 日期口径未知`);
		if (d.start === null)
			assert.ok(
				d.end === null && d.precision === "unknown",
				`${e.id} 未知日期不一致`,
			);
		else
			assert.ok(
				Number.isInteger(d.start) &&
					Number.isInteger(d.end) &&
					d.end >= d.start,
				`${e.id} 日期范围错误`,
			);
	}
	for (const r of atlas.relations) {
		assert.ok(
			eventMap.has(r.from) && eventMap.has(r.to) && r.from !== r.to,
			`${r.id} 端点错误`,
		);
		assert.equal(
			r.type,
			"editorial-comparison",
			"第一轮关系必须显式标注为编辑性比较",
		);
		assert.ok(
			r.basis && r.question && r.sourceIds.length >= 2,
			`${r.id} 比较无依据说明`,
		);
		for (const id of r.sourceIds)
			assert.ok(sourceMap.has(id), `${r.id} 引用未知来源`);
	}
}

function timelineSvg(routes = atlas.routes) {
	const selected = atlas.events.filter((e) =>
		e.routeIds.some((id) => routes.some((r) => r.id === id)),
	);
	const ancient = selected.filter((e) => !modern(e));
	const dates = atlas.events
		.filter(modern)
		.flatMap((e) => [e.date.start, e.date.end]);
	const min = Math.floor(Math.min(...dates) / 10) * 10;
	const max = Math.ceil(Math.max(...dates) / 10) * 10;
	const x = (year) => 360 + ((year - min) / (max - min)) * 600;
	const rowHeight = 96;
	const bottom = 130 + routes.length * rowHeight;
	const legendY = bottom + (ancient.length ? 110 + ancient.length * 28 : 65);
	const height =
		routes.length === 1
			? legendY + 30 * selected.length + 28
			: bottom + 100 + ancient.length * 28;
	const colors = ["#986044", "#386c59", "#426e8a", "#725f7d", "#7a6635"];
	const color = (r) =>
		colors[atlas.clusters.findIndex((c) => c.id === r.cluster)];
	const parts = [
		`<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="${height}" viewBox="0 0 1000 ${height}" role="img" aria-labelledby="history-title history-desc">`,
		`<title id="history-title">${escape(routes.length === 1 ? routes[0].name : "智能问题演进史：20 条路线候选时间线")}</title>`,
		`<desc id="history-desc">${selected.length} 个候选事件。横轴为出版年份，纵轴为研究路线。编号对应路线底稿的节点序号。时间顺序不表示历史影响。古典文本另列，所有结论待深入审校。</desc>`,
		`<rect width="1000" height="${height}" fill="#fffdf8"/>`,
		`<g font-family="system-ui, PingFang SC, sans-serif" fill="#263c35">`,
		`<text x="24" y="36" font-size="23" font-weight="600">${escape(routes.length === 1 ? routes[0].name : "智能问题演进史 · 第一轮候选全景")}</text>`,
		`<text x="24" y="63" font-size="13">${atlas.updatedAt} · ${selected.length} 个独立事件 · 图中编号可点击；仅表示年代位置，不表示继承</text>`,
	];
	const ticks = [
		...new Set([
			min,
			...Array.from(
				{ length: Math.ceil((max - min) / 50) },
				(_, i) => Math.ceil(min / 50) * 50 + i * 50,
			).filter((v) => v > min && v < max),
			max,
		]),
	];
	for (const year of ticks)
		parts.push(
			`<line x1="${x(year)}" x2="${x(year)}" y1="105" y2="${bottom}" stroke="#dfe5de"/><text x="${x(year)}" y="99" text-anchor="middle" font-size="12">${year}</text>`,
		);
	parts.push(
		`<text x="24" y="100" font-size="13">路线／研究对象</text><text x="960" y="83" text-anchor="end" font-size="12">现代出版年份（等比例）</text>`,
	);
	for (const [i, r] of routes.entries()) {
		const y = 150 + i * rowHeight;
		const name = `${String(r.index).padStart(2, "0")} ${r.name.replace(/路线$/, "")}`;
		const chars = Array.from(name);
		const chunks = [
			chars.slice(0, 22).join(""),
			chars.slice(22).join(""),
		].filter(Boolean);
		parts.push(
			`<g data-route="${r.id}"><line x1="360" x2="960" y1="${y}" y2="${y}" stroke="#d4ddd5"/>`,
		);
		chunks.forEach((text, n) =>
			parts.push(
				`<text x="24" y="${y - (chunks.length - 1) * 8 + n * 19}" font-size="14">${escape(text)}</text>`,
			),
		);
		const placed = [];
		for (const [j, e] of routeEvents(r).entries()) {
			if (!modern(e)) continue;
			const cx = x((e.date.start + e.date.end) / 2);
			const dy =
				[0, -21, 21, -32, 32].find((offset) =>
					placed.every((p) => Math.hypot(cx - p.x, y + offset - p.y) >= 23),
				) ?? 0;
			const cy = y + dy;
			placed.push({ x: cx, y: cy });
			const label = `${r.index}.${j + 1} ${e.date.label} ${e.title}`;
			if (dy)
				parts.push(
					`<line x1="${cx}" x2="${cx}" y1="${y}" y2="${cy}" stroke="#c1cbc3"/>`,
				);
			if (e.date.start !== e.date.end)
				parts.push(
					`<line x1="${x(e.date.start)}" x2="${x(e.date.end)}" y1="${cy}" y2="${cy}" stroke="${color(r)}" stroke-width="4"/>`,
				);
			parts.push(
				`<a href="round-one.md#${e.id}" data-event="${e.id}" aria-label="${escape(label)}"><title>${escape(label)}</title><circle cx="${cx}" cy="${cy}" r="10" fill="${color(r)}"/><text x="${cx}" y="${cy + 4}" font-size="11" text-anchor="middle" fill="#ffffff">${j + 1}</text></a>`,
			);
		}
		parts.push("</g>");
	}
	parts.push(
		`<text x="24" y="${bottom + 25}" font-size="13">全部节点为候选；编号与各路线底稿一致。区间日期取中点定位，完整口径见节点详情。</text>`,
	);
	if (ancient.length) {
		parts.push(
			`<text x="24" y="${bottom + 54}" font-size="14" font-weight="600">古典文本入口（单列，不与现代轴等比例；译本年不作为原作年代）</text>`,
		);
		ancient.forEach((e, i) =>
			parts.push(
				`<a href="round-one.md#${e.id}" data-event="${e.id}"><text x="24" y="${bottom + 82 + i * 28}" font-size="14">${escape(`${e.date.label} · ${e.title}`)}</text></a>`,
			),
		);
	}
	if (routes.length === 1) {
		parts.push(
			`<text x="24" y="${legendY}" font-size="14" font-weight="600">编号／日期／候选问题变化（点击查看详情）</text>`,
		);
		routeEvents(routes[0]).forEach((e, i) =>
			parts.push(
				`<a href="round-one.md#${e.id}" data-event="${e.id}"><text x="24" y="${legendY + (i + 1) * 30}" font-size="14">${escape(`${i + 1} · ${e.date.label} · ${e.title}`)}</text></a>`,
			),
		);
	}
	parts.push("</g></svg>");
	return parts.join("\n") + "\n";
}

function roundOne() {
	const lines = [
		"# 第一轮：20 条路线的候选时间线",
		"",
		`检索与整理日期：${atlas.updatedAt}。${atlas.events.length} 个独立候选事件、${atlas.sources.length} 份来源，允许跨路线归属。全部方法摘要是初筛工作稿，待原文深读和事实审校；下列待核问题是本站的阅读任务，不冒充作者自述局限。`,
		"",
		"[返回总览](README.md) · [文献队列](reading-queue.md) · [可交互阅读图](atlas.html) · [编辑数据](atlas.json)",
		"",
		"![全景候选时间线](timeline.svg)",
		"",
		"编号与图中每条路线的节点一致。时间顺序不表示继承；未知年代置于路线末尾。正式文献页记录书目身份，不自动证明节点解释。",
		"",
	];
	const anchored = new Set();
	for (const r of atlas.routes) {
		lines.push(
			`<a id="${r.id}"></a>`,
			"",
			`## ${String(r.index).padStart(2, "0")} ${r.name}${r.pilot ? " · 第二轮样板" : ""}`,
			"",
			`**核心问题：** ${r.question}`,
			"",
			`**当前分支：** ${r.branches.join("；")}。`,
			"",
			`**建议继续检索：** ${r.searchTerms.join("；")}。`,
			"",
			`**已有材料：** ${r.existingMaterials.length ? r.existingMaterials.map(local).join("、") : "未定位到本项目的专门解读；需要从原始来源建立阅读笔记。"}`,
			"",
			"**当前缺口：**",
			"",
			...r.gaps.map((g) => `- ${g}`),
			"",
		);
		for (const [i, e] of routeEvents(r).entries()) {
			if (!anchored.has(e.id)) {
				lines.push(`<a id="${e.id}"></a>`, "");
				anchored.add(e.id);
			}
			lines.push(
				`### ${r.index}.${i + 1} ${e.date.label} · ${e.title}`,
				"",
				`- **当时要研究的问题：** ${e.question}`,
				`- **候选变化：** ${e.approach}`,
				`- **下一轮重点核对：** ${e.openQuestion}`,
				`- **日期口径：** ${e.date.basis}；精度 ${e.date.precision}。`,
				`- **来源：** ${e.sourceIds.map((id) => sourceLine(sourceMap.get(id))).join(" ")}`,
				"- **记录：** `" + e.id + "`；候选，未完成全文论证审校。",
				"",
			);
		}
	}
	lines.push(
		"## 跨路线比较队列",
		"",
		"以下均为本站提出的比较问题，不代表已考证的历史影响：",
		"",
	);
	for (const r of atlas.relations)
		lines.push(
			`- **${r.question}** [${eventMap.get(r.from).title}](#${r.from}) ↔ [${eventMap.get(r.to).title}](#${r.to})。`,
		);
	return lines.join("\n") + "\n";
}

function queue() {
	const lines = [
		"# 文献工作台：第一轮阅读队列",
		"",
		`[返回总览](README.md) · [路线与事件](round-one.md) · [维护方案](../intelligence-history-plan.md)`,
		"",
		`共 ${atlas.sources.length} 条独立来源。已打开并核对书目/摘要/片段 ${atlas.sources.filter((s) => s.discoveryStatus === "record-checked").length} 条；正文深读完成 ${stats["full-text"]} 条。全部来源身份和摘要解释仍需在下一轮按原文复核。`,
		"",
		"优先读取三条样板关联文献，再按跨路线问题扩展。相同来源只存一条；关联路线不代表影响关系。目录、译文或再版均在版本说明中区分。",
		"",
	];
	const pilotIds = new Set(
		atlas.routes
			.filter((r) => r.pilot)
			.flatMap((r) => routeEvents(r).flatMap((e) => e.sourceIds)),
	);
	for (const s of [...atlas.sources].sort(
		(a, b) =>
			Number(pilotIds.has(b.id)) - Number(pilotIds.has(a.id)) ||
			(a.year ?? Infinity) - (b.year ?? Infinity) ||
			a.id.localeCompare(b.id),
	)) {
		const linked = atlas.events.filter((e) => e.sourceIds.includes(s.id));
		lines.push(
			`<a id="${s.id}"></a>`,
			"",
			`## ${s.title}`,
			"",
			`${s.authors} · 书目年 ${s.year ?? "待核"} · ${s.kind}${pilotIds.has(s.id) ? " · 样板优先" : ""}`,
			"",
			`- 原始入口：[访问来源](${s.url})。类型：${s.sourceType}。`,
			`- 阅读状态：**${reading[s.readingStatus]}**。${s.readLocator}`,
			`- 定位日期：${s.checkedAt}。身份状态：${s.discoveryStatus}。`,
			...(s.versionNote ? [`- 版本与日期：${s.versionNote}`] : []),
			...(s.accessNote ? [`- 访问记录：${s.accessNote}`] : []),
			`- 关联事件：${linked.map((e) => `[${e.title}](round-one.md#${e.id})`).join("；")}。`,
			`- 下一步：${linked.map((e) => e.openQuestion).join(" ")}`,
			"",
		);
	}
	return lines.join("\n") + "\n";
}

function readme() {
	const lines = [
		"# 智能问题演进史 · 第一轮",
		"",
		`2026-10-23 建档。**本轮完成的是候选全景与文献工作台；深读、历史关系考证和正式解读尚待第二轮。**`,
		"",
		"## 阅读入口",
		"",
		"- [研究与维护方案](../intelligence-history-plan.md)：长期目标、四轮交付、证据标准和编辑约定。",
		"- [打开可交互时间线](atlas.html)：本地浏览器选择路线、点击节点查看问题与来源；无网络依赖。",
		"- [20 条路线完整底稿](round-one.md)：核心问题、候选转折、已有材料、检索词与缺口。",
		"- [文献队列](reading-queue.md)：逐项来源身份、阅读范围、版本和下一步。",
		"- [可编辑数据](atlas.json) · [Mermaid 源图](timeline.mmd) · [SVG 总图](timeline.svg) · [来源访问记录](source-checks.json)。",
		"",
		"![候选全景](timeline.svg)",
		"",
		"## 本轮范围与状态",
		"",
		`- ${atlas.routes.length} 条路线、${atlas.events.length} 个独立事件、${atlas.sources.length} 份来源、${atlas.relations.length} 个编辑性跨路线比较问题。`,
		`- ${atlas.events.filter(modern).length} 个现代节点，另列 ${classic.length} 个古典文本入口；跨路线重复展示不会重复计算事件。`,
		`- 仅定位 ${stats["discovery-only"]} 条；书目核对 ${stats.metadata} 条；摘要/内容提要 ${stats.abstract} 条；指定片段 ${stats["selected-sections"]} 条；正文深读 ${stats["full-text"]} 条。`,
		"- 采用探索性检索，覆盖截至本轮已发现的候选，不声称系统综述、历史穷尽或截至 2026 年的最新文献完备。",
		"- 全部事件的主张状态都是 candidate；来源网页的存在、摘要阅读和结论成立分别记录。",
		"",
		"## 路线盘点",
		"",
		"| 路线 | 候选节点（可交叉归属） | 本项目相关材料入口 | 下一轮 |",
		"| --- | ---: | ---: | --- |",
	];
	for (const r of atlas.routes)
		lines.push(
			`| [${String(r.index).padStart(2, "0")} ${r.name}](round-one.md#${r.id}) | ${routeEvents(r).length} | ${r.existingMaterials.length} | ${r.pilot ? "样板深读" : "按问题补齐"} |`,
		);
	lines.push(
		"",
		"盘点列为人工选择的相关材料入口，不是已读论文数或文章总量。大模型七卷、果蝇连接组、Strandbeest、强化学习和因果推断已有积累；社会文化、意识、生命自主性及非经典路线缺少专门解读。现有路线匹配器通常只返回首个命中，本轮未据它自动判定全部归属。 本轮人工关联 26 个不同的项目材料入口。当前文献架有 8 项：minimind、nanochat、mcts-survey、alphazero、causal-remix、ylearn、transformer、react；其中 alphazero 与 transformer 分别对应本底稿的 silver-2017 和 vaswani-2017。其他条目保留为项目理解入口，尚未完成逐一的书目映射。",
		"",
		"## 图的编辑与重生成",
		"",
		"编辑 atlas.json 中的来源、事件、路线或比较问题，再从项目根目录执行：",
		"",
		"```sh",
		"node scripts/generate-intelligence-history.mjs",
		"node scripts/generate-intelligence-history.mjs --check",
		"```",
		"",
		"生成器检查 ID、路线映射、来源引用、日期范围和本地材料路径，并重建本目录的底稿、文献队列、SVG、Mermaid 和离线阅读图。--check 不写文件，同时检查生成文件是否与数据一致。修改数据后重生成；手工编辑生成图会在下次生成时被覆盖。",
		"",
		"SVG 横轴为等比例现代出版年，区间日期取中点绘制并保留完整标签；古典文本独立列出。圆点编号对应路线底稿中的节点顺序，同年节点会错开纵向位置。横线只作坐标参考，未绘制历史继承关系。Mermaid 版本是按路线组织的顺序图，间距不表示时间长短。",
		"",
		"## 进入第二轮的具体任务",
		"",
		"1. 学习效率与泛化：补充迁移/组合泛化的反例，将先验、经验、计算预算与评测协议分开，扩展为 8–12 个可审校节点。",
		"2. 控制论与强化学习：补 Bellman、策略梯度及主动推断原始算法；分别重建分支，再考证模型、奖励与自维持的联系。",
		"3. 具身与生态：补 Gibson 原始研究和被动行走；区分机器人演示、动力系统模型、哲学论证和形态计算度量。",
		"4. 每个样板节点补原文定位、论证/推导、反方材料和证据边界；已有工程文章用作理解入口，不替代历史原文。",
		"",
		"## 已发现的资料风险",
		"",
		"- Binet–Simon 的卷年/刊年、信息瓶颈的会议/预印本年尚待核；图中以范围保留。",
		"- Piaget、Vygotsky、Wiener、Marr 等的原作、译本或再版日期需要分开。",
		"- 中观译本出版页的检索摘要疑似串书，未将摘要作为理论依据。",
		"- 若干出版入口返回拒绝访问、验证码、重定向错误或空白壳，保留为仅定位；访问问题见来源记录。",
		"- 尚未识别的历史 dot 文献批次仍独立待查，本轮集合不冒充该批文献。",
		"",
		"## 验收记录",
		"",
		"见 [validation.md](validation.md)。",
	);
	return lines.join("\n") + "\n";
}

function mermaid() {
	const clean = (s) =>
		s.replaceAll(":", "：").replaceAll(";", "；").replaceAll("\n", " ");
	return (
		[
			"timeline",
			"    title 智能问题演进史 · 候选顺序图（间距不代表时长）",
			...atlas.routes.flatMap((r) => [
				`    section ${r.index} ${r.name}`,
				...routeEvents(r).map(
					(e, i) =>
						`        ${clean(e.date.label)} : ${r.index}.${i + 1} ${clean(e.title)}`,
				),
			]),
		].join("\n") + "\n"
	);
}

function html() {
	const views = Object.fromEntries([
		["all", timelineSvg()],
		...atlas.routes.map((r) => [r.id, timelineSvg([r])]),
	]);
	const payload = JSON.stringify({
		atlas,
		views,
		reading,
		dateBasis,
	}).replaceAll("<", "\\u003c");
	return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>智能问题演进史 · 候选时间线</title>
<style>
:root{font-family:system-ui,"PingFang SC",sans-serif;color:#263c35;background:#fffdf8;line-height:1.7}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:24px}h1{font-size:26px;line-height:1.4}h2{font-size:21px}p{max-width:82ch}a{color:#286750}select{font:inherit;max-width:100%;padding:9px;background:#fff;border:1px solid #6b8175;border-radius:4px}label{display:block;margin-bottom:6px}.chart{overflow-x:auto;margin-top:24px}.chart svg{display:block;max-width:none}.chart a:hover circle,.chart a:focus circle{stroke:#13251d;stroke-width:3}.detail{border-top:1px solid #cbd5cb;margin-top:24px;padding-top:8px;overflow-wrap:anywhere}.muted{color:#54695f}.links{display:flex;flex-wrap:wrap;gap:18px}li{margin-bottom:10px}.event-list{padding-left:24px}code{font-size:.92em}button{font:inherit}a:focus-visible,select:focus-visible{outline:3px solid #a1633e;outline-offset:4px}@media(max-width:600px){main{padding:16px}h1{font-size:23px}.chart svg{width:1000px}select{width:100%}}
</style></head><body><main>
<h1>智能问题演进史 · 第一轮候选时间线</h1><p>20 条路线，${atlas.events.length} 个独立事件。先选择路线，再点击图中的编号查看问题、候选方法与原始来源。所有节点仍待深读审校；时间先后不代表历史影响。</p>
<p class="links"><a href="README.md">研究总览</a><a href="round-one.md">完整底稿</a><a href="reading-queue.md">文献队列</a><a href="atlas.json">可编辑数据</a></p>
<label for="route">选择研究路线</label><select id="route"><option value="all">全部 20 条路线</option>${atlas.routes.map((r) => `<option value="${r.id}">${r.index}. ${escape(r.name)}</option>`).join("")}</select>
<p id="route-question" class="muted">横轴为等比例现代出版年；古典文本另列。图可以横向滚动，文字详情保持正常阅读宽度。</p>
<div class="chart" id="chart">${views.all}</div>
<section class="detail" id="selected-event" aria-live="polite"><h2>节点详情</h2><p>选择图中的编号，或选择一条路线后使用下方的文字目录。</p></section>
<section id="route-list"></section><p class="muted">维护方法：编辑 atlas.json 后重新生成。跨路线比较问题见完整底稿，尚待原文考证。</p>
<noscript><p>当前显示完整 SVG 总图。交互筛选需要 JavaScript；完整底稿和文献队列可独立阅读。</p></noscript>
<script id="history-data" type="application/json">${payload}</script>
<script>
const {atlas,views,reading,dateBasis}=JSON.parse(document.getElementById("history-data").textContent);
const safe=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const chart=document.getElementById("chart"),detail=document.getElementById("selected-event"),picker=document.getElementById("route");
function showEvent(id){const e=atlas.events.find(x=>x.id===id);if(!e)return;detail.innerHTML='<h2>'+safe(e.date.label+' · '+e.title)+'</h2><p><strong>问题：</strong>'+safe(e.question)+'</p><p><strong>候选变化：</strong>'+safe(e.approach)+'</p><p><strong>待核问题：</strong>'+safe(e.openQuestion)+'</p><p class="muted">日期口径：'+safe(dateBasis[e.date.basis] ?? e.date.basis)+'；主张状态：候选。</p><ul>'+e.sourceIds.map(id=>{const s=atlas.sources.find(x=>x.id===id);return '<li><a href="'+safe(s.url)+'" target="_blank" rel="noopener noreferrer">'+safe(s.title)+'</a><br>'+safe(s.authors+' · '+(s.year??'书目年待核')+' · '+reading[s.readingStatus])+'<br>'+safe(s.readLocator)+(s.versionNote?'<br>'+safe(s.versionNote):'')+(s.accessNote?'<br>'+safe(s.accessNote):'')+'</li>';}).join('')+'</ul>';}
function chooseRoute(){const id=picker.value;chart.innerHTML=views[id];const r=atlas.routes.find(x=>x.id===id);document.getElementById('route-question').textContent=r?r.question:'横轴为等比例现代出版年；古典文本另列。图可以横向滚动，文字详情保持正常阅读宽度。';const events=r?atlas.events.filter(e=>e.routeIds.includes(id)).sort((a,b)=>(a.date.start??Infinity)-(b.date.start??Infinity)||a.id.localeCompare(b.id)):[];document.getElementById('route-list').innerHTML=r?'<h2>路线文字目录</h2><ol class="event-list">'+events.map(e=>'<li><a href="#selected-event" data-event="'+e.id+'">'+safe(e.date.label+' · '+e.title)+'</a></li>').join('')+'</ol><h2>资料缺口</h2><ul>'+r.gaps.map(g=>'<li>'+safe(g)+'</li>').join('')+'</ul>':'';if(events.length)showEvent(events[0].id);}
picker.addEventListener('change',chooseRoute);document.addEventListener('click',e=>{const link=e.target.closest('[data-event]');if(!link)return;e.preventDefault();showEvent(link.dataset.event);detail.scrollIntoView({block:'start'});});
</script></main></body></html>\n`;
}

validate();
const outputs = {
	"README.md": readme(),
	"round-one.md": roundOne(),
	"reading-queue.md": queue(),
	"timeline.svg": timelineSvg(),
	"timeline.mmd": mermaid(),
	"atlas.html": html(),
};
for (const [name, content] of Object.entries(outputs)) {
	const target = path.join(dir, name);
	if (checkOnly)
		assert.equal(
			fs.readFileSync(target, "utf8"),
			content,
			`${name} 与数据不一致，请重新生成`,
		);
	else fs.writeFileSync(target, content);
}
console.log(
	JSON.stringify(
		{
			mode: checkOnly ? "check" : "generate",
			routes: atlas.routes.length,
			events: atlas.events.length,
			sources: atlas.sources.length,
			relations: atlas.relations.length,
			reading: stats,
			generated: Object.keys(outputs),
		},
		null,
		2,
	),
);
