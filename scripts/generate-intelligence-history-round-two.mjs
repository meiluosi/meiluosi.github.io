import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dir = path.join(root, "docs/intelligence-history/round-two");
const data = JSON.parse(
	fs.readFileSync(path.join(dir, "research.json"), "utf8"),
);
const demos = JSON.parse(
	fs.readFileSync(path.join(dir, "evidence/demos.json"), "utf8"),
);
const check = process.argv.includes("--check");
const esc = (s) =>
	String(s).replace(
		/[&<>"']/g,
		(c) =>
			({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
				c
			],
	);
const cell = (s) => String(s).replaceAll("|", "\\|").replaceAll("\n", " ");
const sorted = (a) =>
	[...a].sort((a, b) => a.year - b.year || a.id.localeCompare(b.id));
const nodes = new Map(data.nodes.map((n) => [n.id, n]));
const types = {
	"method-extension": "方法扩展",
	"benchmark-reuse": "复用评测",
	"method-relation": "明确的方法联系",
	critique: "回应／批评",
	citation: "明确引用",
	"editorial-comparison": "本站比较",
};
const edgeColor = (r) =>
	r.type === "critique"
		? "#b91c1c"
		: r.type === "editorial-comparison"
			? "#737373"
			: "#475569";
assert.equal(data.schemaVersion, 1);
assert.equal(nodes.size, data.nodes.length);
assert.equal(
	new Set(data.nodes.map((n) => n.source.id)).size,
	data.nodes.length,
);
const baseline = fs.readFileSync(path.join(dir, data.baseline.path));
assert.equal(
	crypto.createHash("sha256").update(baseline).digest("hex"),
	data.baseline.sha256,
	"第一轮基线发生变化，需明确迁移记录",
);
const baselineIds = new Set(JSON.parse(baseline).events.map((e) => e.id));
for (const n of data.nodes) {
	assert.match(n.id, /^e-[a-z0-9-]+$/);
	assert.ok(data.routes.some((r) => r.id === n.routeId));
	if (n.baselineEventId) assert.ok(baselineIds.has(n.baselineEventId));
	for (const key of [
		"title",
		"branch",
		"question",
		"mechanism",
		"assumptions",
		"limits",
		"criticism",
		"teachingExample",
		"nextQuestion",
		"dateBasis",
		"dateLabel",
	])
		assert.ok(n[key], `${n.id}缺${key}`);
	assert.equal(n.status, "claims-reviewed");
	assert.ok(Number.isInteger(n.year));
	assert.equal(new URL(n.source.url).protocol, "https:");
	assert.ok(n.source.version && n.source.readScope && n.source.checkedAt);
	assert.equal(n.source.readingStatus, "selected-sections");
	if (n.source.sha256) assert.match(n.source.sha256, /^[a-f0-9]{64}$/);
	assert.ok(n.claims.length);
	for (const c of n.claims) {
		assert.ok(c.id && c.locator && c.text);
		assert.equal(c.basis, "source-reported");
		assert.equal(c.independentlyReplicated, false);
	}
}
const sourceIds = new Set(data.nodes.map((n) => n.source.id));
assert.equal(
	new Set(data.relations.map((r) => r.id)).size,
	data.relations.length,
);
for (const r of data.relations) {
	assert.ok(nodes.has(r.fromId) && nodes.has(r.toId));
	assert.ok(r.type in types && r.locator && r.explanation);
	assert.ok(sourceIds.has(r.evidenceSourceId));
	assert.equal(
		r.status,
		r.type === "editorial-comparison" ? "editorial" : "source-checked",
	);
}
const files = new Map();
const out = (name, value) =>
	files.set(name, value.endsWith("\n") ? value : value + "\n");
const byRoute = (r) => sorted(data.nodes.filter((n) => n.routeId === r.id));
const rels = (r) =>
	data.relations.filter(
		(e) =>
			nodes.get(e.fromId).routeId === r.id &&
			nodes.get(e.toId).routeId === r.id,
	);
const words = (s, max) => {
	const lines = [];
	let line = "";
	let width = 0;
	for (const ch of s) {
		const w = /[\x00-\x7f]/.test(ch) ? 0.55 : 1;
		if (width + w > max) {
			lines.push(line);
			line = "";
			width = 0;
		}
		line += ch;
		width += w;
	}
	if (line) lines.push(line);
	return lines;
};
const textLines = (s, x, y, max, size = 15, gap = 20, color = "#172033") =>
	words(s, max)
		.map(
			(line, i) =>
				`<text x="${x}" y="${y + i * gap}" font-size="${size}" fill="${color}">${esc(line)}</text>`,
		)
		.join("");
const svgStart = (w, h, title) =>
	`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title><style>text{font-family:system-ui,-apple-system,'PingFang SC','Microsoft YaHei',sans-serif}a:focus rect,a:focus circle{stroke:#db2777;stroke-width:4}a:hover text{fill:#1d4ed8}</style><rect width="100%" height="100%" fill="#fcfbf7"/>`;
const linkStart = (n) =>
	`<a href="../atlas.html#${n.id}" data-node="${n.id}" aria-label="${esc(n.title)}"><title>${esc(n.question)}</title>`;
const firstYear =
	Math.floor(Math.min(...data.nodes.map((n) => n.year)) / 10) * 10;
const lastYear =
	Math.ceil(Math.max(...data.nodes.map((n) => n.year)) / 10) * 10;
function timeline(r) {
	const list = byRoute(r),
		w = 1400,
		h = 170 + list.length * 76;
	const axisLeft = 650,
		axisRight = 1320;
	const x = (y) =>
		axisLeft +
		((y - firstYear) / (lastYear - firstYear)) * (axisRight - axisLeft);
	let s = svgStart(w, h, `${r.name}：等比例时间线`);
	s +=
		textLines(r.name, 36, 40, 50, 25, 30) +
		textLines(
			"年份使用共同等比例坐标；纵向分行用于阅读，不表示因果。",
			36,
			74,
			80,
			15,
		);
	for (let year = firstYear; year <= lastYear; year += 10)
		s += `<path d="M${x(year)} 110V${h - 40}" stroke="#dce2e7"/><text x="${x(year)}" y="103" text-anchor="middle" font-size="13" fill="#576273">${year}</text>`;
	list.forEach((n, i) => {
		const y = 140 + i * 76;
		s +=
			linkStart(n) +
			`<rect x="24" y="${y - 22}" width="1352" height="68" rx="8" fill="${i % 2 ? "#f0f3f1" : "#ffffff"}" fill-opacity=".78"/>`;
		s +=
			`<text x="38" y="${y}" font-size="14" fill="${r.color}">${n.dateLabel.startsWith("1985") ? "1985/86" : n.year}</text>` +
			textLines(n.title, 120, y, 32, 16, 22) +
			`<text x="120" y="${y + 43}" font-size="12" fill="#64748b">${esc(n.branch)} · 指定主张审校</text><path d="M${axisLeft} ${y + 8}H${x(n.year)}" stroke="${r.color}" stroke-opacity=".25"/><circle cx="${x(n.year)}" cy="${y + 8}" r="7" fill="${r.color}"/><text x="${x(n.year) + 13}" y="${y + 13}" font-size="13" fill="${r.color}">${n.year}</text></a>`;
	});
	return (
		s +
		`<text x="36" y="${h - 15}" font-size="12" fill="#64748b">2026-10-25 · 仅显示本轮样板；不是该路线的完整历史 · 点击节点查看文献与边界</text></svg>`
	);
}
function branches(r) {
	const list = byRoute(r),
		groups = [...new Set(list.map((n) => n.branch))],
		relations = rels(r);
	const col = 256,
		cardW = 210,
		row = 128,
		w = Math.max(1040, 70 + groups.length * col),
		h = 180 + list.length * row;
	const positions = new Map(
		list.map((n, i) => [
			n.id,
			{ x: 30 + groups.indexOf(n.branch) * col, y: 140 + i * row },
		]),
	);
	let s = svgStart(w, h, `${r.name}：分支关系图`);
	s += `<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0L7,3.5L0,7" fill="#475569"/></marker></defs>`;
	s +=
		textLines(`${r.name} · 分支与关系`, 30, 38, 60, 23) +
		textLines(
			"列为研究分支；纵向仅按年代排序，间距不代表时间。R编号对应下方关系证据表。",
			30,
			74,
			90,
			14,
		);
	groups.forEach((g, i) => {
		const x = 20 + i * col;
		s += `<rect x="${x}" y="94" width="240" height="${h - 115}" rx="12" fill="${i % 2 ? "#edf2f3" : "#f4eee3"}"/><text x="${x + 14}" y="122" font-size="16" fill="${r.color}">${esc(g)}</text>`;
	});
	relations.forEach((e, i) => {
		const a = positions.get(e.fromId),
			b = positions.get(e.toId),
			y1 = a.y + 88,
			y2 = b.y - 8;
		const side = a.x + cardW + 15 + (i % 3) * 5;
		const d = `M${a.x + cardW / 2} ${y1}V${y1 + 13}H${side}V${y2 - 10 - (i % 3) * 8}H${b.x + cardW / 2}V${y2}`;
		s += `<path d="${d}" fill="none" stroke="${edgeColor(e)}" stroke-width="1.6" ${e.status === "editorial" ? 'stroke-dasharray="6 5"' : ""} marker-end="url(#arrow)"/><text x="${side + 3}" y="${(y1 + y2) / 2 + 3}" font-size="10" fill="${edgeColor(e)}">R${i + 1}</text>`;
	});
	list.forEach((n) => {
		const p = positions.get(n.id);
		s +=
			linkStart(n) +
			`<rect x="${p.x}" y="${p.y}" width="${cardW}" height="88" rx="8" fill="#ffffff" stroke="${r.color}" stroke-opacity=".45"/><text x="${p.x + 12}" y="${p.y + 20}" font-size="12" fill="${r.color}">${esc(n.dateLabel)}</text>` +
			textLines(n.title, p.x + 12, p.y + 42, 13, 13, 18) +
			"</a>";
	});
	return (
		s +
		`<text x="30" y="${h - 14}" font-size="12" fill="#64748b">实线：原文可定位的关系（引用不等于影响） · 红线：回应／批评 · 虚线：本站比较 · 无线不代表无关联</text></svg>`
	);
}
const noteLink = (n, prefix = "notes/") =>
	`[${n.year} · ${n.title}](${prefix}${n.id}.md)`;
const detailsMarkdown = (n) =>
	`# ${n.title}\n\n事件时间：${n.dateLabel}（${n.dateBasis}）。状态：指定主张已审校；阅读级别：指定片段。\n\n[原文入口](${n.source.url}) · ${n.source.version}\n\n实际阅读：${n.source.readScope}\n\n## 问题\n\n${n.question}\n\n## 机制与前提\n\n${n.mechanism}\n\n${n.assumptions}\n\n## 主张与原文定位\n\n${n.claims.map((c) => `- **${c.id} / 作者报告**：${c.text}\n  - 原文定位：${c.locator}。本轮未独立复现。`).join("\n")}\n\n## 边界与批评\n\n${n.limits}\n\n${n.criticism}\n\n## 教学例子与后续问题\n\n教学构想（除[已运行的三个算例](../demos.md)外，不宣称已运行）：${n.teachingExample}\n\n下一问：${n.nextQuestion}\n\n## 版本与维护\n\n- 阅读记录日期：${n.source.checkedAt}。${n.source.verification}\n- 本机阅读文件 SHA-256：${n.source.sha256 ?? "无：通过网页正文／转录阅读，未保存可校验PDF。"}\n- 第一轮对应事件：${n.baselineEventId ?? "本轮新增"}。本轮记录不改变第一轮当时的阅读状态。\n- 图表和笔记统一由 [research.json](../research.json) 生成。原PDF临时阅读，不随笔记重发。\n\n[路线叙事](../${n.routeId}.md) · [阅读工作台](../atlas.html#${n.id})\n`;
for (const n of data.nodes) out(`notes/${n.id}.md`, detailsMarkdown(n));
const svg = {};
for (const r of data.routes) {
	assert.ok(byRoute(r).length >= 8, `${r.id}样板缺节点`);
	svg[r.id] = { timeline: timeline(r), branches: branches(r) };
	out(`figures/${r.id}-timeline.svg`, svg[r.id].timeline);
	out(`figures/${r.id}-branches.svg`, svg[r.id].branches);
	out(
		`figures/${r.id}-timeline.mmd`,
		`%% 顺序时间线；Mermaid间距不代表时间。精确比例见SVG。\ntimeline\n    title ${r.name}\n${byRoute(
			r,
		)
			.map((n) => `    ${n.year} : ${n.title.replaceAll(":", "：")}`)
			.join("\n")}`,
	);
	out(
		`figures/${r.id}-branches.mmd`,
		`%% 箭头种类必须保留；无连线不代表无关联。\nflowchart TB\n${[
			...new Set(byRoute(r).map((n) => n.branch)),
		]
			.map(
				(b, i) =>
					`    subgraph g${i}["${b}"]\n${byRoute(r)
						.filter((n) => n.branch === b)
						.map(
							(n) =>
								`        ${n.id.replaceAll("-", "_")}["${n.year} ${n.title}"]`,
						)
						.join("\n")}\n    end`,
			)
			.join("\n")}\n${rels(r)
			.map(
				(e) =>
					`    ${e.fromId.replaceAll("-", "_")} ${e.status === "editorial" ? "-." : "--"} ${types[e.type]} ${e.status === "editorial" ? ".->" : "-->"} ${e.toId.replaceAll("-", "_")}`,
			)
			.join("\n")}`,
	);
	out(
		`${r.id}.md`,
		`# ${r.name}\n\n${r.question}\n\n${r.narrative.join("\n\n")}\n\n## 时间线\n\n![${r.name}时间线](figures/${r.id}-timeline.svg)\n\n[编辑时间线](figures/${r.id}-timeline.mmd) · [交互工作台](atlas.html)\n\n## 分支图\n\n![${r.name}分支图](figures/${r.id}-branches.svg)\n\n[编辑分支图](figures/${r.id}-branches.mmd)\n\n## 节点与文献夹\n\n${byRoute(
			r,
		)
			.map((n) => `- ${noteLink(n)}：${n.question}`)
			.join(
				"\n",
			)}\n\n## 图中关系证据\n\n| 编号 | 两端 | 关系 | 依据与限制 |\n| --- | --- | --- | --- |\n${rels(
			r,
		)
			.map(
				(e, i) =>
					`| R${i + 1} | ${noteLink(nodes.get(e.fromId))} → ${noteLink(nodes.get(e.toId))} | ${types[e.type]} | ${cell(e.locator)}；${cell(e.explanation)} |`,
			)
			.join(
				"\n",
			)}\n\n## 仍需补齐\n\n${r.gaps.map((g) => `- ${g}`).join("\n")}\n`,
	);
}
out(
	"claims.md",
	`# 主张台账\n\n这里只列经过原文定位的作者主张；原文表达与独立验证是两回事。解释、限制与批评见对应笔记。\n\n| 主张ID | 节点 | 作者报告 | 原文定位 | 独立复现 |\n| --- | --- | --- | --- | --- |\n${data.nodes.flatMap((n) => n.claims.map((c) => `| ${c.id} | ${noteLink(n)} | ${cell(c.text)} | ${cell(c.locator)} | 未完成 |`)).join("\n")}`,
);
out(
	"relations.md",
	`# 关系证据台账\n\n方向统一为早期工作 → 后续工作。批评箭头指向提出批评的后续工作。引用不自动提升为历史影响；本站比较是提问工具。\n\n| 两端 | 类型 | 定位 | 关系成立范围 |\n| --- | --- | --- | --- |\n${data.relations.map((e) => `| ${noteLink(nodes.get(e.fromId))} → ${noteLink(nodes.get(e.toId))} | ${types[e.type]} | ${cell(e.locator)} | ${cell(e.explanation)} |`).join("\n")}`,
);
out(
	"literature.md",
	`# 第二轮文献夹\n\n${data.nodes.length}份节点笔记均为指定片段阅读；没有把任何一份自动标为全文完成。来源版本、页码和阅读文件指纹在笔记中。未入选的下载尝试见[访问日志](evidence/access-log.json)。\n\n| 路线 | 阅读笔记 | 实际阅读范围 |\n| --- | --- | --- |\n${data.nodes.map((n) => `| ${data.routes.find((r) => r.id === n.routeId).name} | ${noteLink(n)} | ${cell(n.source.readScope)} |`).join("\n")}`,
);
out(
	"demos.md",
	`# 三个可运行教学算例\n\n结果来源：本站实际运行的精确枚举／确定性更新。**这不是论文复现。** 数据见[demos.json](evidence/demos.json)，重新运行：\n\n\`\`\`sh\nnode scripts/intelligence-history-demos.mjs\nnode scripts/intelligence-history-demos.mjs --check\n\`\`\`\n\n## 1. 训练表现不能唯一决定外推\n\n三个原语、三个重复次数构成9例；留出jump的2次、3次组合。两个人工指定的假设都在7个训练例上完全正确；组合规则在2个测试例上全对，只对jump拒绝组合的规则全错。没有训练神经网络，结果只说明相同训练拟合允许不同外推。\n\n## 2. TD预测沿路径传播\n\nA→B→终点，只有最后一步奖励1；步长0.5、折扣1，A和B初值0。解析目标均为1。\n\n| 完成轮数 | A估计 | B估计 |\n| --- | --- | --- |\n${demos.temporalDifference.trace.map((x) => `| ${x.episode} | ${x.A.toFixed(8)} | ${x.B.toFixed(8)} |`).join("\n")}\n\n运行30轮后两者与解析目标的差小于百万分之一。这是预测例，不检验Q-learning策略收敛或函数逼近。\n\n## 3. 条件互信息可能包含协同\n\nW和A是独立公平二元变量，精确枚举四种组合，单位为比特。\n\n| 下一状态 | I(下一状态;W) | I(下一状态;A) | 联合信息 | I(下一状态;W\|A) |\n| --- | --- | --- | --- | --- |\n${demos.information.rows.map((x) => `| ${x.mode} | ${x.worldInformation} | ${x.actionInformation} | ${x.jointInformation} | ${x.conditionalWorldInformation} |`).join("\n")}\n\n异或情况下，两个输入单独都不提供信息，合起来提供1比特；条件互信息仍为1。因此不能直接把它解释成世界的独有贡献。本例不估计干预因果效应，也没有实现完整的信息分解优化算法。`,
);
const payload = JSON.stringify({ ...data, svg, types }).replaceAll(
	"<",
	"\\u003c",
);
out(
	"atlas.html",
	`<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>智能是什么 · 第二轮研究工作台</title><style>
*{box-sizing:border-box}body{margin:0;background:#fcfbf7;color:#172033;font:16px/1.75 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1540px;margin:auto;padding:32px 22px 80px}h1{font-size:clamp(28px,4vw,44px);line-height:1.3;margin:8px 0 16px}h2{font-size:23px;margin:30px 0 14px}h3{font-size:18px;margin:24px 0 8px}p{margin:10px 0}a{color:#1d4ed8;overflow-wrap:anywhere}small,.muted{color:#596579}.eyebrow{letter-spacing:.14em;font-size:13px;color:#047857}.lead{font-size:19px;max-width:900px}nav,.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:20px 0}select,button{font:inherit;padding:9px 13px;border:1px solid #cbd5e1;border-radius:8px;background:white;color:inherit;max-width:100%}button{cursor:pointer}button[aria-pressed=true]{background:#172033;color:white}select:focus,button:focus,a:focus{outline:3px solid #fb923c;outline-offset:3px}.panel{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:24px;margin-top:20px}.figure{overflow:auto;border:1px solid #d8e0e5;border-radius:12px;background:white}.figure svg{display:block;width:100%;min-width:1100px;height:auto}.figure[data-mode=branches] svg{min-width:1400px}.caption{font-size:14px;color:#596579}.directory{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;padding:0;list-style:none}.directory button{text-align:left;width:100%;height:100%}.directory small{display:block}.claim{border-left:4px solid #2563eb;background:#eff6ff;padding:14px 18px}.warning{border-left:4px solid #b45309;padding:4px 16px}.badge{display:inline-block;font-size:13px;border-radius:20px;padding:2px 10px;background:#e8f2ec;color:#065f46}.relation{padding:14px 0;border-bottom:1px solid #e2e8f0}.relation a{font-weight:600}.source{background:#f8fafc;padding:16px;border-radius:10px}.hash{font:12px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}#detail:focus{outline:none}.links a{margin-right:16px}@media(max-width:600px){main{padding:22px 14px}.panel{padding:16px}.directory{grid-template-columns:1fr}nav{gap:8px}}
</style><main><div class="eyebrow">智能问题演进史 / 第二轮</div><h1>从问题出发，追踪解释如何改变</h1><p class="lead">三条样板路线 · ${data.nodes.length} 个节点 · ${data.relations.length} 条有类型的关系。把问题、机制、证据与边界放回同一条阅读路径。</p><p class="muted">2026-10-25 · 所列主张已核对原文指定片段；论文实验未独立复现。未显示的工作不表示不重要。</p><nav class="links"><a href="README.md">研究说明</a><a href="literature.md">文献夹</a><a href="claims.md">主张台账</a><a href="relations.md">关系依据</a><a href="demos.md">三个教学算例</a><a href="research.json">可编辑研究数据</a></nav><label for="route">选择路线</label> <select id="route">${data.routes.map((r) => `<option value="${r.id}">${r.name}</option>`).join("")}</select><section id="intro" class="panel"></section><div class="controls"><button id="timeline" aria-pressed="true">等比例时间线</button><button id="branches" aria-pressed="false">分支关系图</button><a id="sourceDiagram" href="#">可编辑源图</a><a id="svgDownload" href="#">打开 SVG</a></div><p id="graphCaption" class="caption"></p><div id="figure" class="figure" tabindex="0" aria-label="可横向滚动的研究图"></div><h2>节点目录</h2><p class="caption">图中节点与目录均可打开详情；键盘可使用目录按钮浏览。</p><ul id="directory" class="directory"></ul><section id="detail" class="panel" tabindex="-1" aria-live="polite"></section><section class="panel"><h2>图中关系与原文依据</h2><div id="relations"></div></section><section id="gaps" class="warning"></section><p class="caption">第一轮20条路线候选基线仍独立保留。本页是第二轮研究视图；图和笔记统一从研究数据生成。</p></main><script id="data" type="application/json">${payload}</script><script>
const d=JSON.parse(document.getElementById('data').textContent), el=id=>document.getElementById(id), nodeMap=new Map(d.nodes.map(n=>[n.id,n]));
const e=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let mode='timeline';
function selectNode(id,focus=true){const n=nodeMap.get(id);if(!n)return;if(el('route').value!==n.routeId){el('route').value=n.routeId;render(false)}
el('detail').innerHTML='<span class="badge">指定主张已审校 · 指定片段阅读</span><h2>'+e(n.title)+'</h2><p class="muted">'+e(n.dateLabel)+' · '+e(n.branch)+'</p><h3>当时的问题</h3><p>'+e(n.question)+'</p><h3>解决思路与前提</h3><p>'+e(n.mechanism)+'</p><p>'+e(n.assumptions)+'</p>'+n.claims.map(c=>'<div class="claim"><strong>作者报告</strong><p>'+e(c.text)+'</p><small>定位：'+e(c.locator)+'；本轮未独立复现。</small></div>').join('')+'<h3>边界与批评</h3><p>'+e(n.limits)+'</p><p>'+e(n.criticism)+'</p><h3>帮助理解的例子</h3><p>'+e(n.teachingExample)+'</p><small>这是教学构想；已运行范围仅限<a href="demos.md">三个教学算例</a>。</small><h3>留下的新问题</h3><p>'+e(n.nextQuestion)+'</p><div class="source"><a target="_blank" rel="noopener" href="'+e(n.source.url)+'">阅读原文</a> · <a href="notes/'+n.id+'.md">打开文献笔记</a><p>'+e(n.source.version)+'</p><p>实际阅读：'+e(n.source.readScope)+'</p><small>读取日期：'+e(n.source.checkedAt)+'</small><p class="hash">文件指纹：'+e(n.source.sha256||'网页阅读，无本机PDF指纹')+'</p></div>';
history.replaceState(null,'','#'+id);if(focus){el('detail').focus();el('detail').scrollIntoView({behavior:'smooth',block:'start'})}}
function render(selectFirst=true){const r=d.routes.find(r=>r.id===el('route').value),ns=d.nodes.filter(n=>n.routeId===r.id).sort((a,b)=>a.year-b.year||a.id.localeCompare(b.id));
el('intro').innerHTML='<h2>'+e(r.question)+'</h2>'+r.narrative.map(p=>'<p>'+e(p)+'</p>').join('');
el('figure').innerHTML=d.svg[r.id][mode];el('figure').dataset.mode=mode;el('sourceDiagram').href='figures/'+r.id+'-'+mode+'.mmd';el('svgDownload').href='figures/'+r.id+'-'+mode+'.svg';
el('timeline').setAttribute('aria-pressed',mode==='timeline');el('branches').setAttribute('aria-pressed',mode==='branches');el('graphCaption').textContent=mode==='timeline'?'三条路线使用共同等比例年份坐标。图较宽时可在图内滚动。':'纵向仅按年代排序。箭头指向后续工作；红线表示回应／批评，虚线表示本站比较。R编号与下表逐项对应。';
el('directory').innerHTML=ns.map(n=>'<li><button data-node="'+n.id+'">'+e(n.title)+'<small>'+e(n.dateLabel)+' · '+e(n.branch)+'</small></button></li>').join('');
const rr=d.relations.filter(x=>nodeMap.get(x.fromId).routeId===r.id&&nodeMap.get(x.toId).routeId===r.id);
el('relations').innerHTML=rr.map((x,i)=>'<div class="relation"><strong>R'+(i+1)+' · '+e(d.types[x.type])+'</strong><p><a href="#'+x.fromId+'" data-node="'+x.fromId+'">'+e(nodeMap.get(x.fromId).title)+'</a> → <a href="#'+x.toId+'" data-node="'+x.toId+'">'+e(nodeMap.get(x.toId).title)+'</a></p><p>'+e(x.explanation)+'</p><small>依据：'+e(x.locator)+'</small></div>').join('')+'<p class="caption">跨路线比较另见<a href="relations.md">完整关系台账</a>。引用和方法联系均不自动等于历史影响。</p>';
el('gaps').innerHTML='<h2>仍需补齐</h2><ul>'+r.gaps.map(x=>'<li>'+e(x)+'</li>').join('')+'</ul>';if(selectFirst)selectNode(ns[0].id,false)}
document.addEventListener('click',ev=>{const a=ev.target.closest('[data-node]');if(a){ev.preventDefault();selectNode(a.dataset.node)}});el('route').addEventListener('change',()=>render());for(const m of ['timeline','branches'])el(m).addEventListener('click',()=>{mode=m;render(false)});const initial=location.hash.slice(1);render();if(nodeMap.has(initial))selectNode(initial,false);window.addEventListener('hashchange',()=>{const id=location.hash.slice(1);if(nodeMap.has(id))selectNode(id,false)});
</script></html>`,
);
out(
	"README.md",
	`# 第二轮：三条样板路线\n\n2026-10-25。已形成 **${data.nodes.length} 个指定主张审校节点、${data.relations.length} 条分类型关系、25份文献笔记、三组时间线与分支图**。这是样板批次，不是三条路线已穷尽。全部来源均诚实记录为指定片段阅读；没有声称论文全文、证明附录或实验全部验证。\n\n首选入口：[交互研究工作台](atlas.html)。\n\n| 路线 | 节点 | 阅读入口 |\n| --- | --- | --- |\n${data.routes.map((r) => `| ${r.name} | ${byRoute(r).length} | [路线叙事与两种图](${r.id}.md) |`).join("\n")}\n\n## 文献与证据\n\n- [文献夹](literature.md)：逐篇阅读范围、版本、原文定位、批评和后续问题。\n- [主张台账](claims.md)、[关系台账](relations.md)：作者报告、本站比较与独立验证分开。\n- [三个教学算例](demos.md)：训练样例与外推、TD传播、条件信息与协同；均不冒称论文复现。\n- [审校修订与异议](review.md)：版本变化、未选候选和下一轮优先问题。\n- [验收记录](validation.md)：数据、图、浏览器与项目检查。\n\n## 编辑与增补\n\n本轮唯一编辑源是 [research.json](research.json)。节点复用第一轮事件ID，新增来源有独立ID。第一轮 [atlas.json](../atlas.json)作为历史基线保存；来源现在读过，不会把第一轮当时的状态改为已读。\n\n\`\`\`sh\nnode scripts/intelligence-history-demos.mjs\nnode scripts/generate-intelligence-history-round-two.mjs\nnode scripts/intelligence-history-demos.mjs --check\nnode scripts/generate-intelligence-history-round-two.mjs --check\n\`\`\`\n\n新增时补齐事件、实际阅读范围、带定位的主张和边界；再按证据添加关系。图、卡片与台账同步重生成。Mermaid与SVG是可编辑源图；长期维护应修改研究数据，避免重生成覆盖手工图形修改。\n\n本轮延续项目研究工程方法：问题→机制与前提→证据→边界与批评→下一问。专著按章阅读，不以下载数冒充阅读数；PDF仅临时用于阅读，笔记提供公开入口及可用指纹。\n\n[长期方案](../../intelligence-history-plan.md) · [第一轮底稿](../round-one.md)\n`,
);
for (const [name, content] of files) {
	const target = path.join(dir, name);
	if (check)
		assert.equal(
			fs.readFileSync(target, "utf8"),
			content,
			`生成内容不同：${name}`,
		);
	else {
		fs.mkdirSync(path.dirname(target), { recursive: true });
		fs.writeFileSync(target, content);
	}
}
console.log(
	`${check ? "检查" : "生成"} ${files.size} 个文件；${data.nodes.length}节点，${data.relations.length}关系。第一轮基线保持一致。`,
);
