const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const relationLabels = {
	"conceptual-comparison": "概念比较（本站）",
	"method-comparison": "方法比较（本站）",
	"architecture-comparison": "架构比较",
	"direct-critique": "直接批评",
	"direct-response": "直接回应",
	"editorial-comparison": "本站比较",
	"empirical-response": "实证回应",
	"explicit-adoption": "明确采用",
	"motivated-evaluation": "转为评测问题",
};
function wrap(value, limit) {
	const lines = [];
	let line = "", width = 0;
	for (const ch of String(value)) {
		const size = /[\x00-\x7f]/.test(ch) ? 0.6 : 1;
		if (width + size > limit) { lines.push(line); line = ""; width = 0; }
		line += ch; width += size;
	}
	if (line) lines.push(line);
	return lines;
}
const text = (value, x, y, width, size = 16, lineHeight = 23, color = "#203d3a") => wrap(value, width)
	.map((line, i) => `<text x="${x}" y="${y + i * lineHeight}" font-size="${size}" fill="${color}">${escape(line)}</text>`).join("");
const start = (width, height, title) => `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(title)}"><title>${escape(title)}</title><style>text{font-family:system-ui,-apple-system,'PingFang SC','Microsoft YaHei',sans-serif}a:hover rect,a:focus rect{stroke:#a75532;stroke-width:3}a:focus{outline:none}</style><rect width="100%" height="100%" fill="#faf9f4"/>`;
const link = (n) => `<a href="../atlas.html#${n.id}" data-node="${n.id}" aria-label="${escape(n.title)}"><title>${escape(n.question)}</title>`;
const state = (n) => n.previousRound ? "第二轮延续" : "B批审校节点";

export function createResearchViews(data, previous) {
	const completed = data.status === "completed";
	const files = new Map(), relationTables = new Map(), figures = {};
	const nodes = new Map(previous.nodes.map((n) => [n.id, { ...n, routeIds: [n.routeId], previousRound: true }]));
	for (const n of data.nodes) nodes.set(n.id, n);
	const sources = new Map(data.sources.map((s) => [s.id, s]));
	const routeNodes = {}, routeRelations = {};
	for (const route of data.routes) {
		const current = data.nodes.filter((n) => n.routeIds.includes(route.id));
		const ids = new Set(current.map((n) => n.id));
		for (const edge of data.relations) {
			if (!ids.has(edge.fromId) && !ids.has(edge.toId)) continue;
			for (const id of [edge.fromId, edge.toId]) {
				const n = nodes.get(id);
				if (n.previousRound && n.routeIds.includes(route.id)) ids.add(id);
			}
		}
		routeNodes[route.id] = [...ids].map((id) => nodes.get(id)).sort((a, b) => a.year - b.year || a.id.localeCompare(b.id));
		routeRelations[route.id] = data.relations.filter((e) => ids.has(e.fromId) && ids.has(e.toId));
	}
	const plotted = Object.values(routeNodes).flat();
	const first = Math.floor(Math.min(...plotted.map((n) => n.year)) / 10) * 10;
	const last = Math.max(first + 10, Math.ceil(Math.max(...plotted.map((n) => n.year)) / 10) * 10);
	for (const route of data.routes) {
		const list = routeNodes[route.id], edges = routeRelations[route.id];
		const height = 180 + list.length * 104;
		let svg = start(1400, height, `${route.name}：B批时间线`);
		svg += text(route.name, 32, 43, 55, 26) + text("共同等比例年份轴 · 本批审读节点 · 纵向分行不表示因果", 32, 78, 85, 15);
		const yearX = (year) => 690 + (year - first) / (last - first) * 610;
		for (let year = first; year <= last; year += 10) svg += `<path d="M${yearX(year)} 114V${height - 44}" stroke="#d6ded8"/><text x="${yearX(year)}" y="106" font-size="13" text-anchor="middle" fill="#58695f">${year}</text>`;
		list.forEach((n, i) => {
			const y = 145 + i * 104;
			svg += link(n) + `<rect x="24" y="${y - 23}" width="1352" height="94" rx="9" fill="${i % 2 ? "#f0f2eb" : "#fff"}" fill-opacity=".85"/>`;
			svg += text(n.year, 42, y, 7, 18) + text(n.title, 120, y, 32, 16, 23);
			svg += text(`${n.branch} · ${state(n)}`, 120, y + 53, 45, 12, 16, "#596b62");
			svg += `<path d="M690 ${y + 13}H${yearX(n.year)}" stroke="#5f8276" stroke-opacity=".4"/><circle cx="${yearX(n.year)}" cy="${y + 13}" r="7" fill="${n.previousRound ? "#a75532" : "#356a5c"}"/><text x="${yearX(n.year) + 13}" y="${y + 18}" font-size="13" fill="#356a5c">${n.year}</text></a>`;
		});
		svg += text("时间采用事件年份；所读版本与阅读范围在节点详情中分别记录。", 32, height - 18, 100, 13) + "</svg>";
		files.set(`routes/${route.id}-timeline.svg`, svg);
		const width = Math.max(1100, 790 + edges.length * 64), graphHeight = 210 + list.length * 146;
		let graph = start(width, graphHeight, `${route.name}：分支与关系`);
		graph += '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0,0L7,3.5L0,7" fill="context-stroke"/></marker></defs>';
		graph += text(`${route.name} · 分支与关系`, 32, 42, 65, 25) + text("卡片注明分支；纵向按年代排列，间距不表示时间。右侧R编号对应关系证据表。", 32, 78, 95, 14);
		const positions = new Map(list.map((n, i) => [n.id, 130 + i * 146]));
		edges.forEach((e, i) => {
			const y1 = positions.get(e.fromId) + 48, y2 = positions.get(e.toId) + 65, lane = 730 + i * 64;
			const color = /critique|response/.test(e.type) ? "#a44939" : "#4b736d";
			graph += `<g><title>${escape(relationLabels[e.type] ?? e.type)}：${escape(e.explanation)}</title><path d="M664 ${y1}H${lane}V${y2}H668" fill="none" stroke="${color}" stroke-width="1.7" ${e.status === "editorial-inference" ? 'stroke-dasharray="6 5"' : ""} marker-end="url(#arrow)"/><text x="${lane + 6}" y="${(y1 + y2) / 2}" font-size="12" fill="${color}">R${i + 1}</text></g>`;
		});
		list.forEach((n) => {
			const y = positions.get(n.id);
			graph += link(n) + `<rect x="34" y="${y}" width="630" height="112" rx="10" fill="#fff" stroke="#9bae9e"/>`;
			graph += text(`${n.year} · ${n.branch} · ${state(n)}`, 52, y + 25, 47, 13, 17, "#5c6e62") + text(n.title, 52, y + 58, 31, 18, 25) + "</a>";
		});
		graph += text("实线：可定位的关系；红线：批评或回应；虚线：本站比较。引用不自动等于思想影响。", 32, graphHeight - 42, 95, 13);
		graph += text("没有连线表示本批尚未确认关系；不能据此判断历史上不存在联系。", 32, graphHeight - 18, 95, 13) + "</svg>";
		files.set(`routes/${route.id}-relations.svg`, graph);
		figures[route.id] = { timeline: svg.replaceAll("../atlas.html#", "#"), relations: graph.replaceAll("../atlas.html#", "#") };
		relationTables.set(route.id, edges.length ? `| 编号 | 关系 | 解释 |\n| --- | --- | --- |\n${edges.map((e, i) => `| R${i + 1} | ${nodes.get(e.fromId).year} → ${nodes.get(e.toId).year} · ${relationLabels[e.type] ?? e.type} | ${e.explanation.replaceAll("|", "／")} |`).join("\n")}` : "本批尚未确认本路线内部的关系边；节点先按年代保留。");
	}
	const uiNodes = [...new Set(plotted.map((n) => n.id))].map((id) => {
		const n = nodes.get(id);
		return { ...n, sources: n.source ? [n.source] : n.sourceIds.map((id) => sources.get(id)) };
	});
	const payload = JSON.stringify({ routes: data.routes, nodes: uiNodes, routeNodes, routeRelations, figures, relationLabels }).replaceAll("<", "\\u003c");
	const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>智能问题演进史 · B批研究工作台</title>
<style>
:root{font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#203d3a;background:#f7f7ef;line-height:1.7}*{box-sizing:border-box}body{margin:0}main{max-width:1600px;margin:auto;padding:28px}h1{font-size:clamp(25px,3vw,38px);line-height:1.3;margin:10px 0}h2{font-size:24px;line-height:1.5}h3{font-size:16px;margin-bottom:6px}p{margin:8px 0 16px}a{color:#276454;text-underline-offset:3px}button,select{font:inherit;color:inherit;border:1px solid #b4c1b5;background:#fff;border-radius:8px;padding:8px 13px}button{cursor:pointer}button[aria-pressed=true]{background:#285b4e;color:#fff}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #bb754d;outline-offset:3px}.eyebrow,.muted{color:#627368}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:22px 0}.toolbar select{max-width:100%}.workspace{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(320px,1fr);gap:22px;align-items:start}.panel{min-width:0;background:#fffdf7;border:1px solid #d7dfd2;border-radius:14px;padding:22px}#figure{max-height:640px;overflow:auto;border:1px solid #dce1d6;border-radius:9px}#figure svg{display:block;max-width:none}#directory{display:flex;flex-wrap:wrap;gap:7px;margin:16px 0}#directory button{font-size:13px;text-align:left}#detail{overflow-wrap:anywhere}#detail h2{margin-top:4px}.label{font-size:12px;color:#6b7868}.evidence{border-left:3px solid #a4b6a3;padding:0 0 0 14px;margin:18px 0}details{border-top:1px solid #dce2d4;padding-top:12px;margin-top:12px}summary{cursor:pointer}.table-scroll{max-width:100%;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #dce2d4;padding:10px;vertical-align:top}th{white-space:nowrap}.questions{display:flex;flex-wrap:wrap;gap:16px;margin:20px 0}.questions a{flex:1 1 250px;padding:16px;border:1px solid #c9d6c3;border-radius:10px;background:#eef2e6}#route-narrative{font-size:14px}#route-status{font-size:13px}.badge{color:#925631}noscript{display:block;background:#fff;padding:20px}@media(max-width:1000px){.workspace{grid-template-columns:1fr}}@media(max-width:600px){main{padding:16px}.panel{padding:14px}.toolbar{align-items:stretch}.toolbar label{width:100%}.toolbar select{width:100%}#figure{max-height:440px}h2{font-size:21px}}
</style></head><body><main>
<div class="eyebrow">智能是什么 · 第三轮 B 批 · ${completed ? "指定范围已验收" : "研究进行中"}</div><h1>从问题的转折，读懂路线的演进</h1>
<p>图中每个节点都连接问题、机制、原文定位与边界。当前${data.nodes.length}个本批节点，${data.sources.length}份来源阅读记录；${completed ? "B批已验收，第三轮其余批次待开展。" : "完整 B 批尚未验收。"}</p>
<p><a href="README.md">本批说明</a> · <a href="literature.md">文献夹</a> · <a href="claims.md">主张台账</a> · <a href="relations.md">关系证据</a> · <a href="method-check/README.md">合成方法核验</a> · <a href="evaluation.md">评价条件</a> · <a href="revisits.md">A批回访</a> · <a href="plan.md">范围与验收</a></p>
<div class="questions">${data.questionCards.map((q) => `<a href="questions/${q.id}.md">${escape(q.title)}</a>`).join("")}</div>
<div class="toolbar"><label for="route">研究路线</label><select id="route">${data.routes.map((r) => `<option value="${r.id}">${escape(r.name)}</option>`).join("")}</select><button id="timeline" aria-pressed="true">时间线</button><button id="relations" aria-pressed="false">分支与关系</button><a id="svg-link" href="routes/behavioral-timeline.svg">打开 SVG</a><a id="mmd-link" href="routes/behavioral-timeline.mmd">编辑 Mermaid</a></div>
<div class="workspace"><section class="panel" aria-label="路线图"><div id="route-narrative"></div><p id="route-status" class="muted"></p><p class="muted">图可横向与纵向滚动；下方目录可直接选择节点。</p><div id="figure" tabindex="0" aria-label="可滚动路线图"></div><nav id="directory" aria-label="节点目录"></nav><div id="edge-table" class="table-scroll"></div></section><article id="detail" class="panel" aria-live="polite"></article></div>
<noscript>请阅读<a href="literature.md">文献夹</a>和<a href="README.md">路线文档</a>；SVG图与笔记无需脚本即可打开。</noscript>
<p class="muted">指定片段阅读、原文主张与独立验证分别记录。图和笔记从同一份研究数据生成；先编辑数据，再同步重建。</p>
</main><script>
const data=${payload};
const byId=new Map(data.nodes.map(n=>[n.id,n]));
const $=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
let mode="timeline",selected=null;
const requested=new URLSearchParams(location.search).get("route");
if(data.routes.some(r=>r.id===requested))$("route").value=requested;
function noteHref(n){return (n.previousRound?"../round-two/notes/":"notes/")+n.id+".md";}
function renderRoute(){
 const rid=$("route").value,route=data.routes.find(r=>r.id===rid),list=data.routeNodes[rid],edges=data.routeRelations[rid];
 $("route-narrative").innerHTML=route.narrative.map(p=>"<p>"+esc(p)+"</p>").join("");
 $("route-status").textContent="本批节点 "+list.filter(n=>!n.previousRound).length+" · ${completed ? "已完成本批指定范围验收" : "尚未完成该路线批次验收"}";
 $("figure").innerHTML=data.figures[rid][mode];
 $("directory").innerHTML=list.map(n=>'<button type="button" data-node="'+n.id+'" aria-pressed="'+(n.id===selected)+'">'+esc(n.year+" · "+n.title)+'</button>').join("");
 $("svg-link").href="routes/"+rid+"-"+mode+".svg";$("mmd-link").href="routes/"+rid+"-"+mode+".mmd";
 ["timeline","relations"].forEach(m=>$(m).setAttribute("aria-pressed",String(m===mode)));
 $("edge-table").innerHTML=edges.length?'<table><thead><tr><th>编号／关系</th><th>说明与证据</th></tr></thead><tbody>'+edges.map((e,i)=>'<tr><td>R'+(i+1)+' · '+esc(data.relationLabels[e.type]||e.type)+'</td><td>'+esc(byId.get(e.fromId).title)+" → "+esc(byId.get(e.toId).title)+'<br>'+esc(e.explanation)+'<br><span class="muted">'+esc(e.locator)+'</span></td></tr>').join("")+'</tbody></table>':'<p class="muted">本批尚未确认本路线内部的关系边。</p>';
}
function show(id){
 const n=byId.get(id);if(!n)return;
 const rid=$("route").value;
 if(!data.routeNodes[rid].some(x=>x.id===id))$("route").value=data.routes.find(r=>data.routeNodes[r.id].some(x=>x.id===id)).id;
 selected=id;renderRoute();
 const label=n.previousRound?"第二轮延续记录":"B批指定主张审校";
 let h='<p class="label">'+esc(n.dateLabel+" · "+label)+'</p><h2>'+esc(n.title)+'</h2>';
 for(const [key,title] of [["question","探究的问题"],["mechanism","方法与机制"],["assumptions","成立条件"],["limits","证据边界"],["nextQuestion","下一问"]])h+='<h3>'+title+'</h3><p>'+esc(n[key])+'</p>';
 if(n.evaluation){h+='<h3>评价条件</h3><div class="table-scroll"><table><tbody>';for(const [key,title] of [["object","评价对象"],["task","任务与环境"],["sample","样本／主体"],["metric","输出指标"],["uncertainty","信度与不确定性"],["validity","效度与解释"],["priors","先验与经验"],["resources","计算与时间"]])h+='<tr><th>'+title+'</th><td>'+esc(n.evaluation[key])+'</td></tr>';h+='</tbody></table></div>';}
 h+='<h3>主张与原文定位</h3>' +n.claims.map(c=>{const s=n.sources.find(s=>s.id===c.sourceId)||n.sources[0];const kind=({'source-reported':'原文主张','derived':'本次推导或验证','editorial-inference':'本站分析'})[c.basis]||c.basis;return '<div class="evidence"><p class="label">'+esc(kind)+'</p><p>'+esc(c.text)+'</p><p class="label"><a href="'+esc(s.url)+'">'+esc(s.title||s.version)+'</a> · '+esc(c.locator)+' · '+(c.independentlyReplicated?'已验证所列范围，详见复现报告':'未独立复现')+'</p></div>';}).join("");
 h+='<details open><summary>阅读范围与来源版本</summary>'+n.sources.map(s=>'<div class="evidence"><a href="'+esc(s.url)+'">'+esc(s.title||s.version)+'</a><p>'+esc(s.version)+'</p><p><strong>已读：</strong>'+esc(s.readScope)+'</p><p><strong>未覆盖：</strong>'+esc(s.unreadScope||'其余证明、附录或实验未全面验证，完整记录见第二轮笔记。')+'</p><p class="label">'+(s.sha256?'文件指纹已保存于笔记':'经网页正文或转录阅读，无本地PDF指纹')+'</p></div>').join("")+'</details>';
 h+='<p><a href="'+noteHref(n)+'">完整笔记</a> · <a href="research.json">可编辑研究数据</a></p>';
 if(id==='e-kovacs-conway-2016')h+='<p><a href="method-check/README.md">本站合成核验（不是POT复现）</a></p>';
 if(n.versionComparisonId)h+='<p><a href="versions/'+esc(n.versionComparisonId)+'.md">版本对照与未覆盖范围</a></p>';
 $("detail").innerHTML=h;
}
function choose(id){show(id);const u=new URL(location.href);u.searchParams.set("route",$("route").value);u.hash=id;history.replaceState(null,"",u);}
$("route").addEventListener("change",()=>choose(data.routeNodes[$("route").value][0].id));
["timeline","relations"].forEach(m=>$(m).addEventListener("click",()=>{mode=m;renderRoute();}));
for(const id of ["directory","figure"])$(id).addEventListener("click",e=>{const a=e.target.closest("[data-node]");if(a){e.preventDefault();choose(a.dataset.node);}});
addEventListener("hashchange",()=>show(decodeURIComponent(location.hash.slice(1))));
const initial=decodeURIComponent(location.hash.slice(1));show(byId.has(initial)?initial:data.routeNodes[$("route").value][0].id);
</script></body></html>`;
	files.set("atlas.html", html);
	return { files, relationTables, routeNodes, routeRelations, relationLabels };
}
