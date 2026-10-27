import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { renderRoundFour } from "./lib/intelligence-history-round-four-views.mjs";

const root = path.resolve(import.meta.dirname, "../docs/intelligence-history/round-four");
const read = f => JSON.parse(fs.readFileSync(path.join(root, f), "utf8"));
const d = read("research.json"), results = read("method-check/results.json");
const annotations = read("../index-annotations.json");
const sha = p => crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
for (const [file, hash] of Object.entries(read("baselines.json"))) {
	if (sha(path.resolve(root, "..", file)) !== hash) throw Error(`Historical snapshot changed: ${file}`);
}
const ids = new Set(), claims = new Map(), sources = new Map(d.sources.map(s => [s.id, s]));
const routeIds = new Set(read("../atlas.json").routes.map(r => r.id));
for (const s of d.sources) {
	if (ids.has(s.id)) throw Error(`Duplicate source ${s.id}`);
	ids.add(s.id);
	for (const c of s.claims) {
		if (claims.has(c.id) || !c.locator || c.kind !== "source-reported") throw Error(`Invalid claim ${c.id}`);
		claims.set(c.id, { ...c, source: s });
	}
}
const conclusionIds = new Set(d.conclusions.map(c => c.id));
if (conclusionIds.size !== d.conclusions.length) throw Error("Duplicate conclusion");
for (const c of d.conclusions) {
	if (!c.evidence.every(e => claims.has(e)) || !c.routes.every(r => routeIds.has(r))) throw Error(`Broken evidence or route ${c.id}`);
	for (const field of ["statement", "assumptions", "objection", "response", "discriminator", "scope"]) if (!c[field]) throw Error(`Missing ${field}`);
}
for (const c of d.cases) if (!c.sources.every(s => sources.has(s)) || !c.propositions.every(p => conclusionIds.has(p))) throw Error(`Invalid case ${c.id}`);
for (const q of d.priorQuestionLinks) if (!fs.existsSync(path.resolve(root, "..", q.batch, "questions", q.id + ".md"))) throw Error(`Missing question ${q.id}`);
if (!Object.values(results.checks).every(Boolean) || results.protocolSHA256 !== sha(path.join(root, "method-check/protocol.md"))) throw Error("Method check invalid");
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const md = s => String(s ?? "").replaceAll("|", "／").replaceAll("\n", " ");
const header = "<!-- 由round-four/research.json生成；请修改源数据后重生成。 -->\n\n";
const files = new Map();
const evidence = (c, prefix = "") => c.evidence.map(id => {
	const v = claims.get(id); return `[${id}](${prefix}notes/${v.source.id}.md#${id})`;
}).join("、");
files.set("README.md", header + `# 第四轮第一阶段 ${d.title}\n\n${d.question}\n\n${d.scope}\n\n[交互综合页](index.html) · [完整综述](synthesis.md) · [比较框架](framework.md) · [比较矩阵](comparison.md) · [结论审计](conclusions.md) · [文献夹](literature.md) · [索引修订](index-audit.md) · [方法核验](method-check/README.md) · [后续研究方案](research-designs.md) · [验收记录](validation.md) · [阶段计划](plan.md)\n\n本阶段维护${d.sources.length}份来源记录、${claims.size}条原文定位主张、${d.conclusions.length}条条件性综合结论和${d.cases.length}个比较案例。来源包括补充材料与更正，不能把记录数当成独立研究数量。只有更正通知正文记为全文已读，其余保持选读状态。\n\n## 维护\n\n修改research.json后运行：\n\n\`\`\`sh\npython3 scripts/check-intelligence-round-four.py --check\nnode scripts/generate-intelligence-history-round-four.mjs\nnode scripts/generate-intelligence-history-index.mjs\n\`\`\`\n\n脚本从仓库根目录运行。图形同时提供SVG与Mermaid；改源数据后重生成，避免生成物手改被覆盖。当前覆盖注释维护在[独立注释层](../index-annotations.json)，旧七份研究数据的指纹见[基线](baselines.json)。\n`);
files.set("framework.md", header + `# 跨路线比较框架\n\n证据强弱取决于它是否能回答当前主张。经验结果、形式证明、概念论证不合并成一个分数；“干预”也不自动高于所有其他证据。自然病损、随机操纵与理想do算子分别登记。\n\n| 比较维度 | 要核对什么 | 推论边界 |\n| --- | --- | --- |\n${d.framework.map(x => `| ${x.name} | ${md(x.question)} | ${md(x.boundary)} |`).join("\n")}\n\n## 结论审计规则\n\n1. 先陈述目标主张，再决定何种证据相关，避免让手头指标替代目标。\n2. 原文主张链接具体版本与定位；本站推断明确列出桥接前提。\n3. 记录竞争解释的最强版本、何种证据会区分它们，以及尚未执行的检验。\n4. 将材料可得、实际阅读、原文报告和独立核验分开。\n5. 有限模型的反例可否定一般逻辑推论，不能替代实际机制的经验识别。\n\n## 可复用结论模板\n\n\`\`\`json\n${JSON.stringify({id:"stable-conclusion-id",title:"要回答的具体问题",statement:"带条件的结论",kind:"conditional-synthesis",evidence:["source-claim-id"],assumptions:"任务、对象、测量与模型假设",objection:"最强竞争解释",response:"保留或收窄结论的理由",discriminator:"可区分预测",scope:"适用与未检验边界",routes:["stable-route-id"]},null,2)}\n\`\`\`\n\n模板之外同时维护来源记录、实际阅读范围和证据审计项，见[可编辑数据](research.json)。\n`);
files.set("comparison.md", header + `# 表现到机制的比较矩阵\n\n[交互筛选](index.html#comparison) · [证据关系图](inference-map.svg)\n\n| 案例 | 分析单位／目标 | 方法与观察 | 支持范围 | 尚未识别 | 需对齐条件 |\n| --- | --- | --- | --- | --- | --- |\n${d.cases.map(c => `| ${c.name} | ${md(c.unit + "；" + c.target)} | ${md(c.design + "；" + c.observed)} | ${md(c.supports)} | ${md(c.doesNotIdentify)} | ${md(c.controls)} |`).join("\n")}\n\n${d.cases.map(c => `- **${c.name}**：${c.sources.map(s => `[${s}](notes/${s}.md)`).join("、") || "[本站方法核验](method-check/README.md)"}。对应${c.propositions.map(p => `[${p}](conclusions/${p}.md)`).join("、")}。`).join("\n")}\n`);
files.set("conclusions.md", header + `# 条件性结论与证据审计\n\n${d.conclusions.map(c => `- [${c.title}](conclusions/${c.id}.md)：${c.statement}`).join("\n")}\n\n## 证据适配与未决事项\n\n| 结论 | 证据种类 | 当前支持 | 未区分问题 | 核验状态 | 判定 |\n| --- | --- | --- | --- | --- | --- |\n${d.evidenceAudit.map(a => `| [${a.conclusion}](conclusions/${a.conclusion}.md) | ${md(a.evidence)} | ${md(a.supports)} | ${md(a.unresolved)} | ${md(a.verification)} | ${md(a.verdict)} |`).join("\n")}\n\n原文报告的结果没有被自动标成独立验证；本站形式反例只对协议中的模型与推论负责。\n`);
for (const c of d.conclusions) files.set(`conclusions/${c.id}.md`, header + `# ${c.title}\n\n${c.statement}\n\n类型：${c.kind}；[返回结论审计](../conclusions.md)。\n\n## 条件与推理\n\n${c.assumptions}\n\n## 最强异议与回应\n\n${c.objection}\n\n${c.response}\n\n## 可以区分解释的检验\n\n${c.discriminator}\n\n${c.scope}\n\n## 定位依据\n\n${evidence(c, "../")}\n\n${c.check ? "[本站方法核验](../method-check/README.md)" : "本条未进行独立经验数据复算。"}\n`);
files.set("literature.md", header + `# 第四轮文献夹\n\n${d.sources.length}份来源记录保存实际版本、读段、未读部分与定位主张；补充材料和更正单列。[访问日志](evidence/access-log.json)。\n\n| 来源 | 年份 | 阅读状态 | 本轮范围 |\n| --- | --- | --- | --- |\n${d.sources.map(s => `| [${md(s.title)}](notes/${s.id}.md) | ${s.year} | ${s.readingStatus} | ${md(s.readScope)} |`).join("\n")}\n\n原文文件只用于阅读；项目保存原创笔记与来源链接，不复制完整论文。版本指纹只标识实际取得的文件，不证明正文无误或已全文阅读。\n`);
for (const s of d.sources) files.set(`notes/${s.id}.md`, header + `# ${s.title}\n\n${s.authors.join("；")}，${s.year}。[原始来源](${s.url}) · [返回文献夹](../literature.md)\n\n## 版本与阅读范围\n\n${s.version}\n\n- 阅读状态：${s.readingStatus}（${s.checkedAt}）。\n- 已读：${s.readScope}\n- 未读或未核验：${s.unreadScope}\n- 文件SHA256：${s.sha256 ?? "未取得本地文件；网页转录无本地指纹"}。\n${s.prior ? `- [此前记录](../${s.prior})；旧阅读状态保持历史快照。\n` : ""}\n## 原文定位主张\n\n${s.claims.map(c => `<a id="${c.id}"></a>\n\n### ${c.id}\n\n${c.text}\n\n定位：${c.locator}。类型：${c.kind}。\n`).join("\n")}\n## 进入哪些综合结论\n\n${d.conclusions.filter(c => c.evidence.some(id => s.claims.some(v => v.id === id))).map(c => `- [${c.title}](../conclusions/${c.id}.md)`).join("\n")}\n`);
files.set("synthesis.md", header + `# ${d.title}\n\n${d.question}\n\n本篇结论是条件性的：表现约束能力判断，相关结构约束统计解释，干预结果约束因果作用；每向机制推进一步，都需要说明额外的假设和仍可成立的竞争解释。\n\n[交互页](index.html) · [完整比较矩阵](comparison.md) · [逐条结论审计](conclusions.md)\n\n![证据与主张之间的条件](inference-map.svg)\n\n${d.synthesis.map(s => `## ${s.title}\n\n${s.text}\n\n依据与边界：${s.conclusions.map(id => `[${id}](conclusions/${id}.md)`).join("、")}。`).join("\n\n")}\n\n## 群体研究的演进与版本\n\n![群体共同因素的争论顺序](collective-debate.svg)\n\n箭头只表示这里组织的阅读和争论次序；具体反驳与方法差异以文献笔记为准。\n\n## 回访第三轮\n\n${d.priorQuestionLinks.map(q => `- [${q.batch}／${q.id}](../${q.batch}/questions/${q.id}.md)：${q.addition}`).join("\n")}\n\n## 下一步怎样产生新证据\n\n优先执行[已知模型识别方案](research-designs.md#design-causal)和[群体研究版本复算](research-designs.md#design-version)，再决定是否实施需要招募的[协作条件实验](research-designs.md#design-group)。这些扩展尚未执行，当前完成的独立验证仅是[预先协议下的精确反例](method-check/README.md)。\n`);
files.set("research-designs.md", header + `# 后续研究方案与开放问题\n\n下面是能产生区分证据的设计；状态均为未执行。群体研究以群体为独立单位，不用成员人数虚增有效样本量。\n\n${d.proposals.map(p => `<a id="${p.id}"></a>\n\n## ${p.title}\n\n${p.question}\n\n- 竞争模型：${p.models}\n- 设计：${p.design}\n- 主要结果：${p.outcome}\n- 失败／停止解释规则：${p.falsification}\n- 实施前待定：${p.beforeRun}\n- 对应结论：${p.links.map(c => `[${c}](conclusions/${c}.md)`).join("、")}。状态：${p.status}。`).join("\n\n")}\n\n## 后续主题队列\n\n1. 系统边界与发展：比较耦合、必要条件与认知构成；时间尺度与介入位置先对齐。\n2. 意义与意识解释：补齐理论版本、反对论证及从功能到体验的桥接前提。\n3. 综合修订与网站联动：逐条验收后再形成正式路线页、文献解读与文章；第一阶段完成不等于全部第四轮完成。\n`);
files.set("index-audit.md", header + `# 跨批索引修订记录\n\n2026-10-28。原批次数据与生成笔记保持历史快照；[当前注释层](../index-annotations.json)由全局生成器应用。第一至三轮仍为150条节点记录／144个稳定ID，第四轮综合单独计数。\n\n## 覆盖口径\n\n首轮106个候选中，81个有同ID后续笔记；25个没有同ID匹配。本轮逐项审查后，其中《中论》候选有第24品的部分作品覆盖，另外24个尚无对应作品阅读映射。相关论文、前作或后作不替代指定书籍，也不据此宣称全书已读。\n\n| 候选ID | 映射性质 | 后续材料 | 范围判断 |\n| --- | --- | --- | --- |\n${annotations.candidateMappings.map(x => `| ${x.candidateId} | ${x.status} | ${x.note ? `[${x.coveredBy}](../${x.note})` : "无"} | ${md(x.scope)} |`).join("\n")}\n\n## 当前路线缺口修订\n\n${Object.entries(annotations.routeGapOverrides).map(([id,a]) => `- **${id}**：${a.reason}。`).join("\n")}\n\n## 书目更正\n\n${annotations.sourceCorrections.map(c => `- ${c.batch}／${c.sourceId}：${c.field}中的${c.old}改为${c.new}。${c.reason} [核对入口](${c.evidence})；[当前笔记](../${c.currentNote})。`).join("\n")}\n`);
const checks = Object.keys(results.checks);
files.set("method-check/README.md", header + `# 同观察与不同机制的精确核验\n\n[预先协议](protocol.md) · [精确结果](results.json) · [交互演示](../index.html#method) · [脚本](../../../../scripts/check-intelligence-round-four.py)\n\n性质：本站构造的数学反例。${checks.length}项检查通过；未采样心理学或神经科学数据，未复现互惠模型或原论文实验。\n\n## 三变量高斯模型\n\n共同原因模型F：Xi=(G+ei)/√2，所有G、ei独立标准正态。\n\n顺序模型C：X1=U1；X2=X1/2+U2；X3=X1/3+X2/3+U3。独立零均值正态噪声方差为1、3/4、2/3。\n\n两者均为零均值多元高斯，协方差完全相同（对角1，非对角1/2），因此这里相同的不只是相关矩阵，而是整个观察联合分布。行列式为${results.determinant}。脚本将F重参数化为方差1/2的共同因子与误差，以有理数精确运算。\n\n| 干预 | F的均值(X1,X2,X3) | C的均值(X1,X2,X3) |\n| --- | --- | --- |\n${results.interventions.map(i => `| ${i.target} | ${i.F.join(", ")} | ${i.C.join(", ")} |`).join("\n")}\n\n干预替换目标变量方程并删除其入边；条件化X1=1不是相同操作。两种模型的E[X2\u007cX1=1]均为1/2，但F的E[X2\u007cdo(X1=1)]为0。\n\n## 有限干预也不能穷尽机制\n\nD1为Y=A∧B，D2为Y=A。只在B=1的情境中观察并改变A，两个模型均给出相同结果；在A=1、B=0时输出分别为0、1。\n\n| A | B | D1 | D2 | 原已测情境 |\n| --- | --- | --- | --- | --- |\n${results.circuitTruthTable.map(x => `| ${x.A} | ${x.B} | ${x.D1} | ${x.D2} | ${x.testedPreviously ? "是" : "否"} |`).join("\n")}\n\n## 复核\n\n\`python3 scripts/check-intelligence-round-four.py --check\`\n\n检查：${checks.join("、")}。协议和脚本SHA256存于结果文件。可证明的只是上述推论边界，不能据此判断现实g或群体c的实际生成过程。\n`);
const svgStart = (w,h,title,marker) => `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title><style>text{font-family:system-ui,-apple-system,'PingFang SC',sans-serif;fill:#203d3a}.small{fill:#57675f}</style><defs><marker id="${marker}" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0L10 4L0 8" fill="none" stroke="#577c6c"/></marker></defs><rect width="100%" height="100%" fill="#faf9f4"/>`;
const textLines = (lines,x,y,size=16) => lines.map((line,i) => `<text x="${x}" y="${y+i*(size+10)}" font-size="${size}">${esc(line)}</text>`).join("");
const box = (x,y,w,h,lines) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="#eff3ec" stroke="#bdcbbf"/>${textLines(lines,x+20,y+33)}`;
let map = svgStart(1080,590,"从证据到结论需要哪些条件","inference-arrow");
map += textLines(["从证据到结论，需要哪些条件？"],32,42,26);
map += textLines(["这些是不同解释任务，纵向排布不表示证据等级。"],32,76,15);
const columns = [
	{title:["任务表现","成绩、时间、错误模式"],condition:["任务／评分／资源对齐","留出任务与对象验证"],result:["有适用域的能力判断","不能直接识别算法"],id:"p1"},
	{title:["共同变化","相关结构、潜在因子"],condition:["测量与竞争模型审计","检查是否观察等价"],result:["统计结构与预测约束","不能唯一识别共同原因"],id:"p2"},
	{title:["干预与损伤","改变结构后表现变化"],condition:["操纵特异性与过程读数","新情境与恢复检验"],result:["受检条件中的因果约束","仍需解释计算组织"],id:"p3"}
];
for (const [i,c] of columns.entries()) {const x=32+i*348;map+=`<a href="conclusions/${c.id}.md">${box(x,112,320,95,c.title)}${box(x,263,320,95,c.condition)}${box(x,414,320,95,c.result)}</a><path d="M${x+160} 207V256M${x+160} 358V407" stroke="#577c6c" stroke-width="2" marker-end="url(#inference-arrow)"/>`;}
map += textLines(["综合：保留条件、竞争解释与下一项可区分检验。意义／体验的桥接前提另行论证。"],32,556,16)+"</svg>\n";
files.set("inference-map.svg",map);
files.set("inference-map.mmd",`flowchart TB\n  O["任务表现"] --> T["任务、评分、资源与留出验证"] --> A["限定域能力判断"]\n  S["共同变化"] --> M["测量与竞争模型审计"] --> B["统计约束，不唯一识别共同原因"]\n  I["干预与损伤"] --> C["操纵特异性、过程与新情境"] --> D["因果约束，不自动给出算法"]\n  A --> E["条件、竞争解释、下一项检验"]\n  B --> E\n  D --> E\n`);
const debateItems = [
	["2010 · Woolley等","多任务共同因素与预测","轮次数据仅46组","原文＋补充材料", "woolley-2010"],
	["2017 · Bates与Gupta","个体能力解释的挑战","任务／计分／记录有差别","2016作者接受稿", "bates-gupta-2017"],
	["2021 · Riedl等","22样本，1356组","留出一种任务作预测","未等同新群体外部验证", "riedl-2021"],
	["2022 · 正式更正","AVE：44% → 19.6%","三指标模型零自由度","更正不自动裁决所有结论", "riedl-correction-2022"]
];
let debate = svgStart(1080,450,"群体共同因素的争论与版本","debate-arrow");
debate += textLines(["群体共同因素：什么延续，什么发生争论？"],32,42,25)+textLines(["按争论阅读顺序组织；横向距离不表示年数。"],32,76,15);
for (const [i,item] of debateItems.entries()) {const x=24+i*264;debate+=`<a href="notes/${item[4]}.md">${box(x,122,244,210,[item[0]])}${textLines(item.slice(1,4),x+16,194,14)}</a>`;if(i<3)debate+=`<path d="M${x+244} 224H${x+260}" stroke="#577c6c" marker-end="url(#debate-arrow)"/>`;}
debate += textLines(["共识范围：所测任务中的共同表现结构。争点：构念、预测来源、任务依赖与因果解释。","更正应覆盖旧报告值；平均提取方差与第一特征值所占总方差不能直接比较。"],32,382,16)+"</svg>\n";
files.set("collective-debate.svg",debate);
files.set("collective-debate.mmd",`flowchart LR\n W["2010 Woolley：共同因素与预测"] -->|"复制与解释争议"| B["2017 Bates–Gupta：成员能力解释"]\n B -->|"任务与方法比较"| R["2021 Riedl：多样本与过程预测"]\n R -->|"正式更正"| C["2022：AVE 19.6%；零自由度说明"]\n`);
files.set("index.html",renderRoundFour(d,results,map,debate));
const check=process.argv.includes("--check");
for (const [file,content] of files) {
	const p=path.join(root,file);
	if (check) {if (!fs.existsSync(p)||fs.readFileSync(p,"utf8")!==content) throw Error(`Stale ${file}`);}
	else {fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,content);}
}
console.log(JSON.stringify({mode:check?"verified":"generated",files:files.size,sources:d.sources.length,claims:claims.size,conclusions:d.conclusions.length,cases:d.cases.length}));
