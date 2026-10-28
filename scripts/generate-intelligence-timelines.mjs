import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import crypto from "node:crypto";
const root=path.resolve(import.meta.dirname,".."),history=path.join(root,"docs/intelligence-history");
const read=p=>JSON.parse(fs.readFileSync(path.join(history,p),"utf8"));
const index=read("research-index.json"),names=["round-two",..."abcde".split("").map(c=>`round-three-${c}`),"round-six"],batches=Object.fromEntries(names.map(n=>[n,read(`${n}/research.json`)]));
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&apos;"}[c]));
const md=s=>String(s??"").replace(/["\n\r]/g," ").replaceAll("<","&lt;").replaceAll(">","&gt;");
const asText=x=>Array.isArray(x)?x.join("；"):x??"";
const relationNames={"method-extension":"方法扩展","benchmark-reuse":"基准复用","direct-response":"直接回应","direct-critique":"直接批评","explicit-adoption":"明确采用","method-comparison":"方法比较","conceptual-comparison":"概念比较"};
const routes=index.routes.map(route=>{
 const prior=route.nodes;const additions=batches["round-six"].nodes.filter(n=>n.routeIds.includes(route.id)).map(n=>({id:n.id,batch:"round-six",batchLabel:"第六轮",note:`round-six/notes/${n.sourceIds[0]}.md`}));
 assert(additions.every(n=>!prior.some(p=>p.id===n.id)),`Duplicate node in ${route.id}`);
 const nodes=[...prior,...additions].map(ref=>{
  const b=batches[ref.batch];assert(b,ref.batch);const node=b.nodes.find(n=>n.id===ref.id);assert(node,`${ref.batch} ${ref.id}`);
  const sources=node.source?[node.source]:(node.sourceIds??[]).map(id=>{const source=b.sources.find(s=>s.id===id);assert(source,`${ref.batch} ${id}`);return source;});
  assert(sources.length,`No source for ${ref.id}`);assert(node.question && node.mechanism && node.limits && node.claims?.length,ref.id);
  return {id:node.id,title:node.title,year:node.year??null,dateLabel:node.dateLabel??String(node.year??"年代未定"),dateBasis:node.dateBasis??"原批次未单列",batch:ref.batch,batchLabel:ref.batchLabel,branch:node.branch??"原批次未单列",question:node.question,mechanism:asText(node.mechanism),assumptions:asText(node.assumptions),limits:asText(node.limits),nextQuestion:asText(node.nextQuestion),claims:node.claims.map(c=>({id:c.id,text:c.text,locator:c.locator,basis:c.basis,sourceId:c.sourceId??sources[0].id,independentlyReplicated:c.independentlyReplicated??false})),sources:sources.map(s=>{const corrections=index.annotations.sourceCorrections.filter(c=>c.batch===ref.batch&&c.sourceId===s.id);const recordedAuthors=s.authors??[];let authors=[...recordedAuthors];for(const c of corrections){assert.equal(c.field,"authors",`Unsupported source correction ${c.field}`);assert(authors.includes(c.old),`Correction target missing ${s.id}`);authors=authors.map(name=>name===c.old?c.new:name);}return {id:s.id,title:s.title??s.version,authors,recordedAuthors,corrections,year:s.year??null,url:s.url,version:s.version,readingStatus:s.readingStatus,readScope:s.readScope,unreadScope:s.unreadScope??"原批次未单列；详见原批次阅读范围与限制。",verification:s.verification??"按原批次记录理解，未自动视为独立复现。"};}),priorNote:ref.note};
 });
 const ids=new Set(nodes.map(n=>n.id)),relations=names.flatMap(name=>batches[name].relations.filter(r=>ids.has(r.fromId)&&ids.has(r.toId)).map(r=>({id:r.id,batch:name,fromId:r.fromId,toId:r.toId,type:r.type,typeLabel:relationNames[r.type]??r.type,status:r.status,locator:r.locator??"原记录未单列",explanation:r.explanation??"",sourceIds:r.evidenceSourceIds??(r.evidenceSourceId?[r.evidenceSourceId]:[])})));
 const gaps=[...route.gaps];
 return {id:route.id,name:route.name,question:route.question,gaps,nodes,relations};
});
assert.equal(routes.length,20);assert.equal(index.counts.distinctNodes,144);assert.equal(index.counts.nodeRecords,150);
for(const r of routes)for(const n of r.nodes)for(const c of n.claims)assert(n.sources.some(s=>s.id===c.sourceId),`Missing source ${r.id} ${n.id} ${c.sourceId}`);
const data={schemaVersion:1,asOf:index.asOf,scope:"第二轮至第三轮历史索引加第六轮增量；前者使用索引选定的批次，同ID不同批次阅读范围不自动合并。第六轮来源均为指定片段阅读。",historicalCounts:index.counts,roundSix:{nodes:batches["round-six"].nodes.length,selectedCandidates:batches["round-six"].selectedCandidateIds.length,priorUnmapped:24,remainingUnmapped:20},routes,inputs:Object.fromEntries(["research-index.json",...names.map(n=>`${n}/research.json`)].map(p=>[p,crypto.createHash("sha256").update(fs.readFileSync(path.join(history,p))).digest("hex")]))};
const files=new Map(),out=(p,s)=>files.set(path.join(root,p),s.endsWith("\n")?s:s+"\n");
out("src/data/intelligenceTimeline.generated.json",JSON.stringify(data,null,2));
for(const route of routes){
 const base=`/research/intelligence/timelines/${route.id}/`, dated=route.nodes.filter(n=>Number.isFinite(n.year)).sort((a,b)=>a.year-b.year||a.id.localeCompare(b.id)),ancient=route.nodes.filter(n=>!Number.isFinite(n.year));
 const min=Math.floor(Math.min(...dated.map(n=>n.year))/50)*50,max=Math.ceil(Math.max(...dated.map(n=>n.year))/10)*10, x=y=>Math.round(340+(y-min)*770/(max-min)),height=155+dated.length*55;
 const ticks=[];for(let t=Math.ceil(min/50)*50;t<=max;t+=50)ticks.push(t);if(!ticks.length)ticks.push(min,max);
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 ${height}" role="img" aria-labelledby="title desc"><title id="title">${esc(route.name)}：已审校节点时间线</title><desc id="desc">横轴为原记录中的年份，点按节点打开问题和证据；纵向只用于排版，不表示历史影响。年代未定文本列在页面独立区域。</desc><rect width="1200" height="${height}" fill="#f3f6f1"/><g font-family="system-ui,sans-serif" fill="#263f37"><text x="30" y="42" font-size="24">${esc(route.name)}</text><text x="30" y="75" font-size="15">横轴：记录年份；行距为排版，不表示影响或强弱</text>${ticks.map(t=>`<line x1="${x(t)}" x2="${x(t)}" y1="105" y2="${height-24}" stroke="#d6dfd5"/><text x="${x(t)}" y="99" text-anchor="middle" font-size="15">${t}</text>`).join("")}${dated.map((n,i)=>`<a href="${base}#${n.id}"><title>${esc(n.dateLabel)} · ${esc(n.title)}；问题：${esc(n.question)}；方法：${esc(n.mechanism)}；限制：${esc(n.limits)}</title><text x="30" y="${140+i*55}" font-size="16">${i+1}. ${esc(Array.from(n.title).slice(0,16).join(""))}</text><circle cx="${x(n.year)}" cy="${134+i*55}" r="8" fill="#557d64"/><circle cx="${x(n.year)}" cy="${134+i*55}" r="22" fill="transparent" pointer-events="all"/><text x="${Math.min(x(n.year)+14,1130)}" y="${140+i*55}" font-size="14">${n.year}</text></a>`).join("")}</g></svg>`;
 out(`public/research/intelligence/timelines/${route.id}/timeline.svg`,svg);
 let mermaid=`flowchart TB\n  %% 仅展示年代顺序；没有线不代表没有关系，点按进入原节点；古典年代未定单列。\n${[...dated,...ancient].map((n,i)=>`  n${i}["${md(n.dateLabel)} · ${md(n.title)}<br/>问题：${md(n.question)}<br/>方法：${md(n.mechanism)}<br/>限制：${md(n.limits)}"]\n  click n${i} "${base}#${n.id}"`).join("\n")}\n`;
 out(`public/research/intelligence/timelines/${route.id}/timeline.mmd`,mermaid);
 out(`public/research/intelligence/timelines/${route.id}/timeline.json`,JSON.stringify({scope:data.scope,route},null,2));
}
const check=process.argv.includes("--check");for(const [p,s] of files){if(check)assert.equal(fs.readFileSync(p,"utf8"),s,`Stale ${path.relative(root,p)}`);else{fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,s);}}
console.log(JSON.stringify({mode:check?"verified":"generated",files:files.size,routes:routes.length,routeNodeOccurrences:routes.reduce((s,r)=>s+r.nodes.length,0),distinctNodes:new Set(routes.flatMap(r=>r.nodes.map(n=>n.id))).size,ancientRouteOccurrences:routes.flatMap(r=>r.nodes).filter(n=>n.year===null).length,relationOccurrences:routes.reduce((s,r)=>s+r.relations.length,0)}));
