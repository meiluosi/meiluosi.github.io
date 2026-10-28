// Browser verification of the research workbench and saved exact enumerations.
import {createRequire} from "node:module";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import assert from "node:assert/strict";
const require=createRequire(process.env.PLAYWRIGHT_ANCHOR??path.resolve("package.json"));
const {chromium}=require("playwright");
const root=path.resolve(import.meta.dirname,"../docs/intelligence-history/round-four/meaning");
const out=path.join(root,"evidence"),r=JSON.parse(fs.readFileSync(path.join(root,"method-check/results.json")));
const base=process.env.RESEARCH_PREVIEW_URL??"http://127.0.0.1:8768/",url=base+"round-four/meaning/index.html";
const browser=await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1,reducedMotion:"reduce"});
const errors=[];page.on("pageerror",e=>errors.push(e.message));
try{
 await page.goto(url);await page.evaluate(()=>document.fonts.ready);
 const graphs=[];
 for(const key of ["bridge","timeline","prediction"]){
  await page.locator(`[data-map=${key}]`).focus();await page.keyboard.press("Enter");
  assert.equal(await page.locator("[data-map][aria-pressed=true]").count(),1);
  assert.equal(await page.locator(".figure:visible").count(),1);
  const bounds=await page.locator(`#map-${key} svg`).evaluate(s=>{const v=s.viewBox.baseVal;return{textCount:s.querySelectorAll("text").length,outside:[...s.querySelectorAll("text")].filter(t=>{const b=t.getBBox();return b.x<v.x||b.y<v.y||b.x+b.width>v.width||b.y+b.height>v.height;}).map(t=>t.textContent)};});
  assert.deepEqual(bounds.outside,[]);graphs.push({key,...bounds});
  await page.locator(`#map-${key}`).screenshot({path:out+`/${key}-desktop.png`});
 }
 await page.locator("[data-map=bridge]").click();
 for(let i=1;i<=6;i++){const el=page.locator(`#m${i} summary`);await el.focus();await page.keyboard.press("Enter");assert.equal(await page.locator(`#m${i}`).getAttribute("open"),"");await page.keyboard.press("Space");assert.equal(await page.locator(`#m${i}`).getAttribute("open"),null);}
 for(const c of ["reference","concept-role","access","identity","adversarial"]){await page.selectOption("#case-filter",c);assert.equal(await page.locator("#cases article:visible").count(),1);}
 await page.selectOption("#case-filter","");assert.equal(await page.locator("#cases article:visible").count(),5);
 for(const t of ["IIT","GNWT",""]){await page.selectOption("#theory-filter",t);assert.equal(await page.locator("#prediction-cards article:visible").count(),t?3:6);}
 for(const [key,v] of Object.entries(r.semantic)){
  await page.selectOption("#anchor-condition",key);assert.equal(await page.locator("#mapping-count").textContent(),String(v.candidates.length));
  assert.deepEqual(await page.locator("#mapping-list .mapping").allTextContents(),v.candidates.map(m=>Object.entries(m).map(([s,o])=>s+" → "+o).join("；")));
 }
 assert((await page.locator("#mapping-status").textContent()).includes("矛盾"));
 for(const x of r.signals){await page.selectOption("#stimulus",String(x.S));await page.selectOption("#report-required",String(x.Q));for(const [id,k] of [["signal-a","A"],["signal-b","B"],["report-output","R"]])assert.equal(await page.locator("#"+id).textContent(),String(x[k]));}
 let premiseConditions=0;
 for(const o of ["all","0","1"])for(const a of [false,true])for(const m of [false,true]){
  await page.selectOption("#observed",o);await page.locator("#auxiliary").setChecked(a);await page.locator("#measurement").setChecked(m);
  const expected=r.premises.filter(x=>(o==="all"||x.O===Number(o))&&(!a||x.A===1)&&(!m||x.M===1));
  assert.equal(await page.locator("#premise-count").textContent(),String(expected.length));
  const actual=await page.locator("#premise-table tr").evaluateAll(rows=>rows.map(row=>[...row.cells].map(c=>c.textContent)));
  assert.deepEqual(actual,expected.map(x=>[x.T,x.A,x.M,x.O].map(String)));premiseConditions++;
 }
 await page.selectOption("#anchor-condition","one");await page.locator("#lab-reference").screenshot({path:out+"/reference-desktop.png"});
 await page.selectOption("#stimulus","1");await page.selectOption("#report-required","0");await page.locator("#lab-report").screenshot({path:out+"/report-desktop.png"});
 await page.selectOption("#observed","0");await page.locator("#lab-premises").screenshot({path:out+"/premises-desktop.png"});
 await page.fill("#source-search","概念角色");assert(await page.locator(".reading tbody tr:visible").count()>=1);
 await page.fill("#source-search","unmatched-meaning-589431");assert.equal(await page.locator(".reading tbody tr:visible").count(),0);assert(await page.locator("#source-empty").isVisible());
 await page.fill("#source-search","");assert.equal(await page.locator(".reading tbody tr:visible").count(),12);
 await page.goto(url+"#m5");assert.equal(await page.locator("#m5").getAttribute("open"),"");await page.locator("#m5").screenshot({path:out+"/conclusion-desktop.png"});
 await page.goto(url);await page.screenshot({path:out+"/desktop.png"});
 const duplicates=await page.evaluate(()=>{const ids=[...document.querySelectorAll("[id]")].map(e=>e.id);return ids.filter((id,i)=>ids.indexOf(id)!==i);});assert.deepEqual(duplicates,[]);
 const links=await page.locator("a[href]").evaluateAll(as=>[...new Set(as.map(a=>new URL(a.getAttribute("href"),location.href).href))]);
 const broken=[];for(const u of links){if(u.startsWith(base)){const response=await page.request.get(u);if(!response.ok())broken.push({url:u,status:response.status()});}}assert.deepEqual(broken,[]);
 await page.setViewportSize({width:390,height:844});await page.goto(url);const mobile=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));assert(mobile.scrollWidth<=mobile.width);await page.screenshot({path:out+"/mobile.png"});
 await page.selectOption("#anchor-condition","two");assert.equal(await page.locator("#mapping-count").textContent(),"1");await page.locator("#lab-reference").screenshot({path:out+"/method-mobile.png"});
 await page.selectOption("#report-required","0");assert.equal(await page.locator("#signal-a").textContent(),"1");assert.equal(await page.locator("#signal-b").textContent(),"0");
 await page.locator("#auxiliary").check();await page.locator("#measurement").check();assert.equal(await page.locator("#premise-count").textContent(),"1");
 await page.goto(base+"round-four/index.html");assert(await page.locator('a[href="meaning/index.html"]').count()>=1);await page.locator("[data-test=x1]").click();assert.equal(await page.locator("#model-c").textContent(),"均值 (1, 1/2, 1/2)");
 await page.goto(base+"round-four/boundaries/index.html");assert(await page.locator('a[href="../meaning/index.html"]').count()>=1);await page.selectOption("#history","1100");await page.selectOption("#support","0");assert.equal(await page.locator("#readout-output").textContent(),"3/16");
 await page.goto(base+"index.html");assert.equal(await page.locator("#routes > *").count(),20);assert.equal(await page.locator("#questions > *").count(),15);assert(await page.locator('a[href="round-four/meaning/index.html"]').count()>=2);
 await page.selectOption("#batch","round-three-e");assert.equal(await page.locator("#routes > *").count(),4);await page.selectOption("#batch","");await page.fill("#search","退相干");assert((await page.locator("#nodes").textContent()).includes("退相干"));await page.fill("#search","");
 const globalMobile=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));assert(globalMobile.scrollWidth<=globalMobile.width);await page.screenshot({path:out+"/global-mobile.png"});assert.deepEqual(errors,[]);
 const sha=p=>crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
 fs.writeFileSync(out+"/browser-check.json",JSON.stringify({date:"2026-10-28",browser:"Microsoft Edge via Playwright, headless",viewports:["1440×1000","390×844"],researchSHA256:sha(path.join(root,"research.json")),htmlSHA256:sha(path.join(root,"index.html")),checkerSHA256:sha(import.meta.filename),graphsKeyboardToggle:3,svgTextBounds:graphs,conclusionsKeyboardToggle:6,caseFilters:5,theoryFilters:3,semanticConditions:4,signalConditions:4,premiseConditions,sourceSearch:"matching, empty and reset",deepLink:"m5 opened",duplicateIDs:duplicates,directHrefCount:links.length,brokenLocalLinks:broken,mobile,globalMobile,priorPhaseNavigationAndSmoke:true,globalRoutes:20,globalQuestions:15,globalSearchAndFilter:true,scriptErrors:errors},null,2)+"\n");
 console.log("Meaning workbench: exact enumerations, keyboard, mobile, navigation and links verified.");
}finally{await browser.close();}
