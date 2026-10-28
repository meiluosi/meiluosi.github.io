import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import assert from "node:assert/strict";
import {createRequire} from "node:module";
const require=createRequire(process.env.PLAYWRIGHT_ANCHOR??path.resolve("package.json"));const {chromium}=require("playwright");
const root=path.resolve(import.meta.dirname,".."),phase=path.join(root,"docs/intelligence-history/round-four/integration"),out=path.join(phase,"evidence");
const read=p=>JSON.parse(fs.readFileSync(p));const review=read(path.join(phase,"research.json"));
const semantics=read(path.join(phase,"../meaning/method-check/results.json")),history=read(path.join(phase,"../boundaries/method-check/results.json"));
const base=process.env.INTELLIGENCE_SITE_URL??"http://127.0.0.1:4327",home=base+"/research/intelligence/";
const browser=await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:"reduce"});const page=await context.newPage(),errors=[];page.on("pageerror",error=>errors.push(error.message));
const hydrate=async ()=>{await page.locator('[data-reviewed-lens]').scrollIntoViewIfNeeded();await page.waitForFunction(()=>!document.querySelector('[data-reviewed-lens]')?.closest('astro-island')?.hasAttribute('ssr'));};
const goto=async url=>{await page.goto(url);await page.evaluate(()=>document.fonts.ready);};
const noOverflow=async label=>{const size=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));assert(size.scrollWidth<=size.width,`${label}: ${JSON.stringify(size)}`);return {label,...size};};
const numeric=s=>s.includes('/')?s.split('/').map(Number).reduce((a,b)=>a/b):Number(s);
const viewports=[];
try{
 await goto(home);assert.equal(await page.locator('.research-grid > article').count(),3);await page.screenshot({path:out+'/site-desktop.png'});viewports.push(await noOverflow('overview-desktop'));
 let conclusions=0;
 for(const article of review.articles){await goto(home+article.id+'/');assert.equal(await page.locator('.conclusion').count(),6);for(const r of review.reviews.filter(r=>r.article===article.id)){const summary=page.locator('#'+r.id+' summary');await summary.focus();await page.keyboard.press('Enter');assert.equal(await page.locator('#'+r.id+' details').getAttribute('open'),'');await page.keyboard.press('Space');assert.equal(await page.locator('#'+r.id+' details').getAttribute('open'),null);conclusions++;}viewports.push(await noOverflow(article.id+'-desktop'));}
 await goto(home+'meaning/#m5');await page.locator('#m5 summary').click();await page.locator('#m5').screenshot({path:out+'/conclusion-desktop.png'});
 await page.locator('#m5 .citations a').first().click();await page.waitForURL('**/sources/**');assert(await page.locator(':target').count()===1);assert(await page.locator('.scope-note').first().isVisible());
 await goto(home+'sources/?article=meaning');assert.equal(await page.locator('.source-row:visible').count(),12);await page.fill('synthesis-library input','概念角色');assert(await page.locator('.source-row:visible').count()>=1);await page.fill('synthesis-library input','no-match-382499');assert.equal(await page.locator('.source-row:visible').count(),0);assert(await page.locator('[data-empty]').isVisible());await page.fill('synthesis-library input','');await page.selectOption('synthesis-library select','');assert.equal(await page.locator('.source-row:visible').count(),31);
 await page.selectOption('synthesis-library select','evidence');assert.equal(await page.locator('.source-row:visible').count(),10);await page.reload();assert.equal(await page.locator('.source-row:visible').count(),10);
 await goto(home+'sources/?article=boundaries');assert.equal(await page.locator('.source-row:visible').count(),9);await page.screenshot({path:out+'/sources-desktop.png'});
 // Route linkage and single-island hydration, verified against saved protocol results.
 await goto(base+'/routes/collective-extended/');await hydrate();
 for(const row of history.conditions){await page.selectOption('[data-history]',row.history);await page.locator('[data-support]').setChecked(Boolean(row.support));await page.locator('[data-reset]').setChecked(row.reset);assert.equal(Number(await page.locator('[data-state]').textContent()),numeric(row.state));assert.equal(Number(await page.locator('[data-readout]').textContent()),numeric(row.output));}
 await goto(base+'/routes/philosophy-of-mind/');await hydrate();await page.locator('[data-anchors]').selectOption('one');await page.waitForFunction(()=>document.querySelector('[data-candidates]')?.textContent==='2');
 for(const [key,value] of Object.entries(semantics.semantic)){await page.selectOption('[data-anchors]',key);assert.equal(Number(await page.locator('[data-candidates]').textContent()),value.candidates.length);assert.equal(await page.locator('.mappings li').count(),value.candidates.length);}
 await goto(base+'/routes/consciousness-self/');await hydrate();await page.locator('[data-report-required]').uncheck();await page.waitForFunction(()=>document.querySelector('[data-signal-b]')?.textContent==='0');
 for(const row of semantics.signals){await page.locator('[data-stimulus]').setChecked(Boolean(row.S));await page.locator('[data-report-required]').setChecked(Boolean(row.Q));for(const [id,key] of [['signal-a','A'],['signal-b','B'],['report','R']])assert.equal(Number(await page.locator(`[data-${id}]`).textContent()),row[key]);}
 await page.locator('[data-stimulus]').check();await page.locator('[data-report-required]').uncheck();await page.locator('[data-reviewed-lens]').screenshot({path:out+'/lens-desktop.png'});
 await page.evaluate(()=>document.documentElement.lang='en');await page.waitForFunction(()=>document.querySelector('[data-reviewed-lens] h3')?.textContent.includes('candidate signals'));assert((await page.locator('[data-reviewed-lens] h3').textContent()).includes('candidate signals'));await page.evaluate(()=>document.documentElement.lang='zh-CN');
 // The site uses its configured sage/light palette. Detailed text remains Chinese.
 await goto(home);await page.screenshot({path:out+'/site-theme.png'});
 await page.setViewportSize({width:390,height:844});
 for(const [label,url] of [['overview',home],['meaning',home+'meaning/'],['source',home+'sources/prereg-v4-2022/'],['sources',home+'sources/?article=meaning'],['review',home+'review/'],['queue',home+'next/']]){await goto(url);viewports.push(await noOverflow(label+'-mobile'));if(['overview','meaning','source'].includes(label))await page.screenshot({path:out+'/'+label+'-mobile.png'});}
 for(const [route,mode] of [['collective-extended','history'],['philosophy-of-mind','semantics'],['consciousness-self','report']]){await goto(base+'/routes/'+route+'/');await page.locator('[data-reviewed-lens]').scrollIntoViewIfNeeded();viewports.push(await noOverflow(mode+'-mobile'));await page.locator('[data-reviewed-lens]').screenshot({path:out+'/lens-'+mode+'-mobile.png'});}
 // Client-side navigation must reconnect source filtering after a round trip.
 await goto(home);await page.locator('a[href="/research/intelligence/sources/"]').first().click();await page.waitForURL('**/sources/');await page.selectOption('synthesis-library select','meaning');assert.equal(await page.locator('.source-row:visible').count(),12);
 await page.locator('a[href="/research/intelligence/"]').last().click();await page.waitForURL(home);await page.locator('a[href="/research/intelligence/sources/"]').first().click();await page.waitForURL('**/sources/');await page.selectOption('synthesis-library select','boundaries');assert.equal(await page.locator('.source-row:visible').count(),9);
 const noJs=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}}),staticPage=await noJs.newPage();await staticPage.goto(home+'meaning/');assert.equal(await staticPage.locator('.conclusion').count(),6);assert(await staticPage.locator('#m5').isVisible());await staticPage.screenshot({path:out+'/no-script-mobile.png'});await staticPage.goto(home+'sources/');assert.equal(await staticPage.locator('.source-row').count(),31);await noJs.close();
 assert.deepEqual(errors,[]);
 const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
 const sourceFiles=['src/pages/routes/[id].astro','src/pages/routes.astro','src/pages/library.astro','src/pages/notes.astro','src/layouts/ExploreLayout.astro','src/components/research/ResearchEntry.astro','src/components/features/route-lenses/Lens15ExtendedMind.svelte','src/components/features/route-lenses/Lens17PhilosophyOfMind.svelte','src/components/features/route-lenses/Lens18Consciousness.svelte','src/data/intelligenceResearch.ts','src/components/research/ReviewedInferenceLens.svelte','src/styles/intelligence-research.css',...fs.readdirSync(path.join(root,'src/pages/research/intelligence'),{recursive:true}).filter(f=>f.endsWith('.astro')).map(f=>'src/pages/research/intelligence/'+f)];
 const report={date:'2026-10-28',base,browser:'Microsoft Edge via Playwright',reviewSHA256:sha(path.join(phase,'research.json')),checkerSHA256:sha(import.meta.filename),sourceSHA256:Object.fromEntries(sourceFiles.map(f=>[f,sha(path.join(root,f))])),conclusionsKeyboard:conclusions,sourceFilters:[31,10,9,12],searchEmptyAndReset:true,queryReload:true,navigationReconnect:true,sourceDeepLink:true,historyConditions:history.conditions.length,semanticConditions:4,reportConditions:4,lensLanguageSwitch:true,noJavaScriptReading:true,themeMode:"existing sage/light palette; dark mode not tested",viewports,scriptErrors:errors};
 fs.writeFileSync(out+'/browser-check.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,conclusions,historyConditions:history.conditions.length,viewports:viewports.length}));
}finally{await browser.close();}
