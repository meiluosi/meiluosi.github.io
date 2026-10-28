// Run against the local research preview. Set PLAYWRIGHT_ANCHOR and BROWSER_EXECUTABLE if needed.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const require=createRequire(process.env.PLAYWRIGHT_ANCHOR??path.resolve('package.json'));
const {chromium}=require('playwright');
const root=path.resolve(import.meta.dirname,'../docs/intelligence-history/round-four/boundaries');
const out=path.join(root,'evidence'), r=JSON.parse(fs.readFileSync(path.join(root,'method-check/results.json')));
const base=process.env.RESEARCH_PREVIEW_URL??'http://127.0.0.1:8768/';
const url=base+'round-four/boundaries/index.html';
const browser=await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1,reducedMotion:'reduce'});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 await page.goto(url);await page.evaluate(()=>document.fonts.ready);
 const svg=[];
 for(const key of ['boundary','timeline','time']){
  const b=page.locator('[data-map='+key+']');await b.focus();await page.keyboard.press('Enter');
  assert.equal(await page.locator('[data-map][aria-pressed=true]').count(),1);
  assert.equal(await page.locator('.figure:visible').count(),1);
  const bounds=await page.locator('#map-'+key+' svg').evaluate(s=>{const v=s.viewBox.baseVal;return {textCount:s.querySelectorAll('text').length,outside:[...s.querySelectorAll('text')].filter(t=>{const b=t.getBBox();return b.x<v.x||b.y<v.y||b.x+b.width>v.width||b.y+b.height>v.height;}).map(t=>t.textContent)};});
  assert.deepEqual(bounds.outside,[]);svg.push({key,...bounds});
  await page.locator('#map-'+key).screenshot({path:out+'/'+key+'-desktop.png'});
 }
 await page.locator('[data-map=boundary]').click();
 for(let i=1;i<=6;i++){const s=page.locator('#b'+i+' summary');await s.focus();await page.keyboard.press('Enter');assert.equal(await page.locator('#b'+i).getAttribute('open'),'');await page.keyboard.press('Space');assert.equal(await page.locator('#b'+i).getAttribute('open'),null);}
 for(const c of ['cockpit','notebook','development','autonomy','interaction']){await page.selectOption('#case-filter',c);assert.equal(await page.locator('#cases article:visible').count(),1);assert.equal(await page.locator('#case-count').textContent(),'1 / 5 个案例');}
 await page.selectOption('#case-filter','');assert.equal(await page.locator('#cases article:visible').count(),5);
 for(const v of r.conditions){await page.selectOption('#history',v.history);await page.selectOption('#support',String(v.support));await page.locator('#reset').setChecked(v.reset);assert.equal(await page.locator('#state-output').textContent(),v.state);assert.equal(await page.locator('#readout-output').textContent(),v.output);if(v.reset)assert((await page.locator('#history-status').textContent()).includes('当前状态中介'));}
 for(const k of ['baseline','perturbed']){
  await page.locator('[data-feedback='+k+']').focus();await page.keyboard.press('Enter');
  assert.equal(await page.locator('[data-feedback][aria-pressed=true]').count(),1);
  const a=r.feedback[k==='baseline'?'loopBaseline':'loopPerturbed'],b=r.feedback[k==='baseline'?'replayBaseline':'replayPerturbed'];
  const actual=await page.locator('#feedback-table tr').evaluateAll(rows=>rows.map(row=>[...row.cells].map(c=>c.textContent)));
  assert.deepEqual(actual,a.map((v,i)=>[String(i),v.x,b[i].x,v.e,b[i].e]));
  assert.equal(await page.locator('.trace polyline').count(),4);
 }
 assert((await page.locator('#feedback-status').textContent()).includes('到t=4才分开'));
 await page.selectOption('#history','0011');await page.locator('#reset').uncheck();
 await page.locator('#method').screenshot({path:out+'/method-desktop.png'});
 await page.fill('#source-search','适应性');assert((await page.locator('.reading tbody tr:visible').count())>=1);
 await page.fill('#source-search','unmatched-term-482938');assert.equal(await page.locator('.reading tbody tr:visible').count(),0);assert(await page.locator('#source-empty').isVisible());
 await page.fill('#source-search','');assert.equal(await page.locator('.reading tbody tr:visible').count(),9);
 await page.goto(url+'#b5');assert.equal(await page.locator('#b5').getAttribute('open'),'');await page.locator('#b5').screenshot({path:out+'/conclusion-desktop.png'});
 await page.goto(url);await page.screenshot({path:out+'/desktop.png'});
 const duplicates=await page.evaluate(()=>{const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return ids.filter((id,i)=>ids.indexOf(id)!==i);});assert.deepEqual(duplicates,[]);
 const links=await page.locator('a[href]').evaluateAll(as=>[...new Set(as.map(a=>new URL(a.getAttribute('href'),location.href).href))]);
 const broken=[];for(const u of links){if(u.startsWith(base)){const response=await page.request.get(u);if(!response.ok())broken.push({url:u,status:response.status()});}}assert.deepEqual(broken,[]);
 await page.setViewportSize({width:390,height:844});await page.goto(url);
 const mobile=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));assert(mobile.scrollWidth<=mobile.width);
 await page.screenshot({path:out+'/mobile.png'});
 await page.selectOption('#history','1100');await page.selectOption('#support','0');assert.equal(await page.locator('#readout-output').textContent(),'3/16');await page.locator('#reset').check();assert.equal(await page.locator('#readout-output').textContent(),'0');
 await page.locator('[data-feedback=perturbed]').click();assert((await page.locator('#feedback-status').textContent()).includes('t=3'));
 await page.locator('#method .lab').first().screenshot({path:out+'/method-mobile.png'});
 await page.goto(base+'round-four/index.html');assert.equal(await page.locator('a[href="boundaries/index.html"]').count(),2);
 await page.locator('[data-test=x1]').click();assert.equal(await page.locator('#model-c').textContent(),'均值 (1, 1/2, 1/2)');
 await page.goto(base+'index.html');assert.equal(await page.locator('#routes > *').count(),20);assert.equal(await page.locator('#questions > *').count(),15);assert(await page.locator('a[href="round-four/boundaries/index.html"]').count()>=1);
 await page.selectOption('#batch','round-three-e');assert.equal(await page.locator('#routes > *').count(),4);await page.selectOption('#batch','');await page.fill('#search','退相干');assert((await page.locator('#nodes').textContent()).includes('退相干'));await page.fill('#search','');
 const globalMobile=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));assert(globalMobile.scrollWidth<=globalMobile.width);
 await page.screenshot({path:out+'/global-mobile.png'});
 assert.deepEqual(errors,[]);
 const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
 fs.writeFileSync(out+'/browser-check.json',JSON.stringify({date:'2026-10-28',browser:'Chromium / Microsoft Edge via Playwright, headless',viewports:['1440×1000','390×844'],researchSHA256:sha(path.join(root,'research.json')),htmlSHA256:sha(path.join(root,'index.html')),checkerSHA256:sha(import.meta.filename),graphsKeyboardToggle:3,svgTextBounds:svg,conclusionsKeyboardToggle:6,caseFilters:5,historyConditions:r.conditions.length,feedbackConditions:2,feedbackExactRows:14,sourceSearch:'matching, no results and reset',deepLink:'b5 opened',duplicateIDs:duplicates,directHrefCount:links.length,brokenLocalLinks:broken,mobile,globalMobile,phaseOneNavigation:true,phaseOneInterventionSmoke:true,globalRoutes:20,globalQuestionCards:15,globalFilterAndSearch:true,scriptErrors:errors},null,2)+'\n');
 console.log('Browser interactions, exact values, links and screenshots verified.');
}finally{await browser.close();}
