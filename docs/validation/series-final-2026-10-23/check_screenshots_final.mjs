import {chromium} from '/Users/meros/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
const dir='docs/validation/series-final-2026-10-23';const inventory=JSON.parse(await fs.readFile(`${dir}/inventory.json`,'utf8'));const out=[];
const browser=await chromium.launch({executablePath:'/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',headless:true});
const page=await browser.newPage();let errors=[];page.on('pageerror',e=>errors.push(e.message));await fs.mkdir('docs/screenshots/series-final-2026-10-23',{recursive:true});
try{for(const item of inventory.filter(x=>x.path.includes("eval-06-conclusions")))for(const lang of ['zh','en'])for(const width of [1440,320]){
const file=lang==='zh'?item.path:item.english[0];const source=await fs.readFile(file,'utf8');const slug=lang==='zh'?file.split('/').pop().replace('.md',''):source.match(/^slug: (.+)$/m)[1];const url=(lang==='zh'?'/posts/':'/en/posts/')+slug+'/';
await page.setViewportSize({width,height:1000});errors=[];let result={file,width,lang,url};
try{await page.goto('http://127.0.0.1:4331'+url+'?locale='+(lang==='zh'?'zh-CN':'en'),{waitUntil:'networkidle'});await page.evaluate(()=>document.fonts.ready);
Object.assign(result,await page.evaluate(()=>({title:document.title,overflow:document.documentElement.scrollWidth>innerWidth,katexErrors:document.querySelectorAll('.katex-error').length,wide:[...document.querySelectorAll('.katex-display')].map((e,i)=>({i,wide:e.scrollWidth>e.clientWidth+1,tex:e.querySelector('annotation')?.textContent})).filter(x=>x.wide),nodes:document.querySelectorAll('#article-concept-network svg a').length,scrubbers:document.querySelectorAll('.math-scrubber-bar').length})));
const selector=page.locator('#article-concept-network select').first();await selector.selectOption({index:1});result.branchNodes=await page.locator('#article-concept-network svg a').count();
result.errors=[...errors];result.pass=!result.overflow&&!result.katexErrors&&!result.wide.length&&result.nodes>5&&result.branchNodes>1&&!result.scrubbers&&!errors.length&&!result.title.includes('404');
await page.evaluate(()=>window.scrollTo(0,0)); await page.waitForTimeout(1500); if(true){await page.screenshot({path:`docs/screenshots/series-final-2026-10-23/${slug}-${lang}-${width}.png`});}
}catch(e){result.pass=false;result.failure=String(e)}
out.push(result);await fs.writeFile(`${dir}/browser-final-screenshots.json`,JSON.stringify(out,null,2));console.log(out.length,result.pass?'PASS':'FAIL',slug,lang,width,result.wide?.map(x=>x.i));
}}finally{await browser.close()}
console.log('TOTAL',out.length,'FAILED',out.filter(x=>!x.pass).length);
