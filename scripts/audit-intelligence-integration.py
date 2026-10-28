"""Validate the review overlay and the generated native-site evidence links."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib,json,re,sys,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1];phase=root/'docs/intelligence-history/round-four/integration';dist=root/'dist';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=read(phase/'research.json');errors=[]
for p,h in read(phase/'baselines.json').items():
 if sha(root/p)!=h:errors.append('Changed evidence: '+p)
class Markup(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src'] and v:self.links.append(v)
   if k=='id' and v:self.ids.append(v)
def parse(p):
 x=Markup();x.feed(p.read_text());return x
htmls=[p for p in (dist/'research/intelligence').rglob('*.html') if not set(p.relative_to(dist/'research/intelligence').parts).intersection({'timelines','revisit','verification','map'})];assert len(htmls)==38,len(htmls)
files=htmls+[p for p in (dist/'research/intelligence').rglob('*.svg') if not set(p.relative_to(dist/'research/intelligence').parts).intersection({'timelines','revisit','verification','map'})]+list((dist/'research/intelligence/downloads').rglob('*.md'))
paths=anchors=svg=0
for f in files:
 m=parse(f)
 if len(m.ids)!=len(set(m.ids)):errors.append('Duplicate IDs: '+str(f.relative_to(root)))
 if f.suffix=='.svg':ET.parse(f);svg+=1
 urls=m.links+(re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)',f.read_text()) if f.suffix=='.md' else [])
 for u in urls:
  parsed=urlsplit(u)
  if parsed.scheme or parsed.netloc:continue
  if parsed.path.startswith('/'):target=dist/unquote(parsed.path.lstrip('/'))
  else:target=(f.parent/unquote(parsed.path)).resolve() if parsed.path else f
  if target.is_dir():target=target/'index.html'
  paths+=1
  if not target.is_file():errors.append(f'{f.relative_to(root)} -> missing {u}');continue
  if parsed.fragment and target.suffix in ['.html','.svg']:
   anchors+=1
   if unquote(parsed.fragment) not in parse(target).ids:errors.append(f'{f.relative_to(root)} -> missing anchor {u}')
# Exact native content must match the reviewed layer and original locator text.
for r in d['reviews']:
 page=dist/f'research/intelligence/{r["article"]}/index.html'
 if r['id'] not in parse(page).ids or r['revised'] not in page.read_text():errors.append('Missing reviewed text '+r['id'])
for a in d['articles']:
 original=read(phase.parent/a['folder']/'research.json')
 for s in original['sources']:
  page=dist/f'research/intelligence/sources/{s["id"]}/index.html'
  for c in s['claims']:
   if c['id'] not in parse(page).ids:errors.append('Missing source location '+c['id'])
for route in read(root/'docs/intelligence-history/atlas.json')['routes']:
 page=dist/f'routes/{route["id"]}/index.html'
 if '/research/intelligence/' not in page.read_text():errors.append('Missing route entry '+route['id'])
for name in ['routes','library','notes']:
 assert '/research/intelligence/' in (dist/name/'index.html').read_text()
assert all(q['status']=='design-only' for q in d['queue'])
index=read(root/'docs/intelligence-history/research-index.json');assert index['roundFour']['currentPhase']==4 and index['counts']['distinctNodes']==144
report=dict(date='2026-10-28',nativePages=len(htmls),sources=31,sourceClaims=88,reviewedConclusions=18,protectedFiles=len(read(phase/'baselines.json')),svgDocuments=svg,localReferences=paths,localAnchors=anchors,routeEntries=20,historicalDesignsNotExecuted=9,errors=errors,reviewSHA256=sha(phase/'research.json'),scope='Generated native research pages and downloads; all local targets and explicit HTML/SVG fragments checked. Fourth-round design statuses are a historical snapshot; later progress is tracked separately. Original external sources are not re-downloaded. Historical workbenches are audited separately.')
text=json.dumps(report,ensure_ascii=False,indent=2)+'\n';dest=phase/'evidence/structure-check.json'
if '--check' in sys.argv:assert dest.read_text()==text,'Saved integration audit stale'
else:dest.write_text(text)
print(text)
if errors:raise SystemExit(1)
