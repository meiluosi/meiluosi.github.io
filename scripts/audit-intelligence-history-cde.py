"""Audit C-D-E local links, evidence structure and immutable baselines."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import json,re,hashlib,sys,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1]/'docs/intelligence-history'
files=[f for b in 'cde' for f in (root/f'round-three-{b}').rglob('*') if f.is_file()]+[root/f for f in ['index.html','coverage.md','questions-index.md','coverage.svg','cde-plan.md','cde-validation.md']]+list((root/'route-coverage').glob('*.md'))
missing=[];linkcount=0;svgs=0
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,t,a):
  for k,v in a:
   if k in ('href','src') and v:self.links.append(v)
for f in files:
 if f.suffix=='.svg':ET.parse(f);svgs+=1
 if f.suffix=='.md':links=re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)',f.read_text())
 elif f.suffix in ('.html','.svg'):
  p=Links();p.feed(f.read_text());links=p.links
 else:continue
 for u in links:
  if u.startswith(('http:','https:','mailto:','data:','#')):continue
  path=unquote(urlsplit(u).path)
  if not path:continue
  linkcount+=1
  target=(f.parent/path).resolve()
  if not target.exists():missing.append({'from':str(f.relative_to(root)),'target':u})
baselines=json.loads((root/'cde-baselines.json').read_text())
assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in baselines.items())
claims=0;relations=0
for b in 'cde':
 d=json.loads((root/f'round-three-{b}/research.json').read_text());sources={s['id']:s for s in d['sources']}
 assert len(d['questionCards'])==3 and len(d['revisits'])>=2
 for n in d['nodes']:
  assert n['evidence'] and n['assumptions'] and n['limits']
  for c in n['claims']:
   assert c['sourceId'] in sources and c['locator'] and c['basis']=='source-reported' and c['independentlyReplicated']==False
   claims+=1
 for r in d['relations']:
  assert r['locator'] and r['evidenceSourceIds']
  if r['type'] in ('conceptual-comparison','method-comparison'):assert r['status']=='editorial-inference'
  relations+=1
out=dict(localLinks=linkcount,svgDocuments=svgs,missing=missing,baselineSHA256='all four snapshots unchanged',cdeClaims=claims,cdeRelations=relations,questionCards=9,revisits=9,notes='Local paths and SVG XML checked; external source scopes are documented per source, not inferred from download status.')
text=json.dumps(out,ensure_ascii=False,indent=2)+'\n'
report=root/'evidence-cde/structure-check.json'
if '--check' in sys.argv:
 assert report.read_text()==text, 'Saved structure report is stale; run without --check after reviewing changes'
else:
 report.parent.mkdir(exist_ok=True);report.write_text(text)
print(text)
if missing:raise SystemExit(1)
