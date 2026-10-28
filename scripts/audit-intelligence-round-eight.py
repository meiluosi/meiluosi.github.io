from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1];d=root/'docs/intelligence-history/round-eight';read=lambda p:json.loads(p.read_text());data=read(d/'research.json');idx=read(root/'docs/intelligence-history/research-index.json');reviews=read(root/'docs/intelligence-history/round-four/integration/research.json')['reviews'];out=root/'dist'
assert len(data['axes'])==9 and len(data['comparisons'])==6
assert set(data['questionAssignments'])=={q['id'] for q in idx['questionCards']} and len(data['questionAssignments'])==15
assert {c for a in data['axes'] for c in a['conclusions']}=={r['id'] for r in reviews} and len(reviews)==18
assert all(a['judgment'] and a['objection'] and a['test'] and a['relatedEvidence'] for a in data['axes'])
assert data['status']=='provisional-editorial-synthesis' and len(data['limitations'])>=4
assert (d/'map.mmd').read_bytes()==(root/'public/research/intelligence/map/map.mmd').read_bytes()
assert (d/'map.svg').read_bytes()==(root/'public/research/intelligence/map/map.svg').read_bytes()
svg=ET.parse(d/'map.svg');assert len(svg.findall('.//{http://www.w3.org/2000/svg}a'))==9
class Page(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k=='id' and v:self.ids.append(v)
   if k in ('href','src') and v:self.links.append(v)
def parse(p):
 h=Page();h.feed(p.read_text());return h
page=out/'research/intelligence/map/index.html';h=parse(page);assert len(h.ids)==len(set(h.ids))
for x in data['axes']:assert x['id'] in h.ids and x['judgment'] in page.read_text() and x['objection'] in page.read_text()
for q in idx['questionCards']:assert q['id'] in h.ids and q['title'] in page.read_text()
errors=[];checked=0
for url in h.links:
 p=urlsplit(url)
 if p.scheme or p.netloc or not p.path.startswith('/research/intelligence/'):continue
 target=out/unquote(p.path.lstrip('/'));target=target/'index.html' if target.is_dir() else target;checked+=1
 if not target.is_file():errors.append(url);continue
 if p.fragment and target.suffix=='.html' and unquote(p.fragment) not in parse(target).ids:errors.append(url)
assert not errors,errors
for n in ['research/intelligence/index.html','routes/connectionist/index.html']:
 assert '/research/intelligence/map/' in (out/n).read_text()
report={'date':'2026-10-28','questionDomains':9,'priorQuestionCards':15,'reviewedConclusionsReferenced':18,'comparisons':6,'svgLinks':9,'internalLinksChecked':checked,'errors':errors,'scope':'Checks structural mapping, link targets and provenance; does not validate theories or imply proposed tests were run.'}
(d/'evidence/structure-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
