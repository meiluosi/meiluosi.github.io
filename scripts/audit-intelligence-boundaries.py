"""Audit the phase-two evidence graph, local documents and preserved snapshots."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import json,hashlib,re,sys
import xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1]
history=root/'docs/intelligence-history'
phase=history/'round-four/boundaries'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=read(phase/'research.json');errors=[]
class Markup(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  for key,value in attrs:
   if key in ('href','src') and value:self.links.append(value)
   if key=='id' and value:self.ids.append(value)
def parsed(p):
 m=Markup();m.feed(p.read_text());return m
files=sorted(p for p in phase.rglob('*') if p.suffix in ('.md','.html','.svg'))
files += [history/p for p in ['index.html','coverage.md','questions-index.md','round-four/index.html','round-four/plan.md']]
files += [root/p for p in ['docs/README.md','docs/features.md','docs/roadmap.md','docs/intelligence-history-plan.md']]
# Project documentation includes unrelated historical links. Scope their references to this project only.
paths=anchors=legacy=svg=0
for f in files:
 text=f.read_text();m=parsed(f)
 if f.suffix=='.svg':ET.parse(f);svg+=1
 if f.suffix=='.html' and len(m.ids)!=len(set(m.ids)):errors.append(f'Duplicate HTML id {f.relative_to(root)}')
 urls=m.links+(re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)',text) if f.suffix=='.md' else [])
 for u in urls:
  p=urlsplit(u)
  if p.scheme or p.netloc:continue
  target=(f.parent/unquote(p.path)).resolve() if p.path else f
  if not f.is_relative_to(history) and not target.is_relative_to(history):continue
  paths+=1
  if not target.is_file():errors.append(f'{f.relative_to(root)} -> missing {u}');continue
  if p.fragment:
   if target.is_relative_to(phase) or target in [history/'round-four/index.html']:
    if unquote(p.fragment) not in parsed(target).ids:errors.append(f'{f.relative_to(root)} -> missing anchor {u}')
    anchors+=1
   else:legacy+=1
snapshots={}
for n in ('baselines.json','phase-one-baselines.json'):
 hashes=read(phase/n)
 for p,h in hashes.items():
  if sha(history/p)!=h:errors.append(f'Changed preserved snapshot {p}')
 snapshots[n]=len(hashes)
claims={};source_ids={s['id'] for s in d['sources']};conclusion_ids={c['id'] for c in d['conclusions']}
access=read(phase/'evidence/access-log.json')
for s in d['sources']:
 assert s['readingStatus']=='selected-sections'
 assert all(s[k] for k in ['version','readScope','unreadScope','problem','solution','remaining'])
 assert any(a.get('sha256')==s['sha256'] and a['url']==s['url'] for a in access['retrievals'])
 if s.get('prior'):assert (history/s['prior']).is_file()
 for c in s['claims']:
  assert c['id'] not in claims and c['locator'] and c['kind']=='source-reported';claims[c['id']]=c
for c in d['conclusions']:
 assert c['kind']=='conditional-synthesis' and set(c['evidence'])<=claims.keys()
 assert all(c[k] for k in ['assumptions','objection','response','discriminator','scope'])
for c in d['cases']:assert set(c['sources'])<=source_ids and set(c['propositions'])<=conclusion_ids
for e in d['relations']:assert e['fromId'] in source_ids and e['toId'] in source_ids and set(e['evidence'])<=claims.keys()
for p in d['proposals']:assert p['status']=='design-only' and set(p['links'])<=conclusion_ids and p['checks'] and p['beforeRun']
r=read(phase/'method-check/results.json')
assert all(r['checks'].values()) and r['protocolSHA256']==sha(phase/'method-check/protocol.md')
assert r['scriptSHA256']==sha(root/'scripts/check-intelligence-boundaries.py')
index=read(history/'research-index.json');n=index['counts']
assert n['distinctNodes']==144 and n['nodeRecords']==150 and n['baselineCandidatesWithNotes']==81
assert n['candidatesWithPartialWorkMapping']==1 and n['candidatesWithoutWorkMapping']==24
assert index['roundFour']['currentPhase']==2 and index['roundFour']['nextPhase']==3
assert index['boundariesSHA256']==sha(phase/'research.json')
report=dict(date='2026-10-28',localPathReferences=paths,explicitAnchors=anchors,legacyFragmentsPathOnly=legacy,svgDocuments=svg,snapshots=snapshots,sources=len(d['sources']),selectedSections=len(d['sources']),sourceClaims=len(claims),conclusions=len(conclusion_ids),cases=len(d['cases']),relations=len(d['relations']),editorialComparisons=sum(e['kind']=='editorial-comparison' for e in d['relations']),methodChecks=len(r['checks']),designsNotExecuted=len(d['proposals']),historyConditions=len(r['conditions']),historicalNodeCountsUnchanged=True,errors=errors,scope='Phase-two documents, global entries, phase-one navigation and research links from project docs. External URL uptime and historical dynamic fragments are not tested.')
serialized=json.dumps(report,ensure_ascii=False,indent=2)+'\n';dest=phase/'evidence/structure-check.json'
if '--check' in sys.argv:assert dest.read_text()==serialized,'Saved report stale; review and regenerate.'
else:dest.write_text(serialized)
print(serialized)
if errors:raise SystemExit(1)
