from pathlib import Path
import json,hashlib,sys,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1];h=root/'docs/intelligence-history';d=json.loads((h/'round-six/research.json').read_text());idx=json.loads((h/'research-index.json').read_text());t=json.loads((root/'src/data/intelligenceTimeline.generated.json').read_text());read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();errors=[]
legacy={a['candidateId'] for a in idx['annotations']['candidateMappings'] if a['status']!='partial-work'}
assert len(legacy)==24 and {p['candidateId'] for p in d['priorities']}==legacy
assert set(d['selectedCandidateIds'])<=legacy and len(d['selectedCandidateIds'])==4
assert sum(p['status']=='unresolved' for p in d['priorities'])==20
assert len(d['nodes'])==len(d['sources'])==6 and len(d['relations'])==4
for node in d['nodes']:
 for routeid in node['routeIds']:
  route=next(r for r in t['routes'] if r['id']==routeid);actual=next(n for n in route['nodes'] if n['id']==node['id']);assert actual['batch']=='round-six' and actual['question']==node['question'] and actual['mechanism']==node['mechanism'] and actual['limits']==node['limits']
  for c in node['claims']:assert any(x['id']==c['id'] and x['text']==c['text'] and x['locator']==c['locator'] for x in actual['claims'])
 for sid in node['sourceIds']:
  s=next(s for s in d['sources'] if s['id']==sid);assert s['readingStatus']=='selected-sections' and s['readScope'] and s['unreadScope'] and s['url'].startswith('https://')
  assert (h/'round-six/notes'/f'{sid}.md').is_file()
for revision in d['revisions']:assert revision['id'] in {'p1','p3','p6'} and set(revision['evidence'])<={n['id'] for n in d['nodes']}
assert idx['counts']['distinctNodes']==144 and idx['roundSix']['remainingUnmapped']==20
assert len({n['id'] for r in t['routes'] for n in r['nodes']})==150
ET.parse(h/'round-six/relations.svg')
p=root/'dist/research/intelligence/revisit/index.html';assert p.is_file();text=p.read_text()
for n in d['nodes']:assert n['id'] in text and n['title'] in text
for sid in d['selectedCandidateIds']:assert any(n['id']==sid for n in d['nodes'])
report={'date':'2026-10-28','baselineUnmapped':24,'selectedCandidates':4,'remainingUnmapped':20,'additionalCounterSources':2,'sources':6,'nodes':6,'claims':sum(len(n['claims']) for n in d['nodes']),'relations':4,'reviewUpdates':3,'historicalDistinctNodes':144,'currentDistinctNodes':150,'routeNodeOccurrences':sum(len(r['nodes']) for r in t['routes']),'errors':errors,'researchSHA256':sha(h/'round-six/research.json'),'scope':'Checks new reading records against the fifth-round candidate snapshot and generated route data; no full-text, primary experiment, or raw-data replication claimed.'}
out=json.dumps(report,ensure_ascii=False,indent=2)+'\n';p=h/'round-six/evidence/structure-check.json'
if '--check' in sys.argv:assert p.read_text()==out
else:p.write_text(out)
print(out)
