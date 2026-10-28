"""Audit route projection against the historical records and built site."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json,hashlib,xml.etree.ElementTree as ET,sys
root=Path(__file__).resolve().parents[1];history=root/'docs/intelligence-history';dist=root/'dist';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=read(root/'src/data/intelligenceTimeline.generated.json');index=read(history/'research-index.json');batches={name:read(history/name/'research.json') for name in ['round-two','round-three-a','round-three-b','round-three-c','round-three-d','round-three-e','round-six']};errors=[]
for file,expected in data['inputs'].items():
 if sha(history/file)!=expected:errors.append('Changed input: '+file)
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.links=[];self.nodes=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if a.get('id'):self.ids.append(a['id'])
  if a.get('data-node'):self.nodes.append(a['data-node'])
  for name in ['href','src']:
   if a.get(name):self.links.append(a[name])
def parse(file):
 p=Parser();p.feed(file.read_text());return p
cache={}
def parsed(file):
 if file not in cache:cache[file]=parse(file)
 return cache[file]
assert len(data['routes'])==len(index['routes'])==20
samples={};claims=0;sources=0;relations=0;links=0;ancient=0
for route in data['routes']:
 original=next(r for r in index['routes'] if r['id']==route['id']);expected=[(n['id'],n['batch']) for n in original['nodes']]+[(n['id'],'round-six') for n in batches['round-six']['nodes'] if route['id'] in n['routeIds']];assert [(n['id'],n['batch']) for n in route['nodes']]==expected
 page=dist/'research/intelligence/timelines'/route['id']/'index.html';doc=parsed(page)
 if len(doc.ids)!=len(set(doc.ids)):errors.append('Duplicate IDs in '+route['id'])
 if set(doc.nodes)!={n['id'] for n in route['nodes']}:errors.append('Node HTML mismatch '+route['id'])
 for node in route['nodes']:
  b=batches[node['batch']];srcnode=next(n for n in b['nodes'] if n['id']==node['id'])
  assert node['title']==srcnode['title'] and node['question']==srcnode['question']
  for field in ['mechanism','assumptions','limits','nextQuestion']:
   value=srcnode.get(field,'');value='；'.join(value) if isinstance(value,list) else value
   assert node[field]==value,(route['id'],node['id'],field)
  assert len(node['claims'])==len(srcnode['claims'])
  for actual,source in zip(node['claims'],srcnode['claims']):
   assert actual['id']==source['id'] and actual['text']==source['text'] and actual['locator']==source['locator'] and actual['basis']==source['basis']
   assert actual['id'] in doc.ids,(route['id'],actual['id']);claims+=1
  for source in node['sources']:
   primary=srcnode.get('source') if 'source' in srcnode else next(s for s in b['sources'] if s['id']==source['id'])
   assert source['id']==primary['id'] and source['url']==primary['url'] and source['version']==primary['version'] and source['readScope']==primary['readScope'] and source['readingStatus']==primary['readingStatus']
   corrections=[c for c in index['annotations']['sourceCorrections'] if c['batch']==node['batch'] and c['sourceId']==source['id']]
   expected_authors=primary.get('authors',[])
   for c in corrections:expected_authors=[c['new'] if name==c['old'] else name for name in expected_authors]
   assert source['recordedAuthors']==primary.get('authors',[]) and source['authors']==expected_authors and source['corrections']==corrections
   assert source['url'] in doc.links,(route['id'],node['id'],source['id']);sources+=1
  if node['year'] is None:ancient+=1
 for rel in route['relations']:
  source=next(r for r in batches[rel['batch']]['relations'] if r['id']==rel['id'])
  for key in ['fromId','toId','type','status','explanation']:assert rel[key]==source.get(key,''),(route['id'],rel['id'],key)
  relations+=1
 svg=dist/'research/intelligence/timelines'/route['id']/'timeline.svg';tree=ET.parse(svg);hrefs=[a.get('href') for a in tree.findall('.//{http://www.w3.org/2000/svg}a')];dated=[n for n in route['nodes'] if n['year'] is not None]
 assert len(hrefs)==len(dated) and set(hrefs)=={f'/research/intelligence/timelines/{route["id"]}/#{n["id"]}' for n in dated}
 svg_titles=[a.find('{http://www.w3.org/2000/svg}title').text for a in tree.findall('.//{http://www.w3.org/2000/svg}a')]
 assert all('问题：' in t and '方法：' in t and '限制：' in t for t in svg_titles)
 mmd=(dist/'research/intelligence/timelines'/route['id']/'timeline.mmd').read_text()
 assert mmd.count('<br/>问题：')==len(route['nodes']) and mmd.count('<br/>方法：')==len(route['nodes']) and mmd.count('<br/>限制：')==len(route['nodes'])
 assert (dist/'research/intelligence/timelines'/route['id']/'timeline.mmd').is_file() and (dist/'research/intelligence/timelines'/route['id']/'timeline.json').is_file()
 for url in doc.links:
  parsed_url=urlsplit(url)
  if parsed_url.scheme or parsed_url.netloc or not parsed_url.path.startswith(('/research/intelligence/','/routes/')):continue
  target=dist/unquote(parsed_url.path.lstrip('/'))
  if target.is_dir():target=target/'index.html'
  links+=1
  if not target.is_file():errors.append(f'{route["id"]}: missing {url}');continue
  if parsed_url.fragment and target.suffix=='.html' and unquote(parsed_url.fragment) not in parsed(target).ids:errors.append(f'{route["id"]}: missing fragment {url}')
 if route['id'] in ['learning-efficiency','psychometric','philosophy-of-mind']:samples[route['id']]={'nodes':len(route['nodes']),'claims':sum(len(n['claims']) for n in route['nodes']),'relations':len(route['relations'])}
assert len(samples)==3 and ancient==4
report={'date':'2026-10-28','routes':len(data['routes']),'routeNodeOccurrences':sum(len(r['nodes']) for r in data['routes']),'distinctNodes':len({n['id'] for r in data['routes'] for n in r['nodes']}),'sourceReferences':sources,'claimOccurrences':claims,'relationOccurrences':relations,'unknownDateOccurrences':ancient,'sourceCorrectionsApplied':sum(len(source['corrections']) for r in data['routes'] for n in r['nodes'] for source in n['sources']),'samples':samples,'localLinksChecked':links,'errors':errors,'timelineDataSHA256':sha(root/'src/data/intelligenceTimeline.generated.json'),'scope':'Compares all site node/source/claim projections with the original selected batch, checks SVG and internal research links. Original source uptime and unread material not independently checked.'}
text=json.dumps(report,ensure_ascii=False,indent=2)+'\n';output=history/'round-five/evidence/structure-check.json'
if '--check' in sys.argv:assert output.read_text()==text,'Saved audit stale'
else:output.write_text(text)
print(text)
if errors:raise SystemExit(1)
