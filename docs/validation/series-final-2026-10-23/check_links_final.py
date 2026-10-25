"""Validate rendered internal links and fragments for this volume."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json

root = Path.cwd()
class Page(HTMLParser):
 def __init__(self, text):
  super().__init__(); self.ids=set(); self.links=[]; self.feed(text)
 def handle_starttag(self, tag, attrs):
  data=dict(attrs)
  if 'id' in data: self.ids.add(data['id'])
  if tag=='a' and data.get('href','').startswith(('/', '#')):
   self.links.append(data['href'])

results=[]
inventory=json.loads((root/'docs/validation/series-final-2026-10-23/inventory.json').read_text())
pages=[]
for item in inventory:
 pages.append(root/'dist/posts'/Path(item['path']).stem/'index.html')
 for english in item['english']:
  import re
  slug=re.search(r'^slug: (.+)$',Path(english).read_text(),re.M)[1]
  pages.append(root/'dist/en/posts'/slug/'index.html')
assert len(pages)==84, len(pages)
for path in sorted(pages):
 parsed=Page(path.read_text()); broken=[]
 for link in parsed.links:
  url=urlsplit(link)
  if url.netloc: continue
  dest=(root/'dist'/unquote(url.path).lstrip('/')) if url.path else path
  if dest.is_dir(): dest=dest/'index.html'
  if not dest.exists(): broken.append(link)
  elif url.fragment and dest.suffix=='.html':
   if unquote(url.fragment) not in Page(dest.read_text()).ids: broken.append(link)
 assert not broken, (path,broken)
 results.append(dict(page=str(path.relative_to(root)),links=len(parsed.links),broken=broken))
Path('docs/validation/series-final-2026-10-23/links-final.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
