"""Validate rendered internal links and fragments for this volume."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json

root = Path(__file__).resolve().parents[3]
class Page(HTMLParser):
 def __init__(self, text):
  super().__init__(); self.ids=set(); self.links=[]; self.feed(text)
 def handle_starttag(self, tag, attrs):
  data=dict(attrs)
  if 'id' in data: self.ids.add(data['id'])
  if tag=='a' and data.get('href','').startswith(('/', '#')):
   self.links.append(data['href'])

results=[]
pages=list((root/'dist/posts').glob('2026-10-23-llm-training-rl-*/index.html'))
pages+=list((root/'dist/en/posts').glob('llm-training-rl-*/index.html'))
assert pages, 'Build the site first'
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
Path(__file__).with_name('links-current.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
