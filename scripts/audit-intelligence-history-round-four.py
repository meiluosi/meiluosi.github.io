"""Audit round-four evidence references, local paths, explicit anchors and snapshots."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1] / 'docs/intelligence-history'
phase = root / 'round-four'
read = lambda p: json.loads(p.read_text())
data = read(phase / 'research.json')
index = read(root / 'research-index.json')
annotations = read(root / 'index-annotations.json')
errors = []
files = sorted(phase.rglob('*')) + [root / n for n in
    ['index.html', 'coverage.md', 'questions-index.md', 'coverage.svg']]
files += sorted((root / 'route-coverage').glob('*.md'))

class Markup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                self.links.append(value)
            if key == 'id' and value:
                self.ids.append(value)

paths = anchors = svg_count = legacy_fragments = 0
for file in files:
    if not file.is_file() or file.suffix not in ('.md', '.html', '.svg'):
        continue
    text = file.read_text()
    parser = Markup()
    parser.feed(text)
    if file.suffix == '.svg':
        ET.parse(file)
        svg_count += 1
    if file.suffix == '.html' and len(set(parser.ids)) != len(parser.ids):
        errors.append(f'Duplicate HTML ids: {file.relative_to(root)}')
    links = parser.links
    if file.suffix == '.md':
        links += re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)', text)
    for url in links:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            continue
        target = (file.parent / unquote(parts.path)).resolve() if parts.path else file
        paths += 1
        if not target.is_file():
            errors.append(f'{file.relative_to(root)} → missing {url}')
            continue
        if parts.fragment:
            if target.is_relative_to(phase) and target.suffix in ('.md', '.html', '.svg'):
                p = Markup()
                p.feed(target.read_text())
                if unquote(parts.fragment) not in p.ids:
                    errors.append(f'{file.relative_to(root)} → missing explicit anchor {url}')
                anchors += 1
            else:
                legacy_fragments += 1

baselines = read(phase / 'baselines.json')
for file, expected in baselines.items():
    if hashlib.sha256((root / file).read_bytes()).hexdigest() != expected:
        errors.append(f'Historical snapshot changed: {file}')
claims = {c['id'] for s in data['sources'] for c in s['claims']}
conclusions = {c['id'] for c in data['conclusions']}
assert len(claims) == sum(len(s['claims']) for s in data['sources'])
for s in data['sources']:
    assert s['readScope'] and s['unreadScope'] and s['version']
    assert s['readingStatus'] in ('selected-sections', 'full-text')
    for c in s['claims']:
        assert c['locator'] and c['kind'] == 'source-reported'
for c in data['conclusions']:
    assert c['kind'] in ('conditional-synthesis', 'derived-and-editorial')
    assert c['evidence'] and set(c['evidence']) <= claims
    assert all(c[k] for k in ('assumptions', 'objection', 'response', 'discriminator', 'scope'))
assert {a['conclusion'] for a in data['evidenceAudit']} == conclusions
for p in data['proposals']:
    assert p['status'] == 'design-only' and set(p['links']) <= conclusions
assert len(annotations['candidateMappings']) == 25
counts = index['counts']
assert counts['distinctNodes'] == 144 and counts['nodeRecords'] == 150
assert counts['baselineCandidatesWithNotes'] == 81
assert counts['candidatesWithPartialWorkMapping'] == 1
assert counts['candidatesWithoutWorkMapping'] == 24
report = dict(date='2026-10-28', localPathReferences=paths,
    explicitRoundFourAnchorsChecked=anchors, legacyFragmentsPathOnly=legacy_fragments,
    svgXMLDocuments=svg_count, historicalSnapshotsUnchanged=len(baselines),
    sources=len(data['sources']), sourceClaims=len(claims), conclusions=len(conclusions),
    evidenceAuditItems=len(data['evidenceAudit']), designsNotExecuted=len(data['proposals']),
    candidateAnnotations=len(annotations['candidateMappings']), errors=errors,
    scope='Round-four documents and current global coverage/index. Historical target paths checked; historical dynamic anchors and external URLs are outside this structural audit.')
serialized = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
destination = phase / 'evidence/structure-check.json'
if '--check' in sys.argv:
    assert destination.read_text() == serialized, 'Saved audit is stale; review and regenerate.'
else:
    destination.write_text(serialized)
print(serialized)
if errors:
    raise SystemExit(1)
