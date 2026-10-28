"""Exact enumeration under the saved meaning/consciousness method protocol."""
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1] / 'docs/intelligence-history/round-four/meaning'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
symbols = ['a', 'b', 'c']
objects = ['圆', '方', '三角']
mappings = [dict(zip(symbols, p)) for p in permutations(objects)]
constraints = {
    'none': [],
    'one': [('a', '圆')],
    'two': [('a', '圆'), ('b', '方')],
    'conflict': [('a', '圆'), ('a', '方')],
}
semantic = {key: dict(anchors=anchors, candidates=[m for m in mappings
    if all(m[s] == obj for s, obj in anchors)]) for key, anchors in constraints.items()}
likelihoods = [Fraction(1, 3) ** len(['a', 'b', 'c', 'a']) for _ in mappings]
signals = [dict(S=s, Q=q, R=s*q, A=s, B=s*q) for s,q in product([0,1], repeat=2)]
premises = [dict(T=t, A=a, M=m, O=t*a*m) for t,a,m in product([0,1], repeat=3)]
failed = [x for x in premises if x['O'] == 0]
fixed = [x for x in failed if x['A'] == 1 and x['M'] == 1]
checks = {
    'six_bijections': len(mappings) == 6 and len(semantic['none']['candidates']) == 6,
    'equal_form_likelihood': all(x == Fraction(1,81) for x in likelihoods),
    'one_anchor_two_candidates': len(semantic['one']['candidates']) == 2,
    'two_anchors_one_candidate': semantic['two']['candidates'] == [dict(a='圆', b='方', c='三角')],
    'conflicting_anchors_none': not semantic['conflict']['candidates'],
    'report_required_equivalent': all(x['A'] == x['B'] == x['R'] for x in signals if x['Q'] == 1),
    'no_report_distinguishes_dependency': next(x for x in signals if x['S']==1 and x['Q']==0) == dict(S=1,Q=0,R=0,A=1,B=0),
    'signal_negative_control': next(x for x in signals if x['S']==0 and x['Q']==0) == dict(S=0,Q=0,R=0,A=0,B=0),
    'negative_result_seven_assignments': len(failed) == 7,
    'theory_can_survive_negative': any(x['T'] == 1 for x in failed),
    'independent_premises_isolate_theory': fixed == [dict(T=0,A=1,M=1,O=0)],
}
assert len(checks) == 11 and all(checks.values()), checks
report = dict(date='2026-10-28',kind='exact-synthetic-method-check',
    protocolSHA256=digest(ROOT/'method-check/protocol.md'),scriptSHA256=digest(Path(__file__)),
    mappings=mappings,likelihoods=[str(x) for x in likelihoods],semantic=semantic,signals=signals,
    premises=premises,checks=checks,
    limits='Finite constructed inference checks; not an LLM experiment, consciousness measure, Phi computation or COGITATE replication.')
text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
out = ROOT/'method-check/results.json'
if '--check' in sys.argv:
    assert out.read_text() == text, 'Saved results differ; review before regenerating.'
else:
    out.write_text(text)
print(json.dumps(dict(mode='checked' if '--check' in sys.argv else 'generated',checks=checks)))
