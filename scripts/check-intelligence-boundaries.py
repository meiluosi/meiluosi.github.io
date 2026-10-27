"""Exact checks for the prespecified developmental-state and feedback examples."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1] / 'docs/intelligence-history/round-four/boundaries'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def history(bits):
    h = F(0)
    trajectory = [h]
    for e in bits:
        h = (h + int(e)) / 2
        trajectory.append(h)
    return trajectory

histories = {s: history(s) for s in ['0000', '1111', '1100', '0011']}
conditions = []
for bits, trace in histories.items():
    for reset in [False, True]:
        h = F(0) if reset else trace[-1]
        for support in [0, 1]:
            conditions.append(dict(history=bits, reset=reset, support=support,
                state=str(h), output=str(h + support)))

def loop(perturb=False):
    x, e = F(0), F(1)
    trace = []
    for t in range(7):
        if perturb and t == 2:
            x = F(1)
        trace.append((x, e))
        x, e = (x + e) / 2, (e + x) / 2
    return trace

baseline = loop()

def replay(perturb=False):
    x = F(0)
    trace = []
    for t in range(7):
        if perturb and t == 2:
            x = F(1)
        e = baseline[t][1]
        trace.append((x, e))
        x = (x + e) / 2
    return trace

checks = {
    'terminal_states': histories['0000'][-1] == 0 and histories['1111'][-1] == F(15,16),
    'same_removal_loss': all((h[-1] + 1) - h[-1] == 1 for h in histories.values()),
    'history_at_fixed_support': all(histories['1111'][-1] + u != histories['0000'][-1] + u for u in [0,1]),
    'order_not_exposure_count': histories['1100'][-1] == F(3,16) and histories['0011'][-1] == F(3,4),
    'state_reset_mediates_history': all(len({c['output'] for c in conditions if c['reset'] and c['support']==u})==1 for u in [0,1]),
    'baseline_trajectory_equivalence': baseline == replay(),
    'feedback_intervention': loop(True)[3][1] == F(3,4) and replay(True)[3][1] == F(1,2),
    'synchronous_negative_control': loop(False) == replay(False) and baseline[1] == (F(1,2),F(1,2))
}
assert all(checks.values()), checks
report = dict(date='2026-10-28', kind='exact-synthetic-method-check',
    protocolSHA256=digest(ROOT/'method-check/protocol.md'), scriptSHA256=digest(Path(__file__)),
    histories={s:[str(x) for x in t] for s,t in histories.items()}, conditions=conditions,
    feedback={name:[dict(t=i,x=str(x),e=str(e)) for i,(x,e) in enumerate(trace)]
        for name,trace in [('loopBaseline',baseline),('replayBaseline',replay()),('loopPerturbed',loop(True)),('replayPerturbed',replay(True))]},
    checks=checks,
    limits='Constructed finite models; not a replication of developmental, perceptual-crossing or autopoiesis experiments. Feedback is not itself a criterion of cognitive constitution.')
text = json.dumps(report,ensure_ascii=False,indent=2)+'\n'
out = ROOT/'method-check/results.json'
if '--check' in sys.argv:
    assert out.read_text()==text, 'Saved results differ; review before regenerating.'
else:
    out.write_text(text)
print(json.dumps(dict(mode='checked' if '--check' in sys.argv else 'generated', checks=checks)))
