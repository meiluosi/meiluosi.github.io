"""Toy record checks; no model experiment is represented by these counts."""
import hashlib
import json
from pathlib import Path
rows = [{'task':'a','success':True},{'task':'b','success':False}]
def encode(rows):
    return ''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows).encode()
def digest(blob):
    return hashlib.sha256(blob).hexdigest()
original = encode(rows)
assert digest(original) != digest(encode(rows[::-1]))
changed = [dict(rows[0],success=False),rows[1]]
assert digest(original) != digest(encode(changed))
protocol = {'eval_set':'toy-set-1','scorer':'toy-scorer-1','budget':10,'metric':'success'}
def compatible(a,b):
    return all(a[k]==b[k] for k in protocol)
assert compatible(protocol,dict(protocol))
for key in protocol:
    bad = dict(protocol);bad[key]='different'
    assert not compatible(protocol,bad)
baseline=[60,70,65];candidate=[64,72,66]
diffs=[b-a for a,b in zip(baseline,candidate)]
assert diffs == [4,2,1]
mean=sum(diffs)/len(diffs)
assert abs(mean-7/3)<1e-12
runs=[{'status':'complete','correct':x,'total':100} for x in candidate]
runs.append({'status':'resource_failure','correct':None,'total':None})
completed=[r for r in runs if r['status']=='complete']
assert len(runs)==4 and len(completed)==3 and runs[-1]['correct'] is None
out={'scope':'constructed records, not model results','original_sha256':digest(original),'mean_paired_pp':mean,'candidate_attempts':4,'candidate_completed':3,'rejected_protocol_mutations':len(protocol)}
Path(__file__).with_name('conclusion-examples.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
