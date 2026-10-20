"""Check preference examples and BT algebra; no reward model training."""
import json, math, re
from pathlib import Path
root=Path(__file__).resolve().parents[3]
body=(root/'src/content/posts/2026-10-20-llm-training-preference-02-pairs.md').read_text()
record=json.loads(re.search(r'```json\n(.*?)\n```',body,re.S).group(1))
assert record['chosen'] != record['rejected'] and len(record['reason_codes'])==3
sigmoid=lambda x:1/(1+math.exp(-x))
loss=lambda x:math.log1p(math.exp(-x))
deltas=[0,math.log(3),math.log(9),-math.log(3)]
assert all(math.isclose(sigmoid(x),p) for x,p in zip(deltas,[.5,.75,.9,.25]))
assert math.isclose(sigmoid(2-1),sigmoid(102-101))
assert not math.isclose(sigmoid(2-1),sigmoid(4-2))
h=1e-5
for d in deltas:
 derivative=(loss(d+h)-loss(d-h))/(2*h)
 assert math.isclose(derivative,sigmoid(d)-1,abs_tol=1e-9)
pairs=[(a,b) for a in range(4) for b in range(a+1,4)]
assert len(pairs)==6
assert all(sum(i in p for p in pairs)==3 for i in range(4))
print(json.dumps({'kind':'teaching-example-checks','pair_record_valid':True,'BT_probabilities':[sigmoid(x) for x in deltas],'losses':[loss(x) for x in [0,math.log(3),-math.log(3)]],'finite_difference_gradient_passed':True,'translation_invariant':True,'scaling_changes_probability':True,'four_candidates_pairs':pairs,'scope':'Toy arithmetic and record structure only; no trained model.'},ensure_ascii=False,indent=2))
