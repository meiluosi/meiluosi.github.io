"""Synthetic vote checks, no real raters or judge model."""
import json
from pathlib import Path
from math import isclose

def identity(position,order):
    if position=='tie':return 'tie'
    if position not in (1,2):raise ValueError('invalid judge output')
    return order[position-1]
# Positions in AB order, followed by positions in BA order.
pairs=[(1,2)]*4+[(2,1)]*3+[('tie','tie')]+[(1,1),(2,2)]
normalized=[(identity(a,('A','B')),identity(b,('B','A'))) for a,b in pairs]
assert sum(a==b for a,b in normalized)==8
assert sum(a==b=='A' for a,b in normalized)==4
assert sum(a==b=='B' for a,b in normalized)==3
try:identity('unparseable',('A','B'))
except ValueError:pass
else:raise AssertionError('Parser failure treated as valid vote')
assert 40/100==.4 and isclose(40/70,4/7) and (40+30/2)/100==.55
labels=[True]*90+[False]*10
predictions=[True]*100
agreement=sum(a==b for a,b in zip(labels,predictions))/100
failure_detection=sum(not p for label,p in zip(labels,predictions) if not label)/10
assert agreement==.9 and failure_detection==0
result=dict(stable_pairs=8,total_pairs=10,consistency=.8,
            win_fraction=.4,decisive_win_fraction=40/70,tie_half_score=.55,
            always_pass_agreement=agreement,failure_detection=failure_detection,
            judge_calls=0)
Path(__file__).with_name('evidence-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
