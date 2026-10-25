"""Independent reward calculations; no training or environment execution."""
import itertools
import json
import math
from pathlib import Path

def returns(rewards, gamma):
    return [sum(gamma**(k-t)*r for k,r in enumerate(rewards) if k>=t)
            for t in range(len(rewards))]

def close(xs, ys):
    assert len(xs)==len(ys)
    assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(xs,ys))

close(returns([0,0,1],1),[1,1,1])
close([1-v for v in [.6,.7,.9]],[.4,.3,.1])
close(returns([-.05,-.05,1],1),[.9,.95,1])
close(returns([-.05,-.05,1],.9),[.715,.85,1])
assert math.isclose(returns([-.05,1],.9)[0],.85)
assert 1-2*.05 > .9-.05
assert 1-2*.2 < .9-.2
assert math.isclose(1-2*.1,.9-.1)
assert math.isclose(10*(.1-.05),.5)
# Enumerate finite potential paths, including nonzero terminal potential:
# telescope always holds, but only zero terminal yields the stated constant.
checked=0
for gamma in [.5,.9,1]:
    for phi in itertools.product([0,.2,.6],repeat=4):
        shaped=[gamma*b-a for a,b in zip(phi,phi[1:])]
        total=sum(gamma**t*f for t,f in enumerate(shaped))
        boundary=-phi[0]+gamma**3*phi[-1]
        assert math.isclose(total,boundary,abs_tol=1e-12)
        if phi[-1]==0:
            assert math.isclose(total,-phi[0],abs_tol=1e-12)
        checked+=1
close([.2-0,.6-.2,0-.6],[.2,.4,-.6])
assert math.isclose(.2+.4,.6) # omitting terminal correction changes return
# Enumerate the full two-attempt outcome space; second call is only used
# after first failure, even though latent second outcomes are enumerated.
success=expected_calls=0
for first,second in itertools.product([False,True],repeat=2):
    probability=(.8 if first else .2)*(.8 if second else .2)
    success+=probability*int(first or second)
    expected_calls+=probability*(1 if first else 2)
assert math.isclose(success,.96)
assert math.isclose(expected_calls,1.2)
# Perfectly correlated failures are a counterexample to independent retry gains.
correlated_success=.8
assert correlated_success < success
result=dict(returns=returns([-.05,-.05,1],1),discounted=returns([-.05,-.05,1],.9),
            shaping_paths=checked,retry_success=success,retry_calls=expected_calls,
            correlated_retry_success=correlated_success,model_runs=0)
Path(__file__).with_name('reward-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
