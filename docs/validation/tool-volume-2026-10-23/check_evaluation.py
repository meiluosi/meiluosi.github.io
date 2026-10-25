"""Enumerated teaching metrics; no model measurements."""
from itertools import combinations
from math import comb, isclose
from pathlib import Path
import json
trials=[1,1,1,1,0]
subsets=list(combinations(trials,3))
coverage=sum(any(x) for x in subsets)/len(subsets)
reliability=sum(all(x) for x in subsets)/len(subsets)
assert coverage==1 and reliability==.4
# Verify estimators by exact binomial expectation over possible success counts.
for n in range(1,9):
    for k in range(1,n+1):
        for p in [.1,.5,.8]:
            all_est=any_est=0
            for c in range(n+1):
                weight=comb(n,c)*p**c*(1-p)**(n-c)
                all_est+=weight*comb(c,k)/comb(n,k)
                any_est+=weight*(1-comb(n-c,k)/comb(n,k))
            assert isclose(all_est,p**k,abs_tol=1e-12)
            assert isclose(any_est,1-(1-p)**k,abs_tol=1e-12)
assert isclose(1-.2**3,.992) and isclose(.8**3,.512)
assert (1**3+0**3)/2==.5 and ((1+0)/2)**3==.125
rows=[(True,1,2),(True,2,4),(False,5,20),(False,4,12)]
assert sum(r[1] for r in rows)/4==3
assert sum(r[2] for r in rows)/4==9.5
assert sum(r[1] for r in rows if r[0])/2==1.5
assert sum(r[2] for r in rows if r[0])/2==3
assert sum(r[1] for r in rows)/sum(r[0] for r in rows)==6
before=(60+5)/100;after=(60+15)/100
assert isclose(after-before,.1)
result=dict(subsets=len(subsets),coverage=coverage,reliability=reliability,
            binomial_checks=108,all_task_calls=3,all_task_seconds=9.5,
            success_only_calls=1.5,amortized_calls=6,before=before,after=after,
            regressions=5,model_runs=0)
Path(__file__).with_name('evaluation-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
