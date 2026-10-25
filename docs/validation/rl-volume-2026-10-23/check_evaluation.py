from pathlib import Path
import math,json,itertools
N,C,k=10,2,4
estimate=1-math.comb(N-C,k)/math.comb(N,k)
subsets=list(itertools.combinations(range(N),k))
exact=sum(any(i<C for i in row) for row in subsets)/len(subsets)
assert abs(estimate-exact)<1e-12 and abs(estimate-2/3)<1e-12
# Expectation under the stated independent binomial sampling assumption.
p=.2
mean=sum(math.comb(N,c)*p**c*(1-p)**(N-c)*(1-(math.comb(N-c,k) if N-c>=k else 0)/math.comb(N,k)) for c in range(N+1))
assert abs(mean-(1-(1-p)**k))<1e-12
paired=[50,20,10,20];assert sum(paired)==100
old=(paired[0]+paired[2])/100;new=(paired[0]+paired[1])/100
entropy=lambda ps:-sum(p*math.log(p) for p in ps if p)
h1=entropy([.25]*4);h2=entropy([.7,.1,.1,.1])
assert abs(h1-1.38629436112)<1e-10 and abs(h2-.940447988655)<1e-10
result=dict(kind='synthetic evaluation arithmetic, no model evaluation',pass_k_observation=estimate,pass_k_expectation=mean,before=old,after=new,repairs=20,regressions=10,entropy_uniform=h1,entropy_concentrated=h2)
Path(__file__).with_name('evaluation-examples.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
