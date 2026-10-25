import math,json,itertools
from pathlib import Path
p=[.2,.3,.4,.1];r=[1,0,0,0];z=list(map(math.log,p))
def softmax(z):
 a=[math.exp(x-max(z)) for x in z];return [v/sum(a) for v in a]
def objective(z):return sum(a*b for a,b in zip(softmax(z),r))
g=[p[i]*(r[i]-.2) for i in range(4)]
for i in range(4):
 plus=z.copy();minus=z.copy();plus[i]+=1e-6;minus[i]-=1e-6
 assert abs((objective(plus)-objective(minus))/2e-6-g[i])<1e-9
variances={}
for b in [0,.2,2]:
 samples=[[(r[i]-b)*(int(i==j)-p[j]) for j in range(4)] for i in range(4)]
 mean=[sum(p[i]*samples[i][j] for i in range(4)) for j in range(4)]
 assert all(abs(a-c)<1e-12 for a,c in zip(mean,g))
 variances[str(b)]=sum(p[i]*(samples[i][0]-mean[0])**2 for i in range(4))
for k,v in {'0':.1024,'0.2':.0576,'2':.2304}.items():assert abs(variances[k]-v)<1e-12
inclusive=[0.]*4;loo=[0.]*4
for i,k in itertools.product(range(4),repeat=2):
 for j in range(4):
  inclusive[j]+=p[i]*p[k]*sum((r[t]-(r[i]+r[k])/2)*(int(t==j)-p[j])/2 for t in [i,k])
  loo[j]+=p[i]*p[k]*((r[i]-r[k])*(int(i==j)-p[j])+(r[k]-r[i])*(int(k==j)-p[j]))/2
assert all(abs(a-b/2)<1e-12 for a,b in zip(inclusive,g))
assert all(abs(a-b)<1e-12 for a,b in zip(loo,g))
updated=softmax([a+.1*b for a,b in zip(z,g)])
result=dict(kind='finite-policy arithmetic, no LLM training',gradient=g,updated_probabilities=updated,first_coordinate_variances=variances,two_sample_inclusive=inclusive,two_sample_loo=loo)
Path(__file__).with_name('policy-gradient-examples.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
