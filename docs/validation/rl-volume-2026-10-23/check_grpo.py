import math,json,itertools
from pathlib import Path

def norm(r,ddof=0):
 mean=sum(r)/len(r);s=math.sqrt(sum((x-mean)**2 for x in r)/(len(r)-ddof))
 return s,[0. if s==0 else (x-mean)/s for x in r]
s,a=norm([1,0,0,0]);ss,aa=norm([1,0,0,0],1)
assert abs(s-math.sqrt(3)/4)<1e-12
assert all(abs(x-y)<1e-12 for x,y in zip(a,[math.sqrt(3)]+[-1/math.sqrt(3)]*3))
assert aa==[1.5,-.5,-.5,-.5]
_,small=norm([.51,.50,.50,.50]);assert all(abs(x-y)<1e-12 for x,y in zip(a,small))
assert norm([1]*4)[1]==norm([0]*4)[1]==[0.]*4
mixed={}
for g in [4,8]:
 total=0
 for row in itertools.product([0,1],repeat=g):
  k=sum(row)
  if 0<k<g:total+=.2**k*.8**(g-k)
 closed=1-.2**g-.8**g
 assert abs(total-closed)<1e-12
 mixed[g]=closed
sequence_mean=((2*1)/2+(8*-1)/8)/2
token_mean=(2*1+8*-1)/10
assert sequence_mean==0 and token_mean==-.6
result=dict(kind='constructed arithmetic, no GRPO model training',population_std=s,population_advantages=a,sample_std=ss,sample_advantages=aa,mixed_group_probabilities=mixed,sequence_mean=sequence_mean,token_mean=token_mean)
Path(__file__).with_name('grpo-examples.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
