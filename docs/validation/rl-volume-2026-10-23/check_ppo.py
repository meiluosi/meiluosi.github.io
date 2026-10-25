from pathlib import Path
import json,math

def objective(r,a,e=.2):return min(r*a,min(1+e,max(1-e,r))*a)
rows=[(2,1.4,2.4,0),(2,.6,1.2,2),(-2,.6,-1.6,0),(-2,1.4,-2.8,-2)]
for a,r,value,slope in rows:
 assert math.isclose(objective(r,a),value)
 derivative=(objective(r+1e-6,a)-objective(r-1e-6,a))/2e-6
 assert abs(derivative-slope)<1e-8

def gae(rewards,values,gamma,lam,last):
 deltas=[rewards[i]+gamma*(values[i+1] if i+1<len(values) else last)-values[i] for i in range(len(values))]
 out=[0]*len(values);acc=0
 for i in reversed(range(len(values))):
  acc=deltas[i]+gamma*lam*acc;out[i]=acc
 return deltas,out
r=[0,0,1];v=[.2,.4,.7]
deltas,half=gae(r,v,1,.5,0);_,full=gae(r,v,1,1,0)
for actual,expected in [(deltas,[.2,.3,.3]),(half,[.425,.45,.3]),(full,[.8,.6,.3])]:
 assert all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
# Collection cutoff example: preserve a nonzero continuation estimate.
_,continued=gae([0,0],v[:2],1,1,.7)
assert all(abs(a-b)<1e-12 for a,b in zip(continued,[.5,.3]))
result=dict(kind='constructed PPO arithmetic, no model training',clip_cases=rows,td_residuals=deltas,gae_half=half,gae_one=full,cutoff_bootstrap=continued)
Path(__file__).with_name('ppo-examples.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
