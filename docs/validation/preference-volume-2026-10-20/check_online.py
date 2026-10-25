import math,json
from pathlib import Path
p=[.05,.15,.30,.5];z=list(map(math.log,p));b=.1
sig=lambda v:1/(1+math.exp(-v))
a=b*sig(-b*(z[0]-z[2]));c=b*sig(-b*(z[0]-z[3]));direction=[a+c,0,-a,-c]
def loss(z):return sum(math.log1p(math.exp(-b*(z[0]-z[j]))) for j in [2,3])
for i in range(4):
 plus=z.copy();minus=z.copy();plus[i]+=1e-6;minus[i]-=1e-6
 assert math.isclose((loss(plus)-loss(minus))/2e-6,-direction[i],abs_tol=1e-9)
mean=sum(x*y for x,y in zip(p,direction));eta=1e-5
new=[math.exp(x+eta*d) for x,d in zip(z,direction)];total=sum(new);new=[x/total for x in new]
assert new[1]>p[1]
assert math.isclose((math.log(new[1])-math.log(p[1]))/eta,-mean,abs_tol=1e-7)
def norm(xs):
 m=sum(xs)/len(xs);sd=math.sqrt(sum((x-m)**2 for x in xs)/len(xs));return [(x-m)/sd for x in xs]
assert norm([0,1,2])!=norm([0,1,100])
out={'logit_direction':direction,'weighted_mean':mean,'y2_log_probability_derivative':-mean,'coverage':{str(n):1-.95**n for n in [20,200]},'expected_pairs_exactly_one_y1':100*2*.05*.95,'online_answers_50_steps':10000,'online_pair_judgments_50_steps':5000,'scope':'toy only'}
Path(__file__).with_name('online-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
