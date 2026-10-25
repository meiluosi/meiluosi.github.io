"""Independent numerical checks of the corrected DPO teaching example."""
import math,json
from pathlib import Path
sig=lambda x:1/(1+math.exp(-x))
loss=lambda h,b:math.log1p(math.exp(-b*h))
h=math.log(.42/.30)-math.log(.22/.20)
assert math.isclose(h,math.log(14/11))
rows=[]
for b in [.1,.5,1,2,10,100]:
 g=-b*sig(-b*h);eps=1e-5
 fd=(loss(h+eps,b)-loss(h-eps,b))/(2*eps)
 assert math.isclose(g,fd,rel_tol=1e-5,abs_tol=1e-10)
 rows.append({'beta':b,'loss':loss(h,b),'gradient':g})
assert abs(rows[-1]['gradient'])<abs(rows[-2]['gradient'])
parts=[math.log(.6/.5),math.log(.7/.6)]
assert not math.isclose(*parts)
assert math.isclose(sum(parts),math.log(1.4))
p=[.18,.42,.22,.18];new=[.212,.378,.198,.212]
assert math.isclose(sum(new),1)
assert math.isclose(math.log(new[1]/.3)-math.log(new[2]/.2),h)
# Fixed-reward optimum and objective identity: J(q)=beta*log Z-beta*KL(q||p*)
optima=[]
for b in [.1,1,10]:
 ref=[.6,.4];r=[1.,0.];z=sum(a*math.exp(v/b) for a,v in zip(ref,r));star=[a*math.exp(v/b)/z for a,v in zip(ref,r)]
 kl=lambda a,c:sum(x*math.log(x/y) for x,y in zip(a,c))
 for q in [[.5,.5],[.7,.3],[.99,.01]]:
  j=sum(x*y for x,y in zip(q,r))-b*kl(q,ref)
  assert math.isclose(j,b*math.log(z)-b*kl(q,star),abs_tol=1e-12)
 optima.append({'beta':b,'p_chosen':star[0],'kl':kl(star,ref)})
Path(__file__).with_name('dpo-audit.json').write_text(json.dumps({'rows':rows,'token_contributions':parts,'optima':optima},indent=2)+'\n')
print('DPO loss, gradients, normalized counterexample and optimum identity passed')
