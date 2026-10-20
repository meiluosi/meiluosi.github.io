import math,json
scores=[math.log(2),0,0]; exps=[math.exp(x) for x in scores];weights=[x/sum(exps) for x in exps]
values=[(2,0),(0,4),(2,2)]
out=[sum(w*v[j] for w,v in zip(weights,values)) for j in range(2)]
assert out==[1.5,1.5]
def rot(x,a):return (math.cos(a)*x[0]-math.sin(a)*x[1],math.sin(a)*x[0]+math.cos(a)*x[1])
def dot(x,y):return sum(a*b for a,b in zip(x,y))
q=(.3,-.7);k=(1.2,.5);m=2;n=5;angle=.4
assert math.isclose(dot(rot(q,m*angle),rot(k,n*angle)),dot(q,rot(k,(n-m)*angle)),abs_tol=1e-12)
segments=['A']*4+['B']*3
mask=[[j<=i and segments[i]==segments[j] for j in range(7)] for i in range(7)]
assert [j for j in range(7) if mask[5][j]]==[4,5]
labels=[None,'甲','乙','EOS',None,'丙','EOS'];assert sum(y is not None for y in labels[1:])==5
traces={}
for rate in [.5,1.5,2.5]:
 x=2;seq=[x]
 for _ in range(3):x=(1-rate)*x;seq.append(x)
 traces[str(rate)]=seq
assert traces['2.5']==[2,-3,4.5,-6.75]
clip=[v*2/5 for v in [3,4]];assert math.isclose(math.hypot(*clip),2)
assert 4*2*512*8==32768 and 32768*.75==24576
results=dict(kind='independent-teaching-example-checks',attention_weights=weights,attention_output=out,rope_identity=True,packed_document_b_visible_at_position_5=[4,5],effective_targets=5,gradient_descent=traces,clipped_gradient=clip,nominal_positions=32768,effective_targets_at_75_percent=24576,limitations='Arithmetic and mask semantics only; no model training or benchmark.')
print(json.dumps(results,ensure_ascii=False,indent=2))
