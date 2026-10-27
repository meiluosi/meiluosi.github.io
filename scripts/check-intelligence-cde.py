"""Exact, bounded synthetic checks for research batches C, D, E. No empirical replication."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1] / 'docs/intelligence-history'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def causal():
    def distribution(model, intervention=None):
        out = {f'{x},{y}': F(0) for x, y in product(range(2), repeat=2)}
        for u in (0, 1):
            x = u if intervention is None else intervention
            y = x if model == 'X_to_Y' else u
            out[f'{x},{y}'] += F(1, 2)
        assert sum(out.values()) == 1
        return out
    observed = [distribution(m) for m in ('X_to_Y', 'U_common_cause')]
    assert observed[0] == observed[1] == {'0,0': F(1,2), '0,1': F(0), '1,0': F(0), '1,1': F(1,2)}
    interventions = {m: {str(x): distribution(m, x) for x in (0, 1)} for m in ('X_to_Y', 'U_common_cause')}
    means = {m: [sum(v for k, v in interventions[m][str(x)].items() if k.endswith(',1')) for x in (0, 1)] for m in interventions}
    assert means == {'X_to_Y': [F(0), F(1)], 'U_common_cause': [F(1,2), F(1,2)]}
    assert interventions['U_common_cause']['1']['1,0'] == F(1,2)
    return dict(observed=observed, interventions=interventions, means=means, checks=['观察联合分布一致', '各分布归一化', '干预均值符合独立解析表', '共同原因模型do(X=1)仍有Y=0'])

def collective():
    # Numerators use integer Rademacher variables; divide squared moments by a²+b².
    def enumerate_moments(n, a, b, matrix=False):
        total = count = 0
        cov = [[0]*n for _ in range(n)]
        states = product((-1,1), repeat=n if b else 0)
        for independent in states:
            for shared in ((-1,1) if a else (0,)):
                e = [a*shared + b*(independent[i] if b else 0) for i in range(n)]
                total += sum(e)**2; count += 1
                if matrix:
                    for i in range(n):
                        for j in range(n): cov[i][j] += e[i]*e[j]
        scale = (a*a+b*b)*count
        variance = F(total, scale*n*n)
        if matrix:
            cov = [[F(v,scale) for v in row] for row in cov]
            assert all(cov[i][i] == 1 for i in range(n))
            assert sum(map(sum,cov))/n**2 == variance
        return variance, cov if matrix else None, count
    cases = {}
    for label,a,b,expected in [('independent',0,1,F(1,4)),('shared',1,0,F(1)),('mixed',1,1,F(5,8))]:
        v,cov,count=enumerate_moments(4,a,b,True)
        assert v == expected == F(a*a,a*a+b*b)+F(b*b,4*(a*a+b*b))
        cases[label]=dict(n=4, variance=v, covariance=cov, enumeratedStates=count)
    v16,_,count16=enumerate_moments(16,0,1)
    shared16,_,_=enumerate_moments(16,1,0)
    assert v16 == F(1,16) and shared16 == 1
    return dict(cases=cases,n16IndependentVariance=v16,n16SharedVariance=shared16,n16EnumeratedStates=count16,checks=['四人个体方差均为1','枚举、协方差求和与解析式相同','共享误差不随人数下降','独立误差16人方差为1/16'])

def order_effect():
    A=[(F(1),F(0)),(F(0),F(1))]
    B=[(F(3,5),F(4,5)),(F(-4,5),F(3,5))]
    psi=A[0]
    def dot(x,y): return sum(a*b for a,b in zip(x,y))
    def project(v,state): return tuple(dot(v,state)*x for x in v)
    def quantum(first,second):
        return [[dot(t,t) for w in second for t in [project(w,project(v,psi))]] for v in first]
    qAB=quantum(A,B); qBA=quantum(B,A)
    # Independently specified finite-state Markov machine, states record latest question/answer.
    transition=[[F(9,25),F(16,25)],[F(16,25),F(9,25)]]
    def classic(initial): return [[initial[i]*transition[i][j] for j in range(2)] for i in range(2)]
    cAB=classic([F(1),F(0)]); cBA=classic([F(9,25),F(16,25)])
    assert qAB == cAB == [[F(9,25),F(16,25)],[F(0),F(0)]]
    assert qBA == cBA == [[F(81,625),F(144,625)],[F(256,625),F(144,625)]]
    assert sum(map(sum,qAB)) == sum(map(sum,qBA)) == 1
    aAfterB=sum(row[0] for row in qBA)
    assert aAfterB == F(337,625) and sum(qAB[0]) == 1
    identical=quantum(A,A)
    assert identical == [[F(1),F(0)],[F(0),F(0)]]
    return dict(quantumAB=qAB,quantumBA=qBA,classicalAB=cAB,classicalBA=cBA,AFirstYes=F(1),AAfterBYes=aAfterB,identicalBasisNegativeControl=identical,checks=['两种实现的AB和BA联合分布逐项相等','两种次序均归一化','A的顺序效应为1与337/625','同基负控制无顺序效应'])

batch=next((x for x in sys.argv[1:] if x in ('c','d','e')),None)
if batch is None: raise SystemExit('Usage: python3 scripts/check-intelligence-cde.py c|d|e [--check] [--plot]')
dir=ROOT/f'round-three-{batch}'/'method-check'
result=dict(batch=batch,status='synthetic-verified',protocolSHA256=sha(dir/'protocol.md'),implementationSHA256=sha(Path(__file__)),result={'c':causal,'d':collective,'e':order_effect}[batch]())
def encode(v):
    if isinstance(v,F):return str(v)
    if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
    if isinstance(v,list):return [encode(x) for x in v]
    return v
text=json.dumps(encode(result),ensure_ascii=False,indent=2)+'\n'
path=dir/'results.json'
if '--check' in sys.argv:
    assert path.read_text()==text, 'Saved results or provenance changed'
else:path.write_text(text)
if '--plot' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
    if batch=='c':
        labels=['X → Y\ndo(X=0)','X → Y\ndo(X=1)','U → X, Y\ndo(X=0)','U → X, Y\ndo(X=1)'];values=[0,1,.5,.5];title='Same observations, different interventions';ylabel='E[Y | do(X)]'
    elif batch=='d':
        labels=['Independent\nn=4','Shared\nn=4','Mixed\nn=4','Independent\nn=16','Shared\nn=16'];values=[.25,1,.625,.0625,1];title='Averaging cannot remove shared error';ylabel='Variance of group mean (individual variance = 1)'
    else:
        labels=['A first\nquantum','A after B\nquantum','A first\nclassical','A after B\nclassical'];values=[1,337/625,1,337/625];title='Order effects do not identify a unique physical implementation';ylabel='P(A = yes)'
    ax.bar(labels,values,color=['#356a5c','#a75532','#739c86','#c19a67','#356a5c'][:len(labels)])
    ax.set_ylim(0,1.14);ax.set_ylabel(ylabel);ax.set_title(title,fontsize=11)
    for i,v in enumerate(values):ax.text(i,v+.025,f'{v:g}',ha='center',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.text(.5,-.19,'Exact synthetic examples; not an empirical replication',ha='center',transform=ax.transAxes,fontsize=9)
    fig.savefig(dir/'check.svg',metadata={'Date':None});plt.close(fig)
print(f'{batch.upper()}: 4 exact checks passed; protocol and implementation fingerprints recorded.')
