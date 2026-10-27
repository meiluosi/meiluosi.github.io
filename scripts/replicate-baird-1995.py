"""Independent replication of the 15 July 1995 six-state star (not Figure 5).

Run with Python + numpy + matplotlib. Parameters are fixed in protocol.md.
The scalar, weight-matrix, value-matrix and finite-difference checks use
separate formulations so they can reveal update/sign/normalization errors.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/intelligence-history/round-three-a/replication'
OUT.mkdir(parents=True, exist_ok=True)
GAMMA, ALPHA = 0.99, 0.01
X = np.zeros((6, 7))
X[:5, 0] = 1
for i in range(5):
    X[i, i + 1] = 2
X[5, 0], X[5, 6] = 2, 1
P = np.zeros((6, 6)); P[:, 5] = 1
W0 = np.array([1., 1., 1., 1., 1., 1., 10.])
V0 = X @ W0

def operators(gamma, alpha):
    D = gamma * P - np.eye(6)
    B = D @ X
    wd = np.eye(7) + alpha * X.T @ B
    wr = np.eye(7) - alpha * B.T @ B
    vd = np.eye(6) + alpha * (X @ X.T) @ D
    vr = np.eye(6) - alpha * (X @ X.T) @ D.T @ D
    return D, B, wd, wr, vd, vr

D, B, MD, MR, VD, VR = operators(GAMMA, ALPHA)
# Equation 7 as printed has the opposite sign to equation 3. The main
# implementation follows equation 3 and section 5's epoch aggregation.
literal_eq7 = np.eye(6) - ALPHA * (X @ X.T) @ D
tabular_eq3 = 1 - ALPHA * (1-GAMMA)
tabular_eq7 = 1 + ALPHA * (1-GAMMA)
assert 0 < tabular_eq3 < 1 < tabular_eq7

def scalar_step(w, method, gamma=GAMMA, alpha=ALPHA):
    # All deltas use old weights; no in-place sequential state update here.
    values = [w[0] + 2*w[i] for i in range(1,6)] + [2*w[0] + w[6]]
    change = [0.] * 7
    for state in range(6):
        delta = gamma * values[5] - values[state]
        grad = [0.] * 7
        if state < 5:
            grad[0], grad[state + 1] = 1., 2.
        else:
            grad[0], grad[6] = 2., 1.
        if method == 'residual':
            grad[0] -= gamma * 2
            grad[6] -= gamma
        for j in range(7):
            change[j] += alpha * delta * grad[j]
    return np.array([w[j]+change[j] for j in range(7)])

assert np.linalg.matrix_rank(X) == 6
scalar_errors = {}
for name, M in [('direct', MD), ('residual', MR)]:
    a, b = W0.copy(), W0.copy()
    max_error = 0.
    for _ in range(100):
        a, b = scalar_step(a, name), M @ b
        max_error = max(max_error, float(np.max(np.abs(a-b))))
    assert max_error < 1e-11
    scalar_errors[name] = max_error
# Independent derivative check of J = 1/2 sum(delta^2); paper eq.8 sums states.
probe = np.array([.3, -.8, 1.1, .6, -.2, 1.4, 2.])
def loss(w):
    v = X @ w
    delta = GAMMA * v[5] - v
    return .5 * float(delta @ delta)
fd = []
for j in range(7):
    h = np.zeros(7); h[j] = 1e-5
    fd.append((loss(probe+h)-loss(probe-h))/(2e-5))
grad_error = float(np.max(np.abs(np.array(fd) - B.T @ B @ probe)))
assert grad_error < 1e-6

# Actual repeated updates. Record logarithmic and early steps, not every state.
samples = set([0,1,2,5,10,100,1000,10000,100000])
samples.update(map(int, np.geomspace(1,100000,650)))
traces = {}
monotone_checks = {}
for name, M, count in [('direct',MD,10000),('residual',MR,100000)]:
    w=W0.copy(); rows=[]; prev=float('inf'); increases=0
    for t in range(count+1):
        v=X@w; delta=D@v
        mse=float(delta@delta/6); rmse=float(np.sqrt(v@v/6))
        if mse>prev+1e-12*max(1.,prev): increases+=1
        prev=mse
        if t in samples:
            rows.append({'step':t,'method':name,'evaluation':'iterative','bellmanMSE':mse,'valueRMSE':rmse,'V1':float(v[0]),'V6':float(v[5]),'weightNorm':float(np.linalg.norm(w))})
        if t<count: w=M@w
    traces[name]=rows
    monotone_checks[name]=increases
    exact=np.linalg.matrix_power(M,count)@W0
    assert np.allclose(w,exact,rtol=1e-8,atol=1e-8)
    # Value-space recurrence has no parameter nullspace.
    VM=VD if name=='direct' else VR
    assert np.allclose(X@w,np.linalg.matrix_power(VM,count)@V0,rtol=1e-8,atol=1e-8)
assert traces['direct'][-1]['valueRMSE'] > 1e6
assert monotone_checks['residual']==0
# Evaluate the SAME linear recurrence by repeated squaring at long horizons.
fast=[]
for t in [100000,1000000,10000000,50000000]:
    v=np.linalg.matrix_power(VR,t)@V0; delta=D@v
    fast.append({'step':t,'method':'residual','evaluation':'matrix-power','bellmanMSE':float(delta@delta/6),'valueRMSE':float(np.sqrt(v@v/6)),'V1':float(v[0]),'V6':float(v[5])})
assert fast[-1]['valueRMSE'] < 1e-3
assert max(abs(np.linalg.eigvals(VR))) < 1
assert max(abs(np.linalg.eigvals(VD))) > 1
# Derive and check the two-dimensional symmetry reduction independently.
reduced=np.eye(2)+ALPHA*np.array([[-9,11*GAMMA-2],[-10,15*GAMMA-5]])
for t in [1,100,10000]:
    pair=np.linalg.matrix_power(reduced,t)@np.array([3.,12.])
    full=np.linalg.matrix_power(VD,t)@V0
    assert np.allclose(pair,full[[0,5]],rtol=1e-8,atol=1e-8)

sensitivity=[]
for gamma in [.9,.95,.99]:
    for alpha in [.001,.01]:
        dd,bb,md,mr,vd,vr=operators(gamma,alpha)
        # Sequential state1..state6 updates are a product of rank-one operators.
        sequential=np.eye(7)
        for i in range(6): sequential=(np.eye(7)+alpha*np.outer(X[i],bb[i]))@sequential
        seqvalue=X@sequential@np.linalg.pinv(X)
        sensitivity.append({'gamma':gamma,'alpha':alpha,'directValueSpectralRadius':float(max(abs(np.linalg.eigvals(vd)))),'residualValueSpectralRadius':float(max(abs(np.linalg.eigvals(vr)))),'sequentialDirectValueSpectralRadius':float(max(abs(np.linalg.eigvals(seqvalue))))})

def eigen(z): return [{'real':float(v.real),'imag':float(v.imag),'abs':float(abs(v))} for v in np.linalg.eigvals(z)]
record={
 'protocolVersion':'2026-10-25-before-first-run-amendment-A',
 'scope':'Six-state Fig.1, direct eq.3 with section-5 epoch sum, residual eq.6/8; not literal eq.7 or Fig.5 timing replication',
 'sourceURL':'http://leemon.com/papers/1995b.pdf',
 'sourceSHA256':'a32e916f75e4ca626c470d24fe0a3e587c14bd10edfae686fc9edf7498831979',
 'sourceVersion':'15 Jul 1995 errata, 9 pages',
 'parameters':{'gamma':GAMMA,'alpha':ALPHA,'initialWeights':W0.tolist(),'features':X.tolist(),'transition':P.tolist(),'rewards':[0]*6,'featureNormalization':'none','epochAggregation':'sum'},
 'checks':{'featureRank':6,'scalarMatrixMaxErrors':scalar_errors,'finiteDifferenceGradientMaxError':grad_error,'residualMSEIncreasingSteps':monotone_checks['residual'],'iterativeMatrixPowerAndValueSpaceAgreement':True,'independentTwoDimensionalReductionAgreement':True},
 'directValueEigenvalues':eigen(VD),'residualValueEigenvalues':eigen(VR),
 'sourceEquationConflict':{'page':4,'printedEquation':7,'mainUsesEquation':3,'status':'suspected sign typo, researcher inference not author-confirmed','singleStateEq3Factor':tabular_eq3,'singleStateLiteralEq7Factor':tabular_eq7,'literalEq7ValueEigenvalues':eigen(literal_eq7)},
 'primaryEndpoints':{name:rows[-1] for name,rows in traces.items()},
 'signChangesObserved':{name:any(r['V1']<0 or r['V6']<0 for r in rows) for name,rows in traces.items()},
 'residualLongHorizon':fast,'sensitivity':sensitivity,
 'interpretation':{'divergenceReplicated':True,'fixedSignInfinityClaimSupportedByThisRun':False,'residualConvergenceSupportedBySpectrum':True,'paperFigure5Replicated':False,'generalNonlinearConvergenceTested':False},
 'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform()},
 'scriptSHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'protocolSHA256':hashlib.sha256((OUT/'protocol.md').read_bytes()).hexdigest()
}
(OUT/'results.json').write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
with (OUT/'traces.csv').open('w',newline='') as f:
    keys=['step','method','evaluation','bellmanMSE','valueRMSE','V1','V6','weightNorm'];wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader()
    for row in traces['direct']+traces['residual']+fast: wr.writerow(row)
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
for name,c in [('direct','#b91c1c'),('residual','#047857')]:
    rows=[r for r in traces[name] if r['step']<=10000]
    axs[0].semilogy([r['step'] for r in rows],[r['valueRMSE'] for r in rows],label=name,color=c)
axs[0].set(title='Same six-state problem',xlabel='Epoch (sum of 6 updates)',ylabel='Value RMSE (true value = 0)');axs[0].legend()
rows=traces['direct']
for key,c in [('V1','#1d4ed8'),('V6','#b45309')]: axs[1].plot([r['step'] for r in rows],[r[key] for r in rows],label=key,color=c)
axs[1].set_yscale('symlog',linthresh=1);axs[1].set(title='Direct: oscillating divergence',xlabel='Epoch',ylabel='Signed state value (symlog)');axs[1].legend()
axs[1].set_yticks([-1e18,-1e12,-1e6,0,1e6,1e12,1e18])
rows=[r for r in traces['residual'] if r['step']>0]
axs[2].loglog([r['step'] for r in rows],[r['valueRMSE'] for r in rows],color='#047857',label='Value RMSE: iterative')
axs[2].loglog([r['step'] for r in rows],[np.sqrt(r['bellmanMSE']) for r in rows],color='#1d4ed8',label='Bellman RMSE: iterative')
axs[2].loglog([r['step'] for r in fast],[r['valueRMSE'] for r in fast],'o--',color='#047857',label='Value RMSE: matrix power')
axs[2].set(title='Residual: distinguish two errors',xlabel='Epoch (log scale)',ylabel='Error (log scale)');axs[2].legend(fontsize=8)
for ax in axs: ax.grid(alpha=.18)
fig.suptitle('Baird (July 1995) Fig.1: independent implementation | gamma=0.99, alpha=0.01',fontsize=12)
fig.savefig(OUT/'baird-six-state.svg');fig.savefig(OUT/'baird-six-state.png',dpi=160);plt.close(fig)
print(json.dumps({'checks':record['checks'],'endpoints':record['primaryEndpoints'],'fast':fast,'sensitivity':sensitivity},ensure_ascii=False))
