"""Variance-reduction / tail-robustification theories on the 773-well cache.
Honest split: fit coefficients on TRAIN (613), report on VAL (160), and 3 slices to catch mirages.
Only variance-reduction-type ops are candidates (they transfer; new-signal ops don't -- GBM lesson)."""
import pickle, numpy as np
d=pickle.load(open('cache/snap.pkl','rb'))
wids=list(d.keys())
rng=np.random.default_rng(42); idx=np.arange(len(wids)); rng.shuffle(idx)
VAL=set(wids[i] for i in idx[:160]); TRK=[wids[i] for i in idx[160:]]
VAK=[w for w in wids if w in VAL]

def pooled(keys, fn):
    s=[]
    for k in keys:
        w=d[k]; p=np.asarray(fn(w),float); s.append((p-w['true'])**2)
    return float(np.sqrt(np.mean(np.concatenate(s))))

# 3 slices over VAL ordering to detect slice mirages
sl=[VAK[:54], VAK[54:108], VAK[108:]]
def report(name, fn):
    v=pooled(VAK,fn); a=pooled(sl[0],fn); b=pooled(sl[1],fn); c=pooled(sl[2],fn)
    print('%-38s VAL %.4f | A %.3f B %.3f C %.3f'%(name,v,a,b,c))
    return v

base=lambda w:w['pf12']
print('=== references (VAL) ===')
report('flat', lambda w:w['flat'])
report('pf12 (incumbent central)', base)
report('mean(pf8,pf12)', lambda w:(w['pf8']+w['pf12'])/2)

print('\n=== T1: global shrink pf12 -> flat, sweep a (fit on TRAIN) ===')
best=(99,0)
for a in [0,0.02,0.05,0.08,0.1,0.15,0.2,0.3]:
    tr=pooled(TRK, lambda w:(1-a)*w['pf12']+a*w['flat'])
    if tr<best[0]: best=(tr,a)
print('  best a on TRAIN = %.2f (train %.4f)'%(best[1],best[0]))
report('  T1 apply a=%.2f on VAL'%best[1], lambda w,a=best[1]:(1-a)*w['pf12']+a*w['flat'])

print('\n=== T2: uncertainty-gated shrink to flat  a_i=clip(ss/s0,0,amax) ===')
allss=np.concatenate([d[k]['seed_std'] for k in TRK]); s0=np.median(allss)
def t2(w,scale,amax):
    a=np.clip(w['seed_std']/(scale),0,amax); return (1-a)*w['pf12']+a*w['flat']
bt=(99,None)
for scale in [s0*2,s0*3,s0*4]:
    for amax in [0.15,0.3,0.5]:
        tr=pooled(TRK,lambda w,s=scale,m=amax:t2(w,s,m))
        if tr<bt[0]: bt=(tr,(scale,amax))
print('  best (scale,amax) on TRAIN %.4f'%bt[0], (round(bt[1][0],2),bt[1][1]))
report('  T2 apply on VAL', lambda w:t2(w,*bt[1]))

print('\n=== T3: winsorize deviation from flat, gated by per-well spread (k*MAD) ===')
def t3(w,k):
    dev=w['pf12']-w['flat']; s=np.median(np.abs(dev-np.median(dev)))+1e-6
    cap=k*s; return w['flat']+np.clip(dev,-cap,cap)
bt3=(99,0)
for k in [1.5,2,3,4,6,10]:
    tr=pooled(TRK,lambda w,k=k:t3(w,k))
    if tr<bt3[0]: bt3=(tr,k)
print('  best k on TRAIN %.4f k=%.1f'%(bt3[0],bt3[1]))
report('  T3 apply on VAL', lambda w:t3(w,bt3[1]))

print('\n=== T4: optimal non-neg convex weights over {pf3,pf5,pf8,pf12,flat} (fit TRAIN) ===')
from numpy.linalg import lstsq
keys=['pf3','pf5','pf8','pf12','flat']
def stackX(klist):
    X=[];Y=[]
    for k in klist:
        w=d[k]; X.append(np.stack([w[q] for q in keys],1)); Y.append(w['true'])
    return np.concatenate(X),np.concatenate(Y)
Xtr,Ytr=stackX(TRK)
Xtr=Xtr[::20]; Ytr=Ytr[::20]                     # subsample rows: 5-weight fit needs little data
# simple projected gradient for nonneg weights summing to 1
wv=np.ones(len(keys))/len(keys); L=np.linalg.norm(Xtr,2)**2/len(Ytr)
for _ in range(3000):
    g=Xtr.T@(Xtr@wv-Ytr)/len(Ytr); wv=wv-(0.9/L)*g
    wv=np.clip(wv,0,None); wv=wv/wv.sum()
print('  weights',dict(zip(keys,np.round(wv,3))))
report('  T4 apply on VAL', lambda w:sum(wv[i]*w[k] for i,k in enumerate(keys)))

print('\n=== T5: per-sample tail cap on |pf-flat| absolute (ft), sweep ===')
bt5=(99,0)
for cap in [10,20,30,50,80,120]:
    tr=pooled(TRK,lambda w,c=cap:w['flat']+np.clip(w['pf12']-w['flat'],-c,c))
    if tr<bt5[0]: bt5=(tr,cap)
print('  best cap TRAIN %.4f cap=%d'%(bt5[0],bt5[1]))
report('  T5 apply on VAL', lambda w:w['flat']+np.clip(w['pf12']-w['flat'],-bt5[1],bt5[1]))
