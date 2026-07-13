"""Re-test the ONLY promising op (shrink-to-flat / tail-robustify) on the FULL-PIPELINE proxy (sp45),
which correlates with LB far better than the isolated PF. If it still helps here, it may transfer."""
import pickle, numpy as np
p=pickle.load(open('proxy.pkl','rb')); D=p['DATA']; wids=list(D.keys())
rng=np.random.default_rng(42); idx=np.arange(len(wids)); rng.shuffle(idx)
TRK=[wids[i] for i in idx[160:]]; VAK=[wids[i] for i in idx[:160]]
sl=[VAK[:54],VAK[54:108],VAK[108:]]
def pooled(keys,fn):
    s=[]
    for k in keys:
        w=D[k]; s.append((np.asarray(fn(w),float)-w['true'])**2)
    return float(np.sqrt(np.mean(np.concatenate(s))))
def flat(w): return np.full_like(w['true'],w['lt'])
def report(name,fn):
    print('%-34s VAL %.4f | A %.3f B %.3f C %.3f'%(name,pooled(VAK,fn),
        pooled(sl[0],fn),pooled(sl[1],fn),pooled(sl[2],fn)))

print('=== proxy references (full pipeline) ===')
report('flat (lt)',flat)
report('sp45 (pipeline core)',lambda w:w['sp45'])
report('beam',lambda w:w['beam'])
report('pf8',lambda w:w['pf8'])
report('mean(sp45,beam)',lambda w:(w['sp45']+w['beam'])/2)

print('\n=== T1 on sp45: global shrink sp45 -> flat (fit TRAIN) ===')
best=(99,0)
for a in [0,0.02,0.05,0.08,0.1,0.15,0.2,0.3]:
    tr=pooled(TRK,lambda w,a=a:(1-a)*w['sp45']+a*w['lt'])
    if tr<best[0]: best=(tr,a)
print('  best a TRAIN=%.2f (%.4f)'%(best[1],best[0]))
report('  apply a=%.2f'%best[1],lambda w,a=best[1]:(1-a)*w['sp45']+a*w['lt'])

print('\n=== T5 on sp45: tail cap |sp45-flat| (fit TRAIN) ===')
best=(99,0)
for c in [10,20,30,50,80,120,200]:
    tr=pooled(TRK,lambda w,c=c:w['lt']+np.clip(w['sp45']-w['lt'],-c,c))
    if tr<best[0]: best=(tr,c)
print('  best cap TRAIN=%d (%.4f)'%(best[1],best[0]))
report('  apply cap=%d'%best[1],lambda w,c=best[1]:w['lt']+np.clip(w['sp45']-w['lt'],-c,c))

print('\n=== T6: robust blend sp45 with beam (decorrelated within pipeline) ===')
best=(99,0)
for a in [0,0.1,0.2,0.3,0.4,0.5]:
    tr=pooled(TRK,lambda w,a=a:(1-a)*w['sp45']+a*w['beam'])
    if tr<best[0]: best=(tr,a)
print('  best a TRAIN=%.2f (%.4f)'%(best[1],best[0]))
report('  apply a=%.2f'%best[1],lambda w,a=best[1]:(1-a)*w['sp45']+a*w['beam'])

print('\n=== T7: does sp45 beat flat per-well? (tail diagnosis) ===')
worse=0; tot=0; big=0
for k in VAK:
    w=D[k]; e_sp=np.sqrt(np.mean((w['sp45']-w['true'])**2)); e_fl=np.sqrt(np.mean((flat(w)-w['true'])**2))
    tot+=1; worse+= e_sp>e_fl; big+= (e_sp>e_fl+5)
print('  wells where sp45 worse than flat: %d/%d (badly worse +5ft: %d)'%(worse,tot,big))
