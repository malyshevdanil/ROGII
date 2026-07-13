import numpy as np, pickle, os

CACHE=r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\cache\cache.pkl'
R=pickle.load(open(CACHE,'rb')); wids=sorted(R.keys())
print('cached wells:',len(wids))
def prmse(a,b): a=np.asarray(a);b=np.asarray(b);return float(np.sqrt(np.mean((a-b)**2)))
truth=np.concatenate([R[w]['true'] for w in wids])

# base signal: pf8 (best standalone). test fixed hold-toward-flat weights.
print('\n--- fixed hold weight (blend pf8 with flat), pooled over all wells ---')
for h in [0.0,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
    p=np.concatenate([(1-h)*R[w]['pf8']+h*R[w]['flat'] for w in wids])
    print('hold=%.2f  RMSE=%.3f'%(h,prmse(truth,p)))

# per-well ADAPTIVE hold using seed_std (high uncertainty -> more hold toward flat)
print('\n--- adaptive hold = clip(a*mean_seed_std + b) per well ---')
# normalize seed_std per well by its mean; wells with high avg seed_std get more hold
avg_std=np.array([R[w]['seed_std'].mean() for w in wids])
print('avg seed_std: p10=%.1f p50=%.1f p90=%.1f'%(np.percentile(avg_std,10),np.percentile(avg_std,50),np.percentile(avg_std,90)))
best=(1e9,None)
for lo in [0.0,0.1,0.15]:
    for hi in [0.3,0.4,0.5]:
        # map avg_std percentile-rank -> hold in [lo,hi]
        rank=np.argsort(np.argsort(avg_std))/(len(avg_std)-1)
        holds=lo+(hi-lo)*rank
        p=np.concatenate([(1-holds[i])*R[w]['pf8']+holds[i]*R[w]['flat'] for i,w in enumerate(wids)])
        r=prmse(truth,p)
        if r<best[0]: best=(r,(lo,hi))
        print('adaptive lo=%.2f hi=%.2f  RMSE=%.3f'%(lo,hi,r))
print('best adaptive:',best)

# also try per-position hold: more hold where seed_std is locally high
print('\n--- per-position hold via local seed_std ---')
for k in [0.0,0.01,0.02,0.03]:
    parts=[]
    for w in wids:
        s=R[w]['seed_std']; h=np.clip(k*s,0,0.6)
        parts.append((1-h)*R[w]['pf8']+h*R[w]['flat'])
    print('k=%.2f  RMSE=%.3f'%(k,prmse(truth,np.concatenate(parts))))
