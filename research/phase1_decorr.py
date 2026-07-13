import numpy as np, pickle, os, itertools

CACHE=r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\cache\cache.pkl'
R=pickle.load(open(CACHE,'rb'))
wids=sorted(R.keys())
print('cached wells:',len(wids))

SIGS=['pf3','pf5','pf8','pf12','pfmean','beam','flat','linear']
def prmse(a,b): a=np.asarray(a);b=np.asarray(b);return float(np.sqrt(np.mean((a-b)**2)))

# pooled standalone RMSE
truth=np.concatenate([R[w]['true'] for w in wids])
print('\n--- standalone pooled RMSE ---')
errs={}
for s in SIGS:
    p=np.concatenate([R[w][s] for w in wids]); errs[s]=truth-p
    print('%-8s %.3f'%(s,prmse(truth,p)))

# error-correlation matrix
print('\n--- error correlation matrix ---')
print('        '+' '.join('%7s'%s for s in SIGS))
for a in SIGS:
    row=[]
    for b in SIGS:
        row.append(np.corrcoef(errs[a],errs[b])[0,1])
    print('%-8s'%a+' '.join('%7.3f'%v for v in row))

# best 2-way blends (equal weight) vs best standalone
print('\n--- best equal-weight 2-blends (pooled RMSE) ---')
res=[]
for a,b in itertools.combinations(SIGS,2):
    pa=np.concatenate([R[w][a] for w in wids]); pb=np.concatenate([R[w][b] for w in wids])
    for wA in [0.3,0.4,0.5,0.6,0.7]:
        res.append((prmse(truth,wA*pa+(1-wA)*pb),a,b,wA))
res.sort()
best_solo=min(prmse(truth,np.concatenate([R[w][s] for w in wids])) for s in SIGS)
print('best standalone: %.3f'%best_solo)
for rmse,a,b,wA in res[:12]:
    print('%.3f  %.1f*%s + %.1f*%s'%(rmse,wA,a,1-wA,b))
