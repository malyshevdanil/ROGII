import numpy as np, pandas as pd, pickle, os

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
P=pickle.load(open(r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\proxy.pkl','rb'))
DATA=P['DATA']; gbm=P['gbm']; wids=P['wids']
def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
allt=np.concatenate([DATA[w]['true'] for w in wids])

def robfit(md,y,deg=4):
    x=np.asarray(md,float);x0=x[0];xs=max(x.max()-x.min(),1e-6);xk=(x-x0)/xs
    if len(x)<deg+2:return np.asarray(y,float).copy()
    c=np.polyfit(xk,y,deg)
    for _ in range(4):
        r=y-np.polyval(c,xk);sc=np.median(np.abs(r))*1.4826+1e-6;c=np.polyfit(xk,y,deg,w=1.0/(1.0+(r/(2*sc))**2))
    return np.polyval(c,xk)

# build proxy final prediction per well + typewell TVT range
print('loading typewell ranges...',flush=True)
twrange={}
for w in wids:
    tw=pd.read_csv(os.path.join(TRAIN_DIR,f'{w}__typewell.csv'))
    twrange[w]=(float(tw['TVT'].min()),float(tw['TVT'].max()))

final={}
for w in wids:
    d=DATA[w]; pred=0.7*d['sp45']+0.3*gbm[w]; surf=pred+d['z']; anchor=d['lt']+d['z'][0]
    fit=(robfit(d['md'],surf-anchor,4)+anchor)-d['z']
    final[w]=0.25*pred+0.75*fit

base=prmse(allt,np.concatenate([final[w] for w in wids]))
print('proxy baseline (deg4 proj)         %.4f'%base,flush=True)

# --- Idea 1: clip to typewell TVT range ---
def clip_tw(margin):
    out=[]
    for w in wids:
        lo,hi=twrange[w]; out.append(np.clip(final[w],lo-margin,hi+margin))
    return prmse(allt,np.concatenate(out))
print('\n-- clip to typewell range +/- margin --',flush=True)
for m in [0,5,10,20,50]:
    print('  margin=%3d ft   %.4f'%(m,clip_tw(m)),flush=True)

# --- Idea 2: clip delta from last_tvt to physical bound ---
def clip_delta(bound):
    out=[]
    for w in wids:
        d=final[w]-DATA[w]['lt']; out.append(DATA[w]['lt']+np.clip(d,-bound,bound))
    return prmse(allt,np.concatenate(out))
print('\n-- clip |pred - last_tvt| to bound --',flush=True)
for bnd in [40,60,80,100,150,1e9]:
    print('  bound=%5.0f ft  %.4f'%(bnd,clip_delta(bnd)),flush=True)

# --- Idea 3: clip delta to per-well known-zone-scaled bound ---
# use eval-zone span heuristic: allow |delta| up to k * (typewell span) ... test k
def clip_perwell(k):
    out=[]
    for w in wids:
        lo,hi=twrange[w]; span=hi-lo
        d=final[w]-DATA[w]['lt']; out.append(DATA[w]['lt']+np.clip(d,-k*span,k*span))
    return prmse(allt,np.concatenate(out))
print('\n-- clip |delta| to k*typewell_span --',flush=True)
for k in [0.1,0.15,0.2,0.3,0.5]:
    print('  k=%.2f   %.4f'%(k,clip_perwell(k)),flush=True)
