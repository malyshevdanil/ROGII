import numpy as np, pandas as pd, glob, os, time, pickle
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings('ignore')

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
SNAP=r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\cache\snap.pkl'
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a
def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

# ---- selector (from pipeline) ----
SEL_N=4840.0; SEL_Z=(136.73000000000016,185.5133333333342)
SEL_BINS={0:'pf_scale_5_hold_0.2',1:'pf_scale_3_hold_0.15',2:'pf_scale_12_hold_0.15',
    3:'pf_scale_5_hold_0.15',4:'pf_scale_5_hold_0.05',5:'pf_scale_12_hold_0.05'}
SEL_GLOBAL='pf_scale_8_hold_0.2'
def sel_variant(n_eval,z_span):
    nb=int(n_eval>SEL_N); zb=int(np.searchsorted(SEL_Z,z_span,side='right')); return SEL_BINS.get(nb+2*zb,SEL_GLOBAL)
def apply_variant(name,pfmap,last_known):
    parts=name.split('_'); scale=float(parts[2]); hold=float(parts[parts.index('hold')+1]) if 'hold' in parts else 0.0
    base=pfmap['pf%g'%scale]
    return (1-hold)*base+hold*last_known

def robfit_surf(md,surf,deg=4):
    x=np.asarray(md,float); y=np.asarray(surf,float)
    x0=x[0]; xs=max(x.max()-x.min(),1e-6); xk=(x-x0)/xs
    if len(x)<deg+2: return y.copy()
    c=np.polyfit(xk,y,deg)
    for _ in range(4):
        r=y-np.polyval(c,xk); sc=np.median(np.abs(r))*1.4826+1e-6
        c=np.polyfit(xk,y,deg,w=1.0/(1.0+(r/(2*sc))**2))
    return np.polyval(c,xk)

# ---- rich features for GBM ----
def well_feats(wid, OFFS=(-20,-10,-5,0,5,10,20)):
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    tw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')).sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    km=hw['TVT_input'].notna()
    if km.sum()<20 or (~km).sum()<20 or len(tw)<10: return None
    last=hw[km].iloc[-1]; last_tvt=float(last['TVT_input']); last_MD=float(last['MD']); last_Z=float(last['Z'])
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    gr=interp_nan(hw['GR'].values.astype(float))
    X=hw['X'].values.astype(float);Y=hw['Y'].values.astype(float);Z=hw['Z'].values.astype(float);MD=hw['MD'].values.astype(float)
    mdd=np.gradient(MD);mdd[mdd==0]=1
    kg=gr[:km.sum()]; ktvt=hw.loc[km,'TVT_input'].values; twk=np.interp(ktvt,tw_tvt,tw_gr); v=np.isfinite(kg)&np.isfinite(twk)
    a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.))
    head=np.arctan2(np.gradient(Y),np.gradient(X)+1e-9)
    F={}
    F['md_since']=MD-last_MD;F['gr']=gr;F['gr_grad']=np.gradient(gr)
    F['gr_rstd']=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values
    F['z']=Z-last_Z;F['dzdmd']=np.gradient(Z)/mdd;F['cal_gr']=gr*a+b
    for o in OFFS: F['tda%d'%o]=gr-np.interp(last_tvt+o,tw_tvt,tw_gr)
    F['sin_azi']=np.sin(head);F['cos_azi']=np.cos(head)
    return F,last_tvt
OWN=['md_since','gr','gr_grad','gr_rstd','z','dzdmd','cal_gr','tda-20','tda-10','tda-5','tda0','tda5','tda10','tda20','sin_azi','cos_azi']

R=pickle.load(open(SNAP,'rb')); wids=[w for w in well_ids if w in R]
print('wells',len(wids),flush=True); t0=time.time()
# assemble per-well: SP45(selector), rich features, cache signals, Z at eval
DATA={}
for wid in wids:
    r=well_feats(wid)
    if r is None: continue
    F,lt=r; d=R[wid]; ei=d['ei']
    pfmap={'pf3':d['pf3'],'pf5':d['pf5'],'pf8':d['pf8'],'pf12':d['pf12']}
    n_eval=len(ei);
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'),usecols=['Z','MD','TVT_input','TVT'])
    z_ev=hw['Z'].values[ei]; md_ev=hw['MD'].values[ei]
    z_span=float(np.ptp(z_ev))
    sp45=apply_variant(sel_variant(n_eval,z_span),pfmap,lt)
    own=np.stack([F[c][ei] for c in OWN],1).astype(np.float32)
    DATA[wid]=dict(sp45=sp45.astype(np.float32),own=own,lt=lt,true=d['true'],z=z_ev,md=md_ev,
                   beam=d['beam'],pf8=d['pf8'])
wids=[w for w in wids if w in DATA]
print('assembled %.0fs'%(time.time()-t0),flush=True)

# GBM OOF (rich feats -> delta) : the pipeline's learned component (proxy for fleongg)
gkf=GroupKFold(5); arr=np.array(wids); gbm={}
for tr,va in gkf.split(wids,groups=arr):
    trw=[wids[i] for i in tr]; vaw=[wids[i] for i in va]
    Xtr=np.concatenate([DATA[w]['own'] for w in trw]); ytr=np.concatenate([DATA[w]['true']-DATA[w]['lt'] for w in trw])
    m=lgb.LGBMRegressor(n_estimators=500,num_leaves=63,learning_rate=0.03,subsample=0.8,colsample_bytree=0.7,reg_lambda=5.0,min_child_samples=50,n_jobs=4,verbose=-1)
    m.fit(Xtr,ytr)
    for w in vaw: gbm[w]=(DATA[w]['lt']+m.predict(DATA[w]['own'])).astype(np.float32)
print('GBM OOF %.0fs'%(time.time()-t0),flush=True)

def project(pred_map, deg=4, blend=0.75):
    out={}
    for w in wids:
        d=DATA[w]; surf=pred_map[w]+d['z']; anchor=d['lt']+d['z'][0]
        fit=(robfit_surf(d['md'],surf-anchor,deg)+anchor)-d['z']
        out[w]=((1-blend)*pred_map[w]+blend*fit).astype(np.float32)
    return out

allt=np.concatenate([DATA[w]['true'] for w in wids])
def ev(pm): return prmse(allt,np.concatenate([pm[w] for w in wids]))

sp45={w:DATA[w]['sp45'] for w in wids}
print('\n=== FULL-PIPELINE PROXY (held-out via OOF) ===',flush=True)
print('SP45 (selector)         %.3f'%ev(sp45),flush=True)
blend={w:0.7*DATA[w]['sp45']+0.3*gbm[w] for w in wids}
print('0.7*SP45+0.3*GBM        %.3f'%ev(blend),flush=True)
proj=project(blend,4,0.75)
print('+ projection(deg4,0.75) %.3f  <- PROXY for the real ~7.2 pipeline'%ev(proj),flush=True)
print('\ntotal %.0fs'%(time.time()-t0),flush=True)
# save proxy pieces for follow-up experiments
pickle.dump(dict(DATA=DATA,gbm=gbm,wids=wids),open(r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\proxy.pkl','wb'))
print('saved proxy.pkl',flush=True)
