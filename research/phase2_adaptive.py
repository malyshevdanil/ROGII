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
    F['gr_rm51']=pd.Series(gr).rolling(51,center=True,min_periods=1).mean().values
    F['z']=Z-last_Z;F['dzdmd']=np.gradient(Z)/mdd;F['cal_gr']=gr*a+b
    for o in OFFS: F['tda%d'%o]=gr-np.interp(last_tvt+o,tw_tvt,tw_gr)
    F['sin_azi']=np.sin(head);F['cos_azi']=np.cos(head);F['signed_azi']=np.sign(np.sin(head))*np.abs(head)
    F['incl']=np.degrees(np.arctan2(np.hypot(np.gradient(X),np.gradient(Y)),np.abs(np.gradient(Z))+1e-9))
    F['dxdmd']=np.gradient(X)/mdd;F['dydmd']=np.gradient(Y)/mdd
    return F,last_tvt

OWN=['md_since','gr','gr_grad','gr_rstd','gr_rm51','z','dzdmd','cal_gr',
     'tda-20','tda-10','tda-5','tda0','tda5','tda10','tda20',
     'sin_azi','cos_azi','signed_azi','incl','dxdmd','dydmd']
R=pickle.load(open(SNAP,'rb')); wids=[w for w in well_ids if w in R]
print('wells',len(wids),flush=True); t0=time.time()
WF={}
for wid in wids:
    r=well_feats(wid)
    if r is None: continue
    F,lt=r; d=R[wid]; ei=d['ei']
    own=np.stack([F[c][ei] for c in OWN],1).astype(np.float32)
    WF[wid]=(own,lt,d)
wids=[w for w in wids if w in WF]
print('feats %.0fs'%(time.time()-t0),flush=True)

# GBM OOF (standalone rich feats -> delta)
gkf=GroupKFold(5); arr=np.array(wids); gbm_oof={}
for tr,va in gkf.split(wids,groups=arr):
    trw=[wids[i] for i in tr]; vaw=[wids[i] for i in va]
    Xtr=np.concatenate([WF[w][0] for w in trw]); ytr=np.concatenate([WF[w][2]['true']-WF[w][1] for w in trw])
    m=lgb.LGBMRegressor(n_estimators=600,num_leaves=63,learning_rate=0.03,subsample=0.8,
        colsample_bytree=0.7,reg_lambda=5.0,min_child_samples=50,n_jobs=4,verbose=-1)
    m.fit(Xtr,ytr)
    for w in vaw: gbm_oof[w]=(WF[w][1]+m.predict(WF[w][0])).astype(np.float32)
print('GBM OOF %.0fs'%(time.time()-t0),flush=True)

def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
allt=np.concatenate([WF[w][2]['true'] for w in wids])
pf=np.concatenate([WF[w][2]['pf8'] for w in wids])
gb=np.concatenate([gbm_oof[w] for w in wids])
sd=np.concatenate([WF[w][2]['seed_std'] for w in wids])
lin=np.concatenate([WF[w][2]['linear'] for w in wids])
fl=np.concatenate([WF[w][2]['flat'] for w in wids])

print('\nPF8=%.3f  GBM=%.3f  flat=%.3f'%(prmse(allt,pf),prmse(allt,gb),prmse(allt,fl)),flush=True)
print('fixed 0.7PF+0.3GBM = %.3f'%prmse(allt,0.7*pf+0.3*gb),flush=True)

# --- Theory A: ADAPTIVE weight by seed_std (more GBM where PF uncertain) ---
print('\n--- adaptive GBM weight = clip(base + k*z(seed_std)) ---',flush=True)
sd_z=(sd-sd.mean())/(sd.std()+1e-6)
best=(1e9,None)
for base in [0.2,0.3,0.4]:
    for k in [0.0,0.05,0.1,0.15,0.2]:
        w=np.clip(base+k*sd_z,0,0.8)
        r=prmse(allt,(1-w)*pf+w*gb)
        if r<best[0]: best=(r,(base,k))
print('best adaptive:',best,'  (base,k)',flush=True)

# --- Theory B: full meta-stacker on [pf-scales, beam, flat, linear, GBM, seed_std, md_since] ---
print('\n--- meta-stacker (learn per-row combination) ---',flush=True)
mds=np.concatenate([WF[w][2]['feat']['md_since'] if 'feat' in WF[w][2] else WF[w][0][:,0] for w in wids])
# assemble stacker matrix per well for OOF
def stack_cols(w):
    d=WF[w][2]; lt=WF[w][1]
    return np.stack([d['pf3']-lt,d['pf8']-lt,d['pf12']-lt,d['pfmean']-lt,d['beam']-lt,
                     d['linear']-lt,gbm_oof[w]-lt,d['seed_std'],WF[w][0][:,0]],1).astype(np.float32)
st_oof={}
for tr,va in gkf.split(wids,groups=arr):
    trw=[wids[i] for i in tr]; vaw=[wids[i] for i in va]
    Xtr=np.concatenate([stack_cols(w) for w in trw]); ytr=np.concatenate([WF[w][2]['true']-WF[w][1] for w in trw])
    m=lgb.LGBMRegressor(n_estimators=500,num_leaves=31,learning_rate=0.03,subsample=0.8,
        colsample_bytree=0.8,reg_lambda=8.0,min_child_samples=80,n_jobs=4,verbose=-1)
    m.fit(Xtr,ytr)
    for w in vaw: st_oof[w]=(WF[w][1]+m.predict(stack_cols(w))).astype(np.float32)
stk=np.concatenate([st_oof[w] for w in wids])
print('meta-stacker RMSE=%.3f'%prmse(allt,stk),flush=True)
for w in [0.4,0.5,0.6]:
    print('%.1f*stack+%.1f*PF = %.3f'%(w,1-w,prmse(allt,w*stk+(1-w)*pf)),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
