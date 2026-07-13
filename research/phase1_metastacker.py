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
    F['z']=Z-last_Z;F['dzdmd']=np.gradient(Z)/mdd;F['cal_gr']=gr*a+b
    for o in OFFS: F['tda%d'%o]=gr-np.interp(last_tvt+o,tw_tvt,tw_gr)
    F['sin_azi']=np.sin(head);F['cos_azi']=np.cos(head);F['signed_azi']=np.sign(np.sin(head))*np.abs(head)
    F['incl']=np.degrees(np.arctan2(np.hypot(np.gradient(X),np.gradient(Y)),np.abs(np.gradient(Z))+1e-9))
    F['dxdmd']=np.gradient(X)/mdd;F['dydmd']=np.gradient(Y)/mdd
    return F,last_tvt,hw['TVT'].values

OWNFEATS=['md_since','gr','gr_grad','gr_rstd','z','dzdmd','cal_gr',
          'tda-20','tda-10','tda-5','tda0','tda5','tda10','tda20',
          'sin_azi','cos_azi','signed_azi','incl','dxdmd','dydmd']

R=pickle.load(open(SNAP,'rb')); wids=[w for w in well_ids if w in R]
print('wells',len(wids),flush=True)
t0=time.time()
# build per-well design matrices: pipeline signals (as delta from last) + own feats ; target delta
WF={}
for i,wid in enumerate(wids):
    r=well_feats(wid)
    if r is None: continue
    F,last_tvt,tvt=r; d=R[wid]; ei=d['ei']
    own=np.stack([F[c][ei] for c in OWNFEATS],1).astype(np.float32)
    # pipeline signals as features (delta form)
    sig=np.stack([d['pf3']-last_tvt,d['pf8']-last_tvt,d['pf12']-last_tvt,d['pfmean']-last_tvt,
                  d['beam']-last_tvt,d['linear']-last_tvt,d['seed_std']],1).astype(np.float32)
    WF[wid]=(own,sig,last_tvt,d['true'],d['pf8'])
    if (i+1)%200==0: print('feat %d %.0fs'%(i+1,time.time()-t0),flush=True)
wids=[w for w in wids if w in WF]

def oof(colset):
    gkf=GroupKFold(5); arr=np.array(wids); pred={}
    for tr,va in gkf.split(wids,groups=arr):
        trw=[wids[i] for i in tr]; vaw=[wids[i] for i in va]
        Xtr=np.concatenate([colset(WF[w]) for w in trw]); ytr=np.concatenate([WF[w][3]-WF[w][2] for w in trw])
        m=lgb.LGBMRegressor(n_estimators=700,num_leaves=63,learning_rate=0.03,subsample=0.8,
            colsample_bytree=0.7,reg_lambda=5.0,min_child_samples=50,n_jobs=4,verbose=-1)
        m.fit(Xtr,ytr)
        for w in vaw: pred[w]=(WF[w][2]+m.predict(colset(WF[w]))).astype(np.float32)
    return np.concatenate([pred[w] for w in wids])

def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
allt=np.concatenate([WF[w][3] for w in wids]); pf=np.concatenate([WF[w][4] for w in wids])
print('\nPF8 baseline RMSE=%.3f'%prmse(allt,pf),flush=True)
# meta-stacker: pipeline signals only
sig_only=oof(lambda t:t[1]); print('meta[signals only]      RMSE=%.3f'%prmse(allt,sig_only),flush=True)
# meta: signals + own feats (incl azimuth)
both=oof(lambda t:np.concatenate([t[1],t[0]],1)); print('meta[signals + own+azi] RMSE=%.3f'%prmse(allt,both),flush=True)
# ablation: signals + own WITHOUT azimuth (drop last 6-ish azi cols: sin,cos,signed,incl,dx,dy -> indices 14..19)
noazi_idx=[i for i,c in enumerate(OWNFEATS) if c not in ('sin_azi','cos_azi','signed_azi')]
noazi=oof(lambda t:np.concatenate([t[1],t[0][:,noazi_idx]],1)); print('meta[signals+own NO azi]RMSE=%.3f'%prmse(allt,noazi),flush=True)
# blend meta with pf
for w in [0.5,0.6,0.7]:
    print('%.1f*meta+%.1f*PF RMSE=%.3f'%(w,1-w,prmse(allt,w*both+(1-w)*pf)),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
