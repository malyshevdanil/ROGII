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
    F['sin_azi']=np.sin(head);F['cos_azi']=np.cos(head)
    ei=np.where(~km.values)[0]
    M=np.stack([F[c] for c in ['md_since','gr','gr_grad','gr_rstd','z','dzdmd','cal_gr','tda-20','tda-10','tda-5','tda0','tda5','tda10','tda20','sin_azi','cos_azi']],1)[ei].astype(np.float32)
    tvt=hw['TVT'].values.astype(float)[ei]
    dZ=(Z-last_Z)[ei]                       # known
    tvt_delta=(tvt-last_tvt).astype(np.float32)
    surf_delta=((tvt+Z[ei])-(last_tvt+last_Z)).astype(np.float32)   # S - S_heel = tvt_delta + dZ
    return M, tvt_delta, surf_delta, dZ.astype(np.float32), last_tvt

print('building...',flush=True); t0=time.time()
WF={}
for i,wid in enumerate(well_ids):
    r=well_feats(wid)
    if r is not None: WF[wid]=r
    if (i+1)%200==0: print('%d %.0fs'%(i+1,time.time()-t0),flush=True)
wids=list(WF.keys()); print('wells',len(wids),flush=True)

def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
allt=np.concatenate([WF[w][1] for w in wids])  # tvt_delta true (target space for eval = TVT itself via +last_tvt)
alltvt=np.concatenate([WF[w][1]+WF[w][4] for w in wids])  # true TVT

def oof(target_kind):
    gkf=GroupKFold(5); arr=np.array(wids); pred_tvt={}
    for tr,va in gkf.split(wids,groups=arr):
        trw=[wids[i] for i in tr]; vaw=[wids[i] for i in va]
        Xtr=np.concatenate([WF[w][0] for w in trw])
        if target_kind=='tvt':
            ytr=np.concatenate([WF[w][1] for w in trw])
        else:  # surface
            ytr=np.concatenate([WF[w][2] for w in trw])
        m=lgb.LGBMRegressor(n_estimators=600,num_leaves=63,learning_rate=0.03,subsample=0.8,colsample_bytree=0.7,reg_lambda=5.0,min_child_samples=50,n_jobs=4,verbose=-1)
        m.fit(Xtr,ytr)
        for w in vaw:
            p=m.predict(WF[w][0]); lt=WF[w][4]
            if target_kind=='tvt':
                pred_tvt[w]=lt+p
            else:  # p = surf_delta_pred; TVT = last_tvt + surf_delta - dZ
                pred_tvt[w]=lt+p-WF[w][3]
    return np.concatenate([pred_tvt[w] for w in wids])

print('\nflat (predict last_tvt)  %.3f'%prmse(alltvt,np.concatenate([np.full(len(WF[w][1]),WF[w][4]) for w in wids])),flush=True)
p_tvt=oof('tvt'); print('GBM target=TVT-delta      %.3f'%prmse(alltvt,p_tvt),flush=True)
p_surf=oof('surface'); print('GBM target=SURFACE-delta  %.3f  (predict TVT+Z, reconstruct)'%prmse(alltvt,p_surf),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
