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
    return F,last_tvt,hw['TVT'].values

FEATCOLS=['md_since','gr','gr_grad','gr_rstd','gr_rm51','z','dzdmd','cal_gr',
          'tda-20','tda-10','tda-5','tda0','tda5','tda10','tda20',
          'sin_azi','cos_azi','signed_azi','incl','dxdmd','dydmd']

R=pickle.load(open(SNAP,'rb')); wids=[w for w in well_ids if w in R]
print('cached wells:',len(wids),flush=True)

# build full-res features at cache ei for every well
t0=time.time(); WF={}
for i,wid in enumerate(wids):
    r=well_feats(wid)
    if r is None: continue
    F,last_tvt,tvt=r; ei=R[wid]['ei']
    M=np.stack([F[c][ei] for c in FEATCOLS],1).astype(np.float32)
    WF[wid]=(M,last_tvt)
    if (i+1)%200==0: print('feat %d/%d %.0fs'%(i+1,len(wids),time.time()-t0),flush=True)
wids=[w for w in wids if w in WF]
print('feats built for',len(wids),'%.0fs'%(time.time()-t0),flush=True)

# GroupKFold OOF GBM predictions (delta -> tvt)
gkf=GroupKFold(5); wid_arr=np.array(wids)
oof_gbm={}
groups=np.arange(len(wids))
for fi,(tr,va) in enumerate(gkf.split(wids,groups=wid_arr)):
    trw=[wids[i] for i in tr]; vaw=[wids[i] for i in va]
    Xtr=np.concatenate([WF[w][0] for w in trw]); ytr=np.concatenate([R[w]['true']-WF[w][1] for w in trw])
    m=lgb.LGBMRegressor(n_estimators=600,num_leaves=63,learning_rate=0.03,subsample=0.8,
        colsample_bytree=0.7,reg_lambda=5.0,min_child_samples=50,n_jobs=4,verbose=-1)
    m.fit(Xtr,ytr)
    for w in vaw: oof_gbm[w]=(WF[w][1]+m.predict(WF[w][0])).astype(np.float32)
    print('fold %d done %.0fs'%(fi+1,time.time()-t0),flush=True)

def prmse(a,b):a=np.asarray(a);b=np.asarray(b);return float(np.sqrt(np.mean((a-b)**2)))
allt=np.concatenate([R[w]['true'] for w in wids])
pf=np.concatenate([R[w]['pf8'] for w in wids])
gb=np.concatenate([oof_gbm[w] for w in wids])
fl=np.concatenate([R[w]['flat'] for w in wids])
bm=np.concatenate([R[w]['beam'] for w in wids])
print('\n=== FULL 773-well OOF results ===',flush=True)
print('PF8    RMSE=%.3f'%prmse(allt,pf),flush=True)
print('GBM    RMSE=%.3f'%prmse(allt,gb),flush=True)
print('flat   RMSE=%.3f'%prmse(allt,fl),flush=True)
print('beam   RMSE=%.3f'%prmse(allt,bm),flush=True)
print('corr(PF err, GBM err)=%.3f'%np.corrcoef(allt-pf,allt-gb)[0,1],flush=True)
print('\n--- PF+GBM blend ---',flush=True)
for w in [0.4,0.5,0.6,0.7,0.8]:
    print('%.1f*PF+%.1f*GBM  RMSE=%.3f'%(w,1-w,prmse(allt,w*pf+(1-w)*gb)),flush=True)
best=(1e9,None)
for wp in np.arange(0.2,0.9,0.05):
    for wg in np.arange(0.0,1-wp+1e-9,0.05):
        wf=1-wp-wg
        if wf<-1e-9: continue
        r=prmse(allt,wp*pf+wg*gb+wf*fl)
        if r<best[0]: best=(r,(round(wp,2),round(wg,2),round(wf,2)))
print('\nbest 3-way PF/GBM/flat:',best,flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
