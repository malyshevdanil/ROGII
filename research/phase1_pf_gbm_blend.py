import numpy as np, pandas as pd, glob, os, time, pickle
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
CACHE=r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\cache\cache.pkl'
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a
def tortuosity(X,Y,Z,MD,win=100):
    horiz=np.concatenate([[0],np.cumsum(np.sqrt(np.diff(X)**2+np.diff(Y)**2))]); ds=np.gradient(horiz);ds[ds==0]=1e-6
    v_bend=np.abs(np.gradient(np.gradient(Z)/ds)); head=np.unwrap(np.arctan2(np.gradient(Y),np.gradient(X)+1e-9))
    return None

# feature builder returning arrays for the WHOLE well; caller selects indices
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
    F['md_since']=MD-last_MD; F['gr']=gr; F['gr_grad']=np.gradient(gr)
    F['gr_rstd']=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values
    F['gr_rm51']=pd.Series(gr).rolling(51,center=True,min_periods=1).mean().values
    F['z']=Z-last_Z; F['dzdmd']=np.gradient(Z)/mdd
    F['cal_gr']=gr*a+b
    for o in OFFS: F['tda%d'%o]=gr-np.interp(last_tvt+o,tw_tvt,tw_gr)
    F['sin_azi']=np.sin(head);F['cos_azi']=np.cos(head);F['signed_azi']=np.sign(np.sin(head))*np.abs(head)
    F['incl']=np.degrees(np.arctan2(np.hypot(np.gradient(X),np.gradient(Y)),np.abs(np.gradient(Z))+1e-9))
    F['dxdmd']=np.gradient(X)/mdd;F['dydmd']=np.gradient(Y)/mdd
    return F, last_tvt, hw['TVT'].values, (~km.values)

FEATCOLS=['md_since','gr','gr_grad','gr_rstd','gr_rm51','z','dzdmd','cal_gr',
          'tda-20','tda-10','tda-5','tda0','tda5','tda10','tda20',
          'sin_azi','cos_azi','signed_azi','incl','dxdmd','dydmd']

R=pickle.load(open(CACHE,'rb')); cached=set(R.keys())
print('cached wells:',len(cached),flush=True)
train_wids=[w for w in well_ids if w not in cached]   # GBM trains on NON-cached
print('train wells (non-cached):',len(train_wids),flush=True)

# build training matrix (strided) on non-cached wells
t0=time.time(); Xtr=[];ytr=[]
for i,wid in enumerate(train_wids):
    r=well_feats(wid)
    if r is None: continue
    F,last_tvt,tvt,ev=r
    n=len(tvt); sel=np.arange(0,n,6); sel=sel[ev[sel]]
    if len(sel)<10: continue
    M=np.stack([F[c][sel] for c in FEATCOLS],1).astype(np.float32)
    Xtr.append(M); ytr.append((tvt[sel]-last_tvt).astype(np.float32))
Xtr=np.concatenate(Xtr);ytr=np.concatenate(ytr)
print('train rows',len(Xtr),'%.0fs'%(time.time()-t0),flush=True)
gbm=lgb.LGBMRegressor(n_estimators=600,num_leaves=63,learning_rate=0.03,subsample=0.8,
    colsample_bytree=0.7,reg_lambda=5.0,min_child_samples=50,n_jobs=4,verbose=-1)
gbm.fit(Xtr,ytr)
print('GBM trained %.0fs'%(time.time()-t0),flush=True)

# evaluate on cached wells: GBM pred at cache ei, blend with cached PF
def prmse(a,b):a=np.asarray(a);b=np.asarray(b);return float(np.sqrt(np.mean((a-b)**2)))
allt=[];pf=[];gb=[];fl=[]
for wid in cached:
    r=well_feats(wid)
    if r is None: continue
    F,last_tvt,tvt,ev=r; d=R[wid]; ei=d['ei']
    M=np.stack([F[c][ei] for c in FEATCOLS],1).astype(np.float32)
    gpred=last_tvt+gbm.predict(M)
    allt.append(d['true']); pf.append(d['pf8']); gb.append(gpred.astype(np.float32)); fl.append(d['flat'])
allt=np.concatenate(allt);pf=np.concatenate(pf);gb=np.concatenate(gb);fl=np.concatenate(fl)
print('\n--- on %d cached wells (GBM never saw them) ---'%len(cached),flush=True)
print('PF8 alone        RMSE=%.3f'%prmse(allt,pf),flush=True)
print('GBM alone        RMSE=%.3f'%prmse(allt,gb),flush=True)
print('flat             RMSE=%.3f'%prmse(allt,fl),flush=True)
print('corr(PF err, GBM err)=%.3f'%np.corrcoef(allt-pf,allt-gb)[0,1],flush=True)
print('\n--- PF + GBM blends ---',flush=True)
for w in [0.3,0.4,0.5,0.6,0.7,0.8]:
    print('%.1f*PF + %.1f*GBM   RMSE=%.3f'%(w,1-w,prmse(allt,w*pf+(1-w)*gb)),flush=True)
# 3-way with flat
best=(1e9,None)
for wp in np.arange(0.3,0.85,0.1):
    for wg in np.arange(0.0,1-wp+0.01,0.1):
        wf=1-wp-wg
        if wf<-1e-9: continue
        r=prmse(allt,wp*pf+wg*gb+wf*fl)
        if r<best[0]: best=(r,(round(wp,1),round(wg,1),round(wf,1)))
print('\nbest 3-way PF/GBM/flat:',best,flush=True)
print('%.0fs'%(time.time()-t0),flush=True)
