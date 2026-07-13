import numpy as np, pandas as pd, glob, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a

def q3d_tortuosity(X,Y,Z,MD, win=100):
    # simplified Q-3D: cumulative high-frequency undulation in vertical (TVD) & lateral planes
    n=len(MD)
    horiz=np.concatenate([[0],np.cumsum(np.sqrt(np.diff(X)**2+np.diff(Y)**2))])  # horizontal distance
    # vertical plane: TVD (=-Z or Z) undulation vs horiz distance -> detrended local curvature
    def plane_tort(u, s):
        # u=coordinate, s=arc; rolling: ratio of path length to chord + mean deflection
        du=np.gradient(u); ds=np.gradient(s); ds[ds==0]=1e-6
        slope=du/ds
        dslope=np.abs(np.gradient(slope))     # local bending
        # rolling cumulative bending (tortuosity proxy)
        t_incl=pd.Series(dslope).rolling(win,center=True,min_periods=1).sum().values
        return t_incl
    lat=np.concatenate([[0],np.cumsum(np.abs(np.diff(X)*np.sign(1)+ 1j*0).real)])  # placeholder lateral
    # lateral wander: perpendicular offset from smoothed heading
    ang=np.unwrap(np.arctan2(np.gradient(Y),np.gradient(X)+1e-9))
    lat_bend=pd.Series(np.abs(np.gradient(ang))).rolling(win,center=True,min_periods=1).sum().values
    t_incl=plane_tort(Z, horiz)
    t_q3d=np.sqrt(t_incl**2 + lat_bend**2)
    return t_incl.astype(np.float32), lat_bend.astype(np.float32), t_q3d.astype(np.float32)

def build(wid, stride=6):
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    if 'TVT' not in hw.columns: return None
    km=hw['TVT_input'].notna()
    if km.sum()<20 or (~km).sum()<20: return None
    last=hw[km].iloc[-1]; last_tvt=float(last['TVT_input']); last_MD=float(last['MD']); last_Z=float(last['Z'])
    n=len(hw)
    gr=interp_nan(hw['GR'].values.astype(float))
    X=hw['X'].values.astype(float);Y=hw['Y'].values.astype(float);Z=hw['Z'].values.astype(float);MD=hw['MD'].values.astype(float)
    ti,la,tq=q3d_tortuosity(X,Y,Z,MD)
    mdd=np.gradient(MD);mdd[mdd==0]=1
    base=dict(
        md_since=(MD-last_MD).astype(np.float32),
        gr=gr.astype(np.float32), gr_grad=np.gradient(gr).astype(np.float32),
        gr_rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values.astype(np.float32),
        z=(Z-last_Z).astype(np.float32), dzdmd=(np.gradient(Z)/mdd).astype(np.float32),
    )
    tort=dict(tort_incl=ti, tort_lat=la, tort_q3d=tq)
    ev=~km.values; sel=np.arange(0,n,stride); sel=sel[ev[sel]]
    if len(sel)<10: return None
    df=pd.DataFrame({k:v[sel] for k,v in {**base,**tort}.items()})
    df['target']=(hw['TVT'].values[sel]-last_tvt).astype(np.float32)
    df['well']=wid
    return df

print('building...',flush=True); t0=time.time()
parts=[]
for i,wid in enumerate(well_ids):
    d=build(wid)
    if d is not None: parts.append(d)
    if (i+1)%200==0: print('%d/%d %.0fs'%(i+1,len(well_ids),time.time()-t0),flush=True)
df=pd.concat(parts,ignore_index=True)
print('rows',len(df),'wells',df['well'].nunique(),flush=True)

base_feats=['md_since','gr','gr_grad','gr_rstd','z','dzdmd']
tort_feats=['tort_incl','tort_lat','tort_q3d']

def cv_rmse(feats):
    X=df[feats].values; y=df['target'].values; g=df['well'].values
    oof=np.zeros(len(df))
    for tr,va in GroupKFold(5).split(X,y,g):
        m=lgb.LGBMRegressor(n_estimators=400,num_leaves=63,learning_rate=0.03,
            subsample=0.8,colsample_bytree=0.8,reg_lambda=3.0,min_child_samples=40,n_jobs=4,verbose=-1)
        m.fit(X[tr],y[tr]); oof[va]=m.predict(X[va])
    return float(np.sqrt(np.mean((y-oof)**2)))

flat_rmse=float(np.sqrt(np.mean(df['target'].values**2)))
print('\nflat baseline (predict 0)  RMSE=%.3f'%flat_rmse,flush=True)
print('GBM base feats             RMSE=%.3f'%cv_rmse(base_feats),flush=True)
print('GBM base + tortuosity      RMSE=%.3f'%cv_rmse(base_feats+tort_feats),flush=True)
print('%.0fs'%(time.time()-t0),flush=True)
