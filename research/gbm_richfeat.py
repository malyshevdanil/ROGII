import numpy as np, pandas as pd, glob, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings('ignore')

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a

def tortuosity(X,Y,Z,MD,win=100):
    horiz=np.concatenate([[0],np.cumsum(np.sqrt(np.diff(X)**2+np.diff(Y)**2))])
    ds=np.gradient(horiz); ds[ds==0]=1e-6
    v_slope=np.gradient(Z)/ds
    v_bend=np.abs(np.gradient(v_slope))
    t_incl=pd.Series(v_bend).rolling(win,center=True,min_periods=1).sum().values
    head=np.unwrap(np.arctan2(np.gradient(Y),np.gradient(X)+1e-9))
    lat_bend=pd.Series(np.abs(np.gradient(head))).rolling(win,center=True,min_periods=1).sum().values
    tq=np.sqrt(t_incl**2+lat_bend**2)
    return t_incl.astype(np.float32),lat_bend.astype(np.float32),tq.astype(np.float32)

def build(wid, stride=6, OFFS=(-20,-10,-5,0,5,10,20)):
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    tw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')).sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    km=hw['TVT_input'].notna()
    if km.sum()<20 or (~km).sum()<20 or len(tw)<10: return None
    last=hw[km].iloc[-1]; last_tvt=float(last['TVT_input']); last_MD=float(last['MD']); last_Z=float(last['Z'])
    n=len(hw)
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    gr=interp_nan(hw['GR'].values.astype(float))
    X=hw['X'].values.astype(float);Y=hw['Y'].values.astype(float);Z=hw['Z'].values.astype(float);MD=hw['MD'].values.astype(float)
    mdd=np.gradient(MD);mdd[mdd==0]=1
    # calibration
    kg=interp_nan(hw['GR'].values.astype(float))[:km.sum()]; ktvt=hw.loc[km,'TVT_input'].values
    twk=np.interp(ktvt,tw_tvt,tw_gr); v=np.isfinite(kg)&np.isfinite(twk)
    a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.))
    # groups
    G={}
    G['base']=dict(md_since=(MD-last_MD),gr=gr,gr_grad=np.gradient(gr),
        gr_rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values,
        gr_rm51=pd.Series(gr).rolling(51,center=True,min_periods=1).mean().values,
        z=(Z-last_Z),dzdmd=np.gradient(Z)/mdd)
    ti,la,tq=tortuosity(X,Y,Z,MD)
    G['tort']=dict(tort_incl=ti,tort_lat=la,tort_q3d=tq)
    head=np.arctan2(np.gradient(Y),np.gradient(X)+1e-9)
    G['azi']=dict(sin_azi=np.sin(head),cos_azi=np.cos(head),
        signed_azi=np.sign(np.sin(head))*np.abs(head),
        incl=np.degrees(np.arctan2(np.hypot(np.gradient(X),np.gradient(Y)),np.abs(np.gradient(Z))+1e-9)),
        dxdmd=np.gradient(X)/mdd,dydmd=np.gradient(Y)/mdd)
    cal=gr*a+b
    grm={'cal_gr':cal}
    for o in OFFS: grm['tda%d'%o]=gr-np.interp(last_tvt+o,tw_tvt,tw_gr)
    G['grmatch']=grm
    # self-correlation: rhythmicity of local GR (relates to bimodal ambiguity)
    def local_autocorr(sig,lag,win=101):
        s=pd.Series(sig); m=s.rolling(win,center=True,min_periods=1).mean()
        d=(sig-m.values)
        num=pd.Series(d*np.roll(d,lag)).rolling(win,center=True,min_periods=1).mean().values
        den=pd.Series(d*d).rolling(win,center=True,min_periods=1).mean().values+1e-6
        return num/den
    G['selfcorr']=dict(ac10=local_autocorr(gr,10),ac20=local_autocorr(gr,20),ac40=local_autocorr(gr,40))
    ev=~km.values; sel=np.arange(0,n,stride); sel=sel[ev[sel]]
    if len(sel)<10: return None
    cols={}
    for gname,feats in G.items():
        for fn,arr in feats.items():
            cols['%s__%s'%(gname,fn)]=np.asarray(arr)[sel].astype(np.float32)
    df=pd.DataFrame(cols); df['target']=(hw['TVT'].values[sel]-last_tvt).astype(np.float32); df['well']=wid
    return df

print('building rich features...',flush=True); t0=time.time()
parts=[]
for i,wid in enumerate(well_ids):
    d=build(wid)
    if d is not None: parts.append(d)
    if (i+1)%200==0: print('%d/%d %.0fs'%(i+1,len(well_ids),time.time()-t0),flush=True)
df=pd.concat(parts,ignore_index=True)
allcols=[c for c in df.columns if c not in ('target','well')]
groups={}
for c in allcols:
    g=c.split('__')[0]; groups.setdefault(g,[]).append(c)
print('rows',len(df),'wells',df['well'].nunique(),'| groups:',{k:len(v) for k,v in groups.items()},flush=True)

def cv_rmse(feats):
    X=df[feats].values.astype(np.float32); y=df['target'].values; g=df['well'].values
    oof=np.zeros(len(df))
    for tr,va in GroupKFold(5).split(X,y,g):
        m=lgb.LGBMRegressor(n_estimators=500,num_leaves=63,learning_rate=0.03,subsample=0.8,
            colsample_bytree=0.7,reg_lambda=5.0,min_child_samples=50,n_jobs=4,verbose=-1)
        m.fit(X[tr],y[tr]); oof[va]=m.predict(X[va])
    return float(np.sqrt(np.mean((y-oof)**2)))

flat=float(np.sqrt(np.mean(df['target'].values**2)))
print('\nflat baseline           %.3f'%flat,flush=True)
# cumulative ablation (toolkit-style): add groups one at a time
order=['base','grmatch','tort','azi','selfcorr']
cur=[]
for g in order:
    cur=cur+groups[g]
    print('+%-9s (%2d feats) cumRMSE=%.3f'%(g,len(cur),cv_rmse(cur)),flush=True)
print('%.0fs'%(time.time()-t0),flush=True)
