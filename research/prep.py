"""Shared research harness: cache per-well features + typewell grid + whole-well holdout + baselines.
North star = pooled RMSE on eval-zone TVT over HELD-OUT wells (mimics the private test).
All model experiments import from here so every attempt is ranked on the SAME trustworthy holdout."""
import numpy as np, pandas as pd, glob, os, time, pickle
TRAIN_DIR='data/train'
CACHE=os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache_feat.pkl')
STRIDE=4          # downsample horizontal sequence
TWGRID=256        # typewell resampled to fixed grid (GR as fn of TVT)
OFFS=[-20,-12,-6,-3,0,3,6,12,20]   # typewell-difference probe offsets around last_tvt
FEATS=['gr','gr_sm','gr_grad','gr_rstd','cal_gr','z_rel','dzdmd','md_since','tvtin_rel','mask_kn']+['tda%d'%o for o in OFFS]
NF=len(FEATS)
GR_CH=[0,1,4]+list(range(10,NF))   # GR-derived channels (for calibration-jitter augmentation)

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a

def build(wid):
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    tw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')).sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tw_tvt)<10: return None
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20 or hw['TVT_input'].isna().sum()<20: return None
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input']); last_Z=float(last['Z']); last_MD=float(last['MD'])
    n=len(hw); gr=interp_nan(hw['GR'].values.astype(float))
    gr_sm=pd.Series(gr).rolling(11,center=True,min_periods=1).mean().values
    gr_rstd=pd.Series(gr).rolling(21,center=True,min_periods=1).std().fillna(0).values
    kn_gr=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); v=np.isfinite(kn_gr)&np.isfinite(tw_at_k)
    a,b=(np.polyfit(kn_gr[v],tw_at_k[v],1) if v.sum()>=20 else (1.,0.))
    cal_gr=gr*a+b
    Z=hw['Z'].values.astype(float); MD=hw['MD'].values.astype(float)
    mdd=np.gradient(MD); mdd[mdd==0]=1
    feat=np.zeros((n,NF),np.float32)
    feat[:,0]=gr; feat[:,1]=gr_sm; feat[:,2]=np.gradient(gr); feat[:,3]=gr_rstd; feat[:,4]=cal_gr
    feat[:,5]=Z-last_Z; feat[:,6]=np.gradient(Z)/mdd; feat[:,7]=(MD-last_MD)/1000.
    feat[:,8]=np.nan_to_num(hw['TVT_input'].values.astype(float)-last_tvt,nan=0.0)
    feat[:,9]=hw['TVT_input'].notna().values.astype(float)
    for j,o in enumerate(OFFS): feat[:,10+j]=cal_gr-np.interp(last_tvt+o,tw_tvt,tw_gr)
    target=hw['TVT'].values.astype(float)-last_tvt     # predict TVT delta from last known
    ev=hw['TVT_input'].isna().values.astype(np.float32)
    # typewell resampled to fixed grid over its range, plus where last_tvt sits in it (for alignment models)
    g_tvt=np.linspace(tw_tvt.min(),tw_tvt.max(),TWGRID)
    g_gr=np.interp(g_tvt,tw_tvt,tw_gr).astype(np.float32)
    # normalize typewell GR by calibration so it matches cal_gr scale
    sel=np.arange(0,n,STRIDE)
    return dict(wid=wid,feat=feat[sel],target=target[sel].astype(np.float32),ev=ev[sel],
                last_tvt=last_tvt,tw_min=float(tw_tvt.min()),tw_max=float(tw_tvt.max()),
                g_gr=g_gr,g_tvt=g_tvt.astype(np.float32),cal_a=float(a),cal_b=float(b))

def load(rebuild=False):
    if os.path.exists(CACHE) and not rebuild:
        return pickle.load(open(CACHE,'rb'))
    wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    t0=time.time(); data=[]
    for i,w in enumerate(wids):
        b=build(w)
        if b is not None: data.append(b)
        if (i+1)%150==0: print('  built %d/%d %.0fs'%(i+1,len(wids),time.time()-t0),flush=True)
    pickle.dump(data,open(CACHE,'wb'))
    print('cached %d wells %.0fs'%(len(data),time.time()-t0),flush=True)
    return data

def split(data, n_val=160, seed=42):
    rng=np.random.default_rng(seed); idx=np.arange(len(data)); rng.shuffle(idx)
    val=set(idx[:n_val].tolist())
    tr=[data[i] for i in idx[n_val:]]; va=[data[i] for i in idx[:n_val]]
    return tr,va

def pooled_rmse_eval(preds, wells):
    """preds: list of per-well predicted TVT-delta arrays (same order as wells). Pool eval points only."""
    e2=[];
    for p,w in zip(preds,wells):
        m=w['ev']>0.5
        e2.append((p[m]-w['target'][m])**2)
    e2=np.concatenate(e2); return float(np.sqrt(e2.mean())), int(len(e2))

if __name__=='__main__':
    data=load(rebuild=True)
    tr,va=split(data)
    print('train wells',len(tr),'val wells',len(va),flush=True)
    # ---- baselines on val holdout (eval-zone pooled RMSE) ----
    # flat: predict delta=0 (TVT=last_tvt)
    flat=[np.zeros(len(w['target'])) for w in va]
    r,n=pooled_rmse_eval(flat,va); print('BASELINE flat (delta=0):        pooled RMSE %.3f  (%d eval pts)'%(r,n),flush=True)
    # linear: slope from known tail extrapolated in TVT-delta vs md_since
    lin=[]
    for w in va:
        f=w['feat']; kn=w['feat'][:,9]>0.5; md=f[:,7]  # md_since/1000
        y=w['target']
        if kn.sum()>=10:
            A=np.polyfit(md[kn],y[kn],1); pred=np.polyval(A,md)
        else: pred=np.zeros(len(y))
        lin.append(pred)
    r,n=pooled_rmse_eval(lin,va); print('BASELINE linear tail slope:     pooled RMSE %.3f'%r,flush=True)
    print('\n(reference scale: LB flat baseline=15.883; PF isolated~11 local; pipeline LB=7.096.)',flush=True)
    print('NF=%d channels, %d wells cached, stride=%d'%(NF,len(data),STRIDE),flush=True)
