import numpy as np, pandas as pd, glob, os, time

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a

GRID=np.arange(-20.0,20.01,0.5)

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')).sort_values('TVT'))

# For each well: split KNOWN zone into fit-part (calibrate) and hold-part (localization test),
# so this is a LEGAL diagnostic (no eval-zone truth used). Slide dz over the hold-part.
def localize(wid, mode='value'):
    hw,tw=load(wid)
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<120: return None
    ktvt=kn['TVT_input'].values.astype(float); kgr=interp_nan(kn['GR'].values.astype(float))
    # fit calibration on first 70%, test localization on last 30%
    n=len(kn); cut=int(n*0.7)
    fit_gr=kgr[:cut]; fit_tvt=ktvt[:cut]; ho_gr=kgr[cut:]; ho_tvt=ktvt[cut:]
    if len(ho_gr)<30: return None
    def twgr(t): return np.interp(t,tw_tvt,tw_gr,left=tw_gr[0],right=tw_gr[-1])
    def twgrad(t):
        # typewell GR gradient wrt TVT
        gg=np.gradient(tw_gr,tw_tvt)
        return np.interp(t,tw_tvt,gg,left=gg[0],right=gg[-1])
    # calibrate value on fit part
    tgt=twgr(fit_tvt); v=np.isfinite(fit_gr)&np.isfinite(tgt)
    a,b=(np.polyfit(fit_gr[v],tgt[v],1) if v.sum()>=20 else (1.0,0.0))
    cal_ho=ho_gr*a+b
    ho_grad=np.gradient(ho_gr)  # dGR/dMD along hold (proxy; MD step ~1)
    cal_ho_grad=ho_grad*a
    def misfit(dz):
        t=ho_tvt+dz
        if mode=='value':
            return np.mean((cal_ho-twgr(t))**2)
        elif mode=='deriv':
            # match derivative shape: dGR/dMD vs typewell dGR/dTVT * (dTVT/dMD~local rate)
            # approximate: compare normalized derivatives
            a1=cal_ho_grad; a2=twgrad(t)
            a1=(a1-a1.mean())/(a1.std()+1e-6); a2=(a2-a2.mean())/(a2.std()+1e-6)
            return np.mean((a1-a2)**2)
        else: # combined
            m1=np.mean((cal_ho-twgr(t))**2)
            a1=cal_ho_grad; a2=twgrad(t)
            a1=(a1-a1.mean())/(a1.std()+1e-6); a2=(a2-a2.mean())/(a2.std()+1e-6)
            m2=np.mean((a1-a2)**2)
            return m1/ (np.var(cal_ho)+1e-6) + 0.5*m2
    cost=np.array([misfit(dz) for dz in GRID])
    best=GRID[int(np.argmin(cost))]
    return abs(best)<=2.0

NW=300
res={'value':[], 'deriv':[], 'combined':[]}
t0=time.time()
for i,wid in enumerate(well_ids[:NW]):
    for mode in ['value','deriv','combined']:
        r=localize(wid,mode)
        if r is not None: res[mode].append(r)
    if (i+1)%100==0: print('%d wells %.0fs'%(i+1,time.time()-t0),flush=True)
print('--- KNOWN-zone hold-out localization within 2ft (legal, calibrated on fit-part) ---',flush=True)
for mode in ['value','deriv','combined']:
    r=np.array(res[mode]); print('%-10s %.1f%% (n=%d)'%(mode,100*r.mean(),len(r)),flush=True)
print('%.0fs'%(time.time()-t0),flush=True)
