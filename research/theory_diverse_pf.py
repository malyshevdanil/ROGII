import numpy as np, pandas as pd, glob, os, time, itertools

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def prmse(a,b):a=np.asarray(a);b=np.asarray(b);return float(np.sqrt(np.mean((a-b)**2)))

# PF with configurable process noise (VN,PN), momentum (MOM), GR smoothing (grsm)
def run_pf(hw,tw_tvt,tw_gr,seed,N=150,MOM=0.998,VN=0.002,PN=0.005,grsm=0):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]; last=kn.iloc[-1]
    gr_ref=tw_gr.copy()
    if grsm>0:
        gr_ref=pd.Series(tw_gr).rolling(grsm*2+1,center=True,min_periods=1).mean().values
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); gs=float(np.clip(np.nanstd(kn['GR'].fillna(0).values-tw_at_k),10.,60.))
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values); dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.0
    rng=np.random.default_rng(seed); ls=float(last['TVT_input'])+float(last['Z'])
    pos=ls+4.5*rng.standard_normal(N); rate=ir+0.01*rng.standard_normal(N); w=np.ones(N)/N
    RP=0.1;RR=0.001;RESAMP=0.5
    md_v=ev['MD'].values.astype(float); z_v=ev['Z'].values.astype(float)
    gr_v=hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)[ev.index.to_numpy()]
    if grsm>0:
        gr_v=pd.Series(gr_v).rolling(grsm*2+1,center=True,min_periods=1).mean().values
    res=np.empty(len(ev)); prev=float(last['MD']); ll=0.0
    for i in range(len(ev)):
        ds=max(md_v[i]-prev,1.0); rate=MOM*rate+VN*rng.standard_normal(N); pos=pos+rate*ds+PN*rng.standard_normal(N)
        tvt_p=np.clip(pos-z_v[i],tw_tvt[0]-100,tw_tvt[-1]+100); pos=tvt_p+z_v[i]
        eg=np.interp(tvt_p,tw_tvt,gr_ref); d=(gr_v[i]-eg)/gs; lk=np.maximum(np.exp(-0.5*np.minimum(d*d,600.)),1e-300)
        avg=float((w*lk).sum()); ll+=np.log(max(avg,1e-300)); w=w*lk; ws=w.sum(); w=w/ws if ws>0 else np.ones(N)/N
        if 1.0/(w*w).sum()<RESAMP*N:
            cum=np.cumsum(w);u0=rng.uniform(0,1.0/N);idx=np.clip(np.searchsorted(cum,u0+np.arange(N)/N),0,N-1)
            pos=pos[idx]+RP*rng.standard_normal(N);rate=rate[idx]+RR*rng.standard_normal(N);w=np.ones(N)/N
        res[i]=float(np.dot(w,pos-z_v[i])); prev=md_v[i]
    return res,ll

def ens(hw,tw,cfg,ns=12,scale=8.0):
    tw_s=tw.sort_values('TVT'); tw_tvt=tw_s['TVT'].values.astype(float); tw_gr=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    P=[];L=[]
    for s in range(ns):
        r,ll=run_pf(hw,tw_tvt,tw_gr,s,**cfg); P.append(r); L.append(ll)
    P=np.stack(P,0); L=np.array(L); ln=L-L.max(); wt=np.exp(ln/scale); wt/=wt.sum()
    return (wt[:,None]*P).sum(0)

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')))

CONFIGS={
 'default':dict(MOM=0.998,VN=0.002,PN=0.005,grsm=0),
 'hi_noise':dict(MOM=0.995,VN=0.006,PN=0.015,grsm=0),   # more exploratory
 'lo_noise':dict(MOM=0.999,VN=0.001,PN=0.002,grsm=0),   # stiffer
 'gr_smooth':dict(MOM=0.998,VN=0.002,PN=0.005,grsm=5),  # smoothed GR sensitivity
}
NW=100; wells=[]
for wid in well_ids[-NW:]:
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells',len(wells),flush=True)
t0=time.time()
preds={k:[] for k in CONFIGS}; truth=[]
for k,(hw,tw,true) in enumerate(wells):
    truth.append(true)
    for name,cfg in CONFIGS.items():
        preds[name].append(ens(hw,tw,cfg))
    if (k+1)%25==0: print('%d %.0fs'%(k+1,time.time()-t0),flush=True)
truth=np.concatenate(truth)
P={n:np.concatenate(v) for n,v in preds.items()}
print('\n--- standalone RMSE ---',flush=True)
for n in CONFIGS: print('%-10s %.3f'%(n,prmse(truth,P[n])),flush=True)
print('\n--- error correlation matrix (want < 0.9 for decorrelation) ---',flush=True)
names=list(CONFIGS); E={n:truth-P[n] for n in names}
print('           '+' '.join('%9s'%n[:8] for n in names),flush=True)
for a in names:
    print('%-10s '%a[:8]+' '.join('%9.3f'%np.corrcoef(E[a],E[b])[0,1] for b in names),flush=True)
print('\n--- blends ---',flush=True)
print('default alone           %.3f'%prmse(truth,P['default']),flush=True)
print('mean(all 4 configs)     %.3f'%prmse(truth,np.mean([P[n] for n in names],0)),flush=True)
# best pairwise blend with default
for n in names:
    if n=='default':continue
    best=min(prmse(truth,w*P['default']+(1-w)*P[n]) for w in [0.4,0.5,0.6,0.7])
    print('best default+%-10s  %.3f'%(n,best),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
