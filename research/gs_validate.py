import numpy as np, pandas as pd, glob, os, time
TRAIN_DIR='data/train'
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def run_pf(hw,tw_tvt,tw_gr,seed,N=150,MOM=0.998,VN=0.002,PN=0.005,gs_mul=1.0):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]; last=kn.iloc[-1]
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); gs=float(np.clip(np.nanstd(kn['GR'].fillna(0).values-tw_at_k),10.,60.))*gs_mul
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values); dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.0
    rng=np.random.default_rng(seed); ls=float(last['TVT_input'])+float(last['Z'])
    pos=ls+4.5*rng.standard_normal(N); rate=ir+0.01*rng.standard_normal(N); w=np.ones(N)/N
    RP=0.1;RR=0.001;RESAMP=0.5; md_v=ev['MD'].values.astype(float); z_v=ev['Z'].values.astype(float)
    gr_v=hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)[ev.index.to_numpy()]
    res=np.empty(len(ev)); prev=float(last['MD']); ll=0.0
    for i in range(len(ev)):
        ds=max(md_v[i]-prev,1.0); rate=MOM*rate+VN*rng.standard_normal(N); pos=pos+rate*ds+PN*rng.standard_normal(N)
        tvt_p=np.clip(pos-z_v[i],tw_tvt[0]-100,tw_tvt[-1]+100); pos=tvt_p+z_v[i]
        eg=np.interp(tvt_p,tw_tvt,tw_gr); d=(gr_v[i]-eg)/gs; lk=np.maximum(np.exp(-0.5*np.minimum(d*d,600.)),1e-300)
        avg=float((w*lk).sum()); ll+=np.log(max(avg,1e-300)); w=w*lk; ws=w.sum(); w=w/ws if ws>0 else np.ones(N)/N
        if 1.0/(w*w).sum()<RESAMP*N:
            cum=np.cumsum(w);u0=rng.uniform(0,1.0/N);idx=np.clip(np.searchsorted(cum,u0+np.arange(N)/N),0,N-1)
            pos=pos[idx]+RP*rng.standard_normal(N);rate=rate[idx]+RR*rng.standard_normal(N);w=np.ones(N)/N
        res[i]=float(np.dot(w,pos-z_v[i])); prev=md_v[i]
    return res,ll
def ens(hw,tw,cfg,ns=12,scale=8.0):
    tw_s=tw.sort_values('TVT'); tt=tw_s['TVT'].values.astype(float); tg=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    P=[]
    for s in range(ns):
        r,ll=run_pf(hw,tt,tg,s,**cfg); P.append(r)
    return np.mean(np.stack(P,0),0) if False else _wavg(P,[run_pf(hw,tt,tg,s,**cfg)[1] for s in range(ns)],scale)
def _wavg(P,L,scale):
    P=np.stack(P,0);L=np.array(L);ln=L-L.max();wt=np.exp(ln/scale);wt/=wt.sum();return (wt[:,None]*P).sum(0)
CFG={'default':dict(gs_mul=1.0),'lo':dict(MOM=0.999,VN=0.001,PN=0.002),'gs_tight':dict(gs_mul=0.6)}
# THREE independent well slices to check generalization
SLICES={'A_last80':wids[-80:],'B_mid':wids[-160:-80],'C_early':wids[100:180]}
for sname,sl in SLICES.items():
    wells=[]
    for wid in sl:
        hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv'); tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv')
        km=hw['TVT_input'].notna()
        if km.sum()<30 or (~km).sum()<30: continue
        wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
    t0=time.time(); pr={k:[] for k in CFG}; truth=[]
    for hw,tw,true in wells:
        truth.append(true)
        for n,c in CFG.items(): pr[n].append(ens(hw,tw,c))
    truth=np.concatenate(truth); P={n:np.concatenate(v) for n,v in pr.items()}
    v1=prmse(truth,0.65*P['default']+0.35*P['lo'])
    g30=prmse(truth,0.70*P['default']+0.30*P['gs_tight'])
    combo=prmse(truth,0.60*P['default']+0.20*P['lo']+0.20*P['gs_tight'])
    print('%s (n=%d %.0fs): default=%.3f | v1(lo)=%.3f | gsT0.30=%.3f | d0.6+lo0.2+gsT0.2=%.3f'%(
        sname,len(wells),time.time()-t0,prmse(truth,P['default']),v1,g30,combo),flush=True)
