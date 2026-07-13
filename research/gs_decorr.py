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
    P=[];L=[]
    for s in range(ns):
        r,ll=run_pf(hw,tt,tg,s,**cfg); P.append(r); L.append(ll)
    P=np.stack(P,0);L=np.array(L);ln=L-L.max();wt=np.exp(ln/scale);wt/=wt.sum();return (wt[:,None]*P).sum(0)
# default = winner base config; variants decorrelate via gs (GR-sensitivity), NOT process noise
CFG={'default':dict(MOM=0.998,VN=0.002,PN=0.005,gs_mul=1.0),
     'lo':dict(MOM=0.999,VN=0.001,PN=0.002,gs_mul=1.0),         # the current v1 partner (process-noise decorr)
     'gs_tight':dict(MOM=0.998,VN=0.002,PN=0.005,gs_mul=0.6),   # GR matters MORE
     'gs_loose':dict(MOM=0.998,VN=0.002,PN=0.005,gs_mul=1.6)}   # continuity prior dominates
wells=[]
for wid in wids[-80:]:
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv'); tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv')
    km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells',len(wells),flush=True); t0=time.time()
pr={k:[] for k in CFG}; truth=[]
for k,(hw,tw,true) in enumerate(wells):
    truth.append(true)
    for n,c in CFG.items(): pr[n].append(ens(hw,tw,c))
    if (k+1)%20==0: print('%d %.0fs'%(k+1,time.time()-t0),flush=True)
truth=np.concatenate(truth); P={n:np.concatenate(v) for n,v in pr.items()}
print('\nstandalone:',flush=True)
for n in CFG: print('  %-9s %.3f'%(n,prmse(truth,P[n])),flush=True)
# correlation of ERRORS vs default (lower = more decorrelated = better blend partner)
ed=P['default']-truth
print('\nerror-corr vs default (lower=better partner):',flush=True)
for n in CFG:
    if n=='default':continue
    print('  %-9s %.3f'%(n,np.corrcoef(ed,P[n]-truth)[0,1]),flush=True)
print('\ncurrent v1 (0.65 default + 0.35 lo)   %.3f'%prmse(truth,0.65*P['default']+0.35*P['lo']),flush=True)
print('--- default + gs_tight sweep ---',flush=True)
for wl in [0.2,0.3,0.4,0.5]:
    print('  d%.2f+gsT%.2f  %.3f'%(1-wl,wl,prmse(truth,(1-wl)*P['default']+wl*P['gs_tight'])),flush=True)
print('--- default + gs_loose sweep ---',flush=True)
for wl in [0.2,0.3,0.4,0.5]:
    print('  d%.2f+gsL%.2f  %.3f'%(1-wl,wl,prmse(truth,(1-wl)*P['default']+wl*P['gs_loose'])),flush=True)
print('--- triple: default + lo + gs_tight ---',flush=True)
best=(9,None)
for wlo in [0.15,0.25,0.35]:
    for wg in [0.1,0.2,0.3]:
        wd=1-wlo-wg
        if wd<=0:continue
        r=prmse(truth,wd*P['default']+wlo*P['lo']+wg*P['gs_tight'])
        if r<best[0]:best=(r,('d%.2f'%wd,'lo%.2f'%wlo,'gsT%.2f'%wg))
print('  best triple:',best,flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
