import numpy as np, pandas as pd, glob, os, time
TRAIN_DIR='data/train'
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

def run_pf(hw,tw_tvt,tw_gr,seed,N=150,MOM=0.998,VN=0.002,PN=0.005,gs_mul=1.0,
           pn_df=0,tw_jit=0.0):
    # pn_df>0 -> Student-t process noise (heavy tails, variance-matched); tw_jit -> per-seed typewell GR jitter
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]; last=kn.iloc[-1]
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); gs=float(np.clip(np.nanstd(kn['GR'].fillna(0).values-tw_at_k),10.,60.))*gs_mul
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values); dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.0
    rng=np.random.default_rng(seed); ls=float(last['TVT_input'])+float(last['Z'])
    # optional per-seed typewell jitter (bootstrap the reference)
    if tw_jit>0:
        tg=tw_gr + tw_jit*float(np.nanstd(tw_gr))*rng.standard_normal(len(tw_gr))
    else:
        tg=tw_gr
    # heavy-tail noise generator (variance-matched to standard normal)
    if pn_df>0:
        sc=np.sqrt((pn_df-2.0)/pn_df) if pn_df>2 else 1.0
        nz=lambda n: rng.standard_t(pn_df,n)*sc
    else:
        nz=lambda n: rng.standard_normal(n)
    pos=ls+4.5*rng.standard_normal(N); rate=ir+0.01*rng.standard_normal(N); w=np.ones(N)/N
    RP=0.1;RR=0.001;RESAMP=0.5; md_v=ev['MD'].values.astype(float); z_v=ev['Z'].values.astype(float)
    gr_v=hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)[ev.index.to_numpy()]
    res=np.empty(len(ev)); prev=float(last['MD']); ll=0.0
    for i in range(len(ev)):
        ds=max(md_v[i]-prev,1.0); rate=MOM*rate+VN*nz(N); pos=pos+rate*ds+PN*nz(N)
        tvt_p=np.clip(pos-z_v[i],tw_tvt[0]-100,tw_tvt[-1]+100); pos=tvt_p+z_v[i]
        eg=np.interp(tvt_p,tw_tvt,tg); d=(gr_v[i]-eg)/gs; lk=np.maximum(np.exp(-0.5*np.minimum(d*d,600.)),1e-300)
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

CFG={'default':dict(),
     'heavytail':dict(pn_df=3),      # Student-t process noise (bimodal jumps)
     'twjit':dict(tw_jit=0.15),      # bootstrap the typewell reference
     'lo':dict(MOM=0.999,VN=0.001,PN=0.002)}  # proven v1 partner for reference
SLICES={'A_last80':wids[-80:],'B_mid':wids[-160:-80]}
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
    ed=P['default']-truth
    print('\n=== %s (n=%d %.0fs) ==='%(sname,len(wells),time.time()-t0),flush=True)
    for n in CFG: print('  standalone %-9s %.3f'%(n,prmse(truth,P[n])),flush=True)
    print('  error-corr vs default:',flush=True)
    for n in CFG:
        if n=='default':continue
        print('    %-9s %.3f'%(n,np.corrcoef(ed,P[n]-truth)[0,1]),flush=True)
    print('  v1 ref (0.65d+0.35lo)              %.3f'%prmse(truth,0.65*P['default']+0.35*P['lo']),flush=True)
    for n in ['heavytail','twjit']:
        for wl in [0.2,0.3,0.4]:
            print('  d%.2f+%s%.2f  %.3f'%(1-wl,n,wl,prmse(truth,(1-wl)*P['default']+wl*P[n])),flush=True)
    # best triple lo+new
    for n in ['heavytail','twjit']:
        print('  triple d0.6+lo0.2+%s0.2  %.3f'%(n,prmse(truth,0.6*P['default']+0.2*P['lo']+0.2*P[n])),flush=True)
