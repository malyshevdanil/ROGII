import numpy as np, pandas as pd, glob, os, time
TRAIN_DIR='data/train'
wids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
def prmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def run_pf(hw,tw_tvt,tw_gr,seed,N=150,MOM=0.998,VN=0.002,PN=0.005,gs_mul=1.0,tw_jit=0.0):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]; last=kn.iloc[-1]
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); gs=float(np.clip(np.nanstd(kn['GR'].fillna(0).values-tw_at_k),10.,60.))*gs_mul
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values); dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.0
    rng=np.random.default_rng(seed); ls=float(last['TVT_input'])+float(last['Z'])
    tg = tw_gr + tw_jit*float(np.nanstd(tw_gr))*rng.standard_normal(len(tw_gr)) if tw_jit>0 else tw_gr
    pos=ls+4.5*rng.standard_normal(N); rate=ir+0.01*rng.standard_normal(N); w=np.ones(N)/N
    RP=0.1;RR=0.001;RESAMP=0.5; md_v=ev['MD'].values.astype(float); z_v=ev['Z'].values.astype(float)
    gr_v=hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)[ev.index.to_numpy()]
    res=np.empty(len(ev)); prev=float(last['MD']); ll=0.0
    for i in range(len(ev)):
        ds=max(md_v[i]-prev,1.0); rate=MOM*rate+VN*rng.standard_normal(N); pos=pos+rate*ds+PN*rng.standard_normal(N)
        tvt_p=np.clip(pos-z_v[i],tw_tvt[0]-100,tw_tvt[-1]+100); pos=tvt_p+z_v[i]
        eg=np.interp(tvt_p,tw_tvt,tg); d=(gr_v[i]-eg)/gs; lk=np.maximum(np.exp(-0.5*np.minimum(d*d,600.)),1e-300)
        avg=float((w*lk).sum()); ll+=np.log(max(avg,1e-300)); w=w*lk; ws=w.sum(); w=w/ws if ws>0 else np.ones(N)/N
        if 1.0/(w*w).sum()<RESAMP*N:
            cum=np.cumsum(w);u0=rng.uniform(0,1.0/N);idx=np.clip(np.searchsorted(cum,u0+np.arange(N)/N),0,N-1)
            pos=pos[idx]+RP*rng.standard_normal(N);rate=rate[idx]+RR*rng.standard_normal(N);w=np.ones(N)/N
        res[i]=float(np.dot(w,pos-z_v[i])); prev=md_v[i]
    return res,ll
def ens(hw,tt,tg,cfg,ns,scale=8.0,seed0=0):
    P=[];L=[]
    for s in range(seed0,seed0+ns):
        r,ll=run_pf(hw,tt,tg,s,**cfg); P.append(r); L.append(ll)
    P=np.stack(P,0);L=np.array(L);ln=L-L.max();wt=np.exp(ln/scale);wt/=wt.sum();return (wt[:,None]*P).sum(0)

# ~50 wells for speed, N=500 (production particles). Base ensemble fixed at 64 seeds (proxy for prod's 128;
# what matters is the PARTNER seed count, varied below). Partners: lo, gs, tj.
sub=wids[-160:-134]  # ~26 wells (slice-B region, where gs was weakest -> stress test)
wells=[]
for wid in sub:
    hw=pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv'); tw=pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv')
    km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    ts=tw.sort_values('TVT'); wells.append((hw,ts['TVT'].values.astype(float),ts['GR'].fillna(ts['GR'].mean()).values.astype(float),hw['TVT'].values[np.where(~km.values)[0]]))
print('wells',len(wells),'(N=500 particles)',flush=True); t0=time.time()
CFGp={'lo':dict(MOM=0.999,VN=0.001,PN=0.002),'gs':dict(gs_mul=0.6),'tj':dict(tw_jit=0.15)}
# base at 48 seeds (fixed)
base=[];truth=[]
part={k:{ns:[] for ns in [16,32,64]} for k in CFGp}
for hw,tt,tg,true in wells:
    truth.append(true)
    base.append(ens(hw,tt,tg,dict(),48))
    for k,c in CFGp.items():
        for ns in [16,32,64]:
            part[k][ns].append(ens(hw,tt,tg,c,ns))
    if (len(base))%10==0: print(' %d wells %.0fs'%(len(base),time.time()-t0),flush=True)
truth=np.concatenate(truth); base=np.concatenate(base)
P={k:{ns:np.concatenate(part[k][ns]) for ns in [16,32,64]} for k in CFGp}
print('\nbase-only RMSE %.3f'%prmse(truth,base),flush=True)
print('4-way (0.40 base + 0.20 each partner) at various PARTNER seed counts:',flush=True)
for ns in [16,32,64]:
    pred=0.40*base+0.20*P['lo'][ns]+0.20*P['gs'][ns]+0.20*P['tj'][ns]
    print('  partners@%2d seeds -> 4-way RMSE %.3f'%(ns,prmse(truth,pred)),flush=True)
print('\n(prod uses: v3/twjit partners @32, 4way @21; base @128. If 21/32 ~ 64 -> seed cut is safe.)',flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
