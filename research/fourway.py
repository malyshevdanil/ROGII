import numpy as np, pandas as pd, glob, os, time, pickle, itertools
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
def ens(hw,tw,cfg,ns=12,scale=8.0):
    tw_s=tw.sort_values('TVT'); tt=tw_s['TVT'].values.astype(float); tg=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    P=[];L=[]
    for s in range(ns):
        r,ll=run_pf(hw,tt,tg,s,**cfg); P.append(r); L.append(ll)
    P=np.stack(P,0);L=np.array(L);ln=L-L.max();wt=np.exp(ln/scale);wt/=wt.sum();return (wt[:,None]*P).sum(0)
CFG={'default':dict(),'lo':dict(MOM=0.999,VN=0.001,PN=0.002),'gs':dict(gs_mul=0.6),'tj':dict(tw_jit=0.15)}
SLICES={'A':wids[-80:],'B':wids[-160:-80],'C':wids[100:180]}
store={}
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
    store[sname]=(truth,P)
    print('%s done n=%d %.0fs'%(sname,len(wells),time.time()-t0),flush=True)
try:
    pickle.dump(store,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fourway_store.pkl'),'wb'))
except Exception as e:
    print('pickle skipped:',e,flush=True)
# ---- evaluate blends across slices ----
def score(wfun):
    # wfun: dict name->weight applied per slice; return per-slice + avg
    rs=[]
    for s in store:
        tr,P=store[s]; pred=sum(wfun[n]*P[n] for n in wfun); rs.append(prmse(tr,pred))
    return rs,float(np.mean(rs))
print('\n=== reference blends (per-slice A,B,C | avg) ===',flush=True)
refs={'default':{'default':1.0},
      'v1 lo(0.65/0.35)':{'default':0.65,'lo':0.35},
      'v3 gs-triple(0.6/0.2/0.2)':{'default':0.6,'lo':0.2,'gs':0.2},
      'twjit-triple(0.6/0.2/0.2)':{'default':0.6,'lo':0.2,'tj':0.2}}
for name,w in refs.items():
    rs,av=score(w); print('  %-30s %s | %.3f'%(name,' '.join('%.3f'%x for x in rs),av),flush=True)
print('\n=== 4-way blends (default + lo + gs + tj) ===',flush=True)
best=(9,None)
# grid over partner weights; default gets remainder
for wl,wg,wt in itertools.product([0.1,0.15,0.2],[0.1,0.15,0.2],[0.1,0.15,0.2]):
    wd=1-wl-wg-wt
    if wd<0.4: continue
    w={'default':wd,'lo':wl,'gs':wg,'tj':wt}
    rs,av=score(w)
    if av<best[0]: best=(av,w,rs)
av,w,rs=best
print('  BEST 4-way: d%.2f lo%.2f gs%.2f tj%.2f -> %s | avg %.3f'%(w['default'],w['lo'],w['gs'],w['tj'],' '.join('%.3f'%x for x in rs),av),flush=True)
# also the symmetric default0.55/others each ~0.15
for w in [{'default':0.55,'lo':0.15,'gs':0.15,'tj':0.15},{'default':0.4,'lo':0.2,'gs':0.2,'tj':0.2},{'default':0.5,'lo':0.2,'gs':0.15,'tj':0.15}]:
    rs,av=score(w); print('  d%.2f lo%.2f gs%.2f tj%.2f -> %s | avg %.3f'%(w['default'],w['lo'],w['gs'],w['tj'],' '.join('%.3f'%x for x in rs),av),flush=True)
