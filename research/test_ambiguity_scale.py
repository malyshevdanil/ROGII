import numpy as np, pandas as pd, glob, os, time

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def prmse(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);return float(np.sqrt(np.mean((a-b)**2)))

# ---- known-zone GR-match ambiguity (principled, legal) ----
GRID=np.arange(-20.0,20.01,0.5)
def known_ambiguity(hw, tw):
    tw_s=tw.sort_values('TVT'); tw_tvt=tw_s['TVT'].values.astype(float); tw_gr=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn=hw[hw['TVT_input'].notna()]
    ktvt=kn['TVT_input'].values.astype(float); kgr=kn['GR'].interpolate().bfill().ffill().values.astype(float)
    v=np.isfinite(kgr)
    if v.sum()<30: return 1.0, 0.0
    def twgr(t): return np.interp(t,tw_tvt,tw_gr,left=tw_gr[0],right=tw_gr[-1])
    # calibrate on known
    tgt=twgr(ktvt)
    coef=np.polyfit(kgr[v],tgt[v],1); cal=kgr*coef[0]+coef[1]
    cost=np.array([np.mean((cal[v]-twgr(ktvt[v]+dz))**2) for dz in GRID])
    i1=int(np.argmin(cost)); m1=cost[i1]
    mask=np.abs(GRID-GRID[i1])>=6.0
    if mask.any():
        j=np.flatnonzero(mask)[int(np.argmin(cost[mask]))]; m2=cost[j]
    else: m2=m1
    ratio=m2/max(m1,1e-9)     # ~1 => ambiguous (2 equally-good), large => clear single minimum
    return ratio, m1

def run_pf(hw,tw,seed,N=300):
    tw_s=tw.sort_values('TVT'); tw_tvt=tw_s['TVT'].values.astype(float); tw_gr=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]; last=kn.iloc[-1]
    tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr); gs=float(np.clip(np.nanstd(kn['GR'].fillna(0).values-tw_at_k),10.,60.))
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values); dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.0
    rng=np.random.default_rng(seed); ls=float(last['TVT_input'])+float(last['Z'])
    pos=ls+4.5*rng.standard_normal(N); rate=ir+0.01*rng.standard_normal(N); w=np.ones(N)/N
    MOM=0.998;VN=0.002;PN=0.005;RP=0.1;RR=0.001;RESAMP=0.5
    md_v=ev['MD'].values.astype(float); z_v=ev['Z'].values.astype(float)
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
    return res, ll

def combine(preds,liks,scale):
    if scale=='mean': return preds.mean(0)
    ln=liks-liks.max(); wt=np.exp(ln/scale); wt/=wt.sum(); return (wt[:,None]*preds).sum(0)

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')))

NW=50; NSEED=48
wells=[]
for wid in well_ids[-NW:]:
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells:',len(wells),flush=True)

scales=[3.0,5.0,8.0,12.0]
rows=[]
t0=time.time()
data=[]  # (true, {scale:pred}, ambiguity)
for k,(hw,tw,true) in enumerate(wells):
    ratio,m1=known_ambiguity(hw,tw)
    preds=[];liks=[]
    for s in range(NSEED):
        r,ll=run_pf(hw,tw,s); preds.append(r); liks.append(ll)
    preds=np.stack(preds,0); liks=np.array(liks)
    pd_={s:combine(preds,liks,s) for s in scales}
    data.append((true,pd_,ratio))
    if (k+1)%10==0: print('%d wells %.0fs'%(k+1,time.time()-t0),flush=True)

# which scale best per well vs ambiguity
best_scale=[]; ambs=[]
for true,pd_,ratio in data:
    errs={s:prmse(true,pd_[s]) for s in scales}
    best_scale.append(min(errs,key=errs.get)); ambs.append(ratio)
ambs=np.array(ambs); best_scale=np.array(best_scale,float)
print('ambiguity ratio: min=%.2f median=%.2f max=%.2f'%(ambs.min(),np.median(ambs),ambs.max()),flush=True)
print('corr(ambiguity, best_scale) = %.3f  (expect NEGATIVE: low ratio=ambiguous->want high scale/hedge)'%np.corrcoef(ambs,best_scale)[0,1],flush=True)

# adaptive rule: ambiguous (ratio<thr) -> scale12, else -> scale5 (commit). test vs best fixed(12) and oracle
for thr in [1.1,1.2,1.3,1.5,2.0]:
    at,ap=[],[]
    for true,pd_,ratio in data:
        s = 12.0 if ratio<thr else 5.0
        at.append(true); ap.append(pd_[s])
    print('adaptive thr=%.1f (amb->12 else->5)  RMSE=%.3f'%(thr,prmse(np.concatenate(at),np.concatenate(ap))),flush=True)
# baselines
for s in scales:
    at=[d[0] for d in data]; ap=[d[1][s] for d in data]
    print('fixed scale=%-4s RMSE=%.3f'%(s,prmse(np.concatenate(at),np.concatenate(ap))),flush=True)
oa_t=[];oa_p=[]
for true,pd_,ratio in data:
    errs={s:prmse(true,pd_[s]) for s in scales}; bs=min(errs,key=errs.get); oa_t.append(true); oa_p.append(pd_[bs])
print('ORACLE per-well  RMSE=%.3f'%prmse(np.concatenate(oa_t),np.concatenate(oa_p)),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
