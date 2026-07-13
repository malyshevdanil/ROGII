import numpy as np, pandas as pd, glob, os, time

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def prmse(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);return float(np.sqrt(np.mean((a-b)**2)))

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

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')))

NW=50; NSEED=48
wells=[]
for wid in well_ids[-NW:]:
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells:',len(wells),'seeds:',NSEED,flush=True)

def combine(preds, liks, scale):
    if scale=='mean': return preds.mean(0)
    ln=liks-liks.max(); wt=np.exp(ln/scale); wt/=wt.sum(); return (wt[:,None]*preds).sum(0)

scales=[3.0,5.0,8.0,12.0,'mean']
pooled={s:([],[]) for s in scales}   # (true, pred)
per_well_best=[]   # oracle: best scale per well
per_well_spread=[] # bimodality proxy: mean over steps of seed std
t0=time.time()
for k,(hw,tw,true) in enumerate(wells):
    preds=[];liks=[]
    for s in range(NSEED):
        r,ll=run_pf(hw,tw,s); preds.append(r); liks.append(ll)
    preds=np.stack(preds,0); liks=np.array(liks)
    spread=float(np.mean(preds.std(0)))   # avg seed disagreement across the well
    per_well_spread.append(spread)
    errs={}
    for s in scales:
        p=combine(preds,liks,s)
        pooled[s][0].append(true); pooled[s][1].append(p)
        errs[s]=prmse(true,p)
    per_well_best.append(min(errs.values()))
    if (k+1)%10==0: print('%d wells %.0fs'%(k+1,time.time()-t0),flush=True)

print('--- fixed scale, pooled RMSE ---',flush=True)
for s in scales:
    print('scale=%-5s RMSE=%.3f'%(str(s),prmse(np.concatenate(pooled[s][0]),np.concatenate(pooled[s][1]))),flush=True)
# oracle per-well scale selection (upper bound on adaptive gain)
oracle_true=np.concatenate([pooled[3.0][0][i] for i in range(len(wells))])
# recompute oracle properly: for each well pick min-rmse scale
allt=[];allp=[]
for i,(hw,tw,true) in enumerate(wells):
    best_s=min(scales, key=lambda s: prmse(pooled[s][0][i],pooled[s][1][i]))
    allt.append(pooled[best_s][0][i]); allp.append(pooled[best_s][1][i])
print('ORACLE per-well scale  RMSE=%.3f  (upper bound if we could pick scale per well)'%prmse(np.concatenate(allt),np.concatenate(allp)),flush=True)

# does spread predict which scale is best? correlation of spread with (mean-scale better than scale5)
import numpy as np
spread=np.array(per_well_spread)
adv=np.array([prmse(pooled[5.0][0][i],pooled[5.0][1][i]) - prmse(pooled['mean'][0][i],pooled['mean'][1][i]) for i in range(len(wells))])
# adv>0 means 'mean' (hedge) beats scale5 (commit) for that well
print('wells where hedge(mean) beats commit(scale5): %d/%d'%((adv>0).sum(),len(wells)),flush=True)
print('corr(seed-spread, hedge-advantage) = %.3f'%np.corrcoef(spread,adv)[0,1],flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
