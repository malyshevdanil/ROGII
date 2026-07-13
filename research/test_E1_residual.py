import numpy as np, pandas as pd, glob, os, time

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def run_pf(hw,tw,seed,N=200):
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
    res=np.empty(len(ev)); std=np.empty(len(ev)); prev=float(last['MD'])
    for i in range(len(ev)):
        ds=max(md_v[i]-prev,1.0); rate=MOM*rate+VN*rng.standard_normal(N); pos=pos+rate*ds+PN*rng.standard_normal(N)
        tvt_p=np.clip(pos-z_v[i],tw_tvt[0]-100,tw_tvt[-1]+100); pos=tvt_p+z_v[i]
        eg=np.interp(tvt_p,tw_tvt,tw_gr); d=(gr_v[i]-eg)/gs; lk=np.maximum(np.exp(-0.5*np.minimum(d*d,600.)),1e-300)
        w=w*lk; ws=w.sum(); w=w/ws if ws>0 else np.ones(N)/N
        if 1.0/(w*w).sum()<RESAMP*N:
            cum=np.cumsum(w);u0=rng.uniform(0,1.0/N);idx=np.clip(np.searchsorted(cum,u0+np.arange(N)/N),0,N-1)
            pos=pos[idx]+RP*rng.standard_normal(N);rate=rate[idx]+RR*rng.standard_normal(N);w=np.ones(N)/N
        est=float(np.dot(w,pos-z_v[i])); res[i]=est; std[i]=float(np.sqrt(np.dot(w,(pos-z_v[i]-est)**2))); prev=md_v[i]
    return res,std,ev.index.to_numpy()

def ens(hw,tw,ns=12,scale=8.0):
    P=[];S=[];L=[]
    ll_all=[]
    for s in range(ns):
        r,st,ei=run_pf(hw,tw,s); P.append(r); S.append(st)
    P=np.stack(P,0); S=np.stack(S,0)
    return P.mean(0), S.mean(0), P.std(0), ei

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')))

NW=60
rows=[]
t0=time.time()
allres=[]; allfeat=[]
for k,wid in enumerate(well_ids[-NW:]):
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    pred,pfstd,seedstd,ei=ens(hw,tw)
    true=hw['TVT'].values[ei]
    resid=true-pred     # what E1 would try to learn
    # cheap observables at each eval point
    md=hw['MD'].values[ei]; last_md=float(hw[km]['MD'].iloc[-1]); md_since=md-last_md
    gr=hw['GR'].interpolate().bfill().ffill().values[ei]
    z=hw['Z'].values[ei]
    feat=np.stack([md_since, pfstd, seedstd, gr, np.abs(pred-float(hw[km]['TVT_input'].iloc[-1]))],1)
    allres.append(resid); allfeat.append(feat)
    if (k+1)%20==0: print('%d %.0fs'%(k+1,time.time()-t0),flush=True)
resid=np.concatenate(allres); feat=np.concatenate(allfeat,0)
names=['md_since','pf_std','seed_std','gr','dist_from_last']
print('PF residual std: %.2f (this is the learnable target scale)'%resid.std(),flush=True)
print('--- correlation of PF residual with cheap observables ---',flush=True)
for j,nm in enumerate(names):
    c=np.corrcoef(feat[:,j],resid)[0,1]
    # also correlation with |residual| (magnitude predictability)
    cm=np.corrcoef(feat[:,j],np.abs(resid))[0,1]
    print('%-16s corr(resid)=%+.3f   corr(|resid|)=%+.3f'%(nm,c,cm),flush=True)
# can a linear fit on these reduce residual at all? (in-sample upper bound)
from numpy.linalg import lstsq
X=np.column_stack([feat,np.ones(len(feat))])
coef,*_=lstsq(X,resid,rcond=None); pred_r=X@coef
print('residual RMSE before=%.3f  after in-sample linear fit=%.3f (in-sample UPPER bound)'%(np.sqrt((resid**2).mean()),np.sqrt(((resid-pred_r)**2).mean())),flush=True)
print('%.0fs'%(time.time()-t0),flush=True)
