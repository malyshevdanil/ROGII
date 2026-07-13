import numpy as np, pandas as pd, glob, os, time, sys, pickle
sys.path.insert(0, r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad')
import beam_mod
from beam_mod import beam_search, BEAMS

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
OUT=r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad\cache'
os.makedirs(OUT, exist_ok=True)
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a

def run_pf(hw,tw_tvt,tw_gr,seed,N=200):
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

SCALES=[3.,5.,8.,12.]
def process(wid, nseed=24):
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    tw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')).sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30 or len(tw)<10: return None
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    ev=hw[~km]; ei=ev.index.to_numpy()
    last=hw[km].iloc[-1]; last_tvt=float(last['TVT_input']); last_MD=float(last['MD']); last_Z=float(last['Z'])
    # PF ensemble
    preds=[]; liks=[]
    for s in range(nseed):
        r,ll=run_pf(hw,tw_tvt,tw_gr,s); preds.append(r); liks.append(ll)
    preds=np.stack(preds,0); liks=np.array(liks); ln=liks-liks.max()
    pf={}
    for sc in SCALES:
        wt=np.exp(ln/sc); wt/=wt.sum(); pf['pf%g'%sc]=(wt[:,None]*preds).sum(0).astype(np.float32)
    pf['pfmean']=preds.mean(0).astype(np.float32)
    seed_std=preds.std(0).astype(np.float32)
    # beam ensemble
    hgr=interp_nan(hw['GR'].values.astype(float))[ei]
    beams=[beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r) for (bs,mc,es,r,tag) in BEAMS]
    beam=np.stack(beams,0).mean(0).astype(np.float32)
    # baselines
    md=ev['MD'].values.astype(float); z=ev['Z'].values.astype(float)
    flat=np.full(len(ei),last_tvt,np.float32)
    tail=hw[km].tail(200); lin_c=np.polyfit(tail['MD'].values,tail['TVT_input'].values,1)
    linear=np.polyval(lin_c,md).astype(np.float32)
    true=hw['TVT'].values[ei].astype(np.float32)
    # cheap features
    kn=hw[km]; kx=float(kn['X'].median()); ky=float(kn['Y'].median())
    feat=dict(md_since=(md-last_MD).astype(np.float32), z=z.astype(np.float32),
              gr=interp_nan(hw['GR'].values.astype(float))[ei].astype(np.float32),
              seed_std=seed_std, n_eval=len(ei), z_span=float(np.ptp(z)),
              kx=kx, ky=ky, last_tvt=last_tvt)
    return dict(wid=wid, ei=ei.astype(np.int32), true=true, flat=flat, linear=linear,
                beam=beam, seed_std=seed_std, **pf, feat=feat)

t0=time.time(); done=0; results={}
for i,wid in enumerate(well_ids):
    try:
        r=process(wid)
        if r is not None: results[wid]=r; done+=1
    except Exception as e:
        print('ERR',wid,e,flush=True)
    if (i+1)%50==0:
        print('%d/%d done=%d %.0fs'%(i+1,len(well_ids),done,time.time()-t0),flush=True)
        with open(os.path.join(OUT,'cache.pkl'),'wb') as f: pickle.dump(results,f)
with open(os.path.join(OUT,'cache.pkl'),'wb') as f: pickle.dump(results,f)
print('CACHE DONE: %d wells, %.0fs'%(done,time.time()-t0),flush=True)
