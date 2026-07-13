import numpy as np, pandas as pd, glob, os, time, sys
sys.path.insert(0, r'C:\Users\user\AppData\Local\Temp\claude\d--ROGII\6761a874-94c4-40fb-a605-d49c9b58b718\scratchpad')
import beam_mod
from beam_mod import beam_search, BEAMS

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def prmse(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);return float(np.sqrt(np.mean((a-b)**2)))

def interp_nan(a):
    a=a.copy();n=len(a);idx=np.arange(n);m=np.isnan(a)
    if m.all():return np.zeros(n)
    a[m]=np.interp(idx[m],idx[~m],a[~m]);return a

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')).sort_values('TVT'))

# Full banded DP, BIDIRECTIONAL moves in typewell index, smoothness-penalized.
def full_dp(gr_eval, tw_tvt, tw_gr, start_idx, es, mc, maxstep=3, band=120):
    n=len(gr_eval); nt=len(tw_gr)
    lo=max(0,start_idx-band); hi=min(nt,start_idx+n+band)
    idxs=np.arange(lo,hi); m=len(idxs)
    INF=1e18
    prev=(tw_gr[idxs]-gr_eval[0])**2/es + np.abs(idxs-start_idx).astype(float)*mc
    back=np.zeros((n,m),np.int32)
    ar=np.arange(m)
    for i in range(1,n):
        emis=(tw_gr[idxs]-gr_eval[i])**2/es
        cur=np.full(m,INF); bk=np.zeros(m,np.int32)
        for d in range(-maxstep,maxstep+1):
            src=ar-d
            valid=(src>=0)&(src<m)
            cand=np.full(m,INF)
            cand[valid]=prev[src[valid]]+ (abs(d)*mc)
            tot=cand+emis
            better=tot<cur
            cur=np.where(better,tot,cur); bk=np.where(better,src,bk)
        prev=cur; back[i]=bk
    j=int(np.argmin(prev)); path=np.zeros(n,np.int32); path[n-1]=j
    for i in range(n-1,0,-1): j=back[i,j]; path[i-1]=j
    return tw_tvt[idxs[path]]

def dp_ensemble(hw, tw):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    last_tvt=float(kn.iloc[-1]['TVT_input'])
    start=int(np.clip(np.searchsorted(tw_tvt,last_tvt),0,len(tw_tvt)-1))
    gr_all=interp_nan(hw['GR'].values.astype(float)); hgr=gr_all[ev.index.to_numpy()]
    # smooth like beam does (rolling mean r), ensemble a few (es,mc) configs
    outs=[]
    for r in [2,3,5]:
        g=pd.Series(hgr).rolling(2*r+1,center=True,min_periods=1).mean().values
        for es,mc in [(100.,15.),(150.,25.),(64.,8.)]:
            outs.append(full_dp(g,tw_tvt,tw_gr,start,es,mc))
    return np.stack(outs,0).mean(0), ev.index.to_numpy()

def beam_ens(hw,tw):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    last_tvt=float(kn.iloc[-1]['TVT_input'])
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    gr_all=interp_nan(hw['GR'].values.astype(float)); hgr=gr_all[ev.index.to_numpy()]
    beams=[beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r) for (bs,mc,es,r,tag) in BEAMS]
    return np.stack(beams,0).mean(0)

_hw,_tw=load(well_ids[0]); beam_ens(_hw,_tw)
NW=60; wells=[]
for wid in well_ids[-NW:]:
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells:',len(wells),flush=True)

t0=time.time(); at,bd,fd=[],[],[]
for k,(hw,tw,true) in enumerate(wells):
    bm=beam_ens(hw,tw)
    dp,ei=dp_ensemble(hw,tw)
    at.append(true); bd.append(bm); fd.append(dp)
    if (k+1)%20==0: print('%d %.0fs'%(k+1,time.time()-t0),flush=True)
at=np.concatenate(at)
print('beam ensemble   RMSE=%.3f'%prmse(at,np.concatenate(bd)),flush=True)
print('full DP ensemble RMSE=%.3f'%prmse(at,np.concatenate(fd)),flush=True)
# also a 50/50 blend (decorrelation check)
bl=0.5*np.concatenate(bd)+0.5*np.concatenate(fd)
print('beam+DP blend    RMSE=%.3f'%prmse(at,bl),flush=True)
print('%.0fs'%(time.time()-t0),flush=True)
