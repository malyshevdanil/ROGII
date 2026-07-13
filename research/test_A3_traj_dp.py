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

# Trajectory-informed DP: state = fractional typewell TVT (not index). We track TVT directly on a
# fine grid, expected step per MD = local (dSurface - dZ). Penalize deviation from expected tightly.
def traj_dp(hw, tw, es, dev_pen, maxdev_ft=1.5, grid_step=0.25):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    last=kn.iloc[-1]; last_tvt=float(last['TVT_input']); last_MD=float(last['MD']); last_Z=float(last['Z'])
    # calibrate GR->twGR on known
    kgr=interp_nan(kn['GR'].values.astype(float)); tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)
    v=np.isfinite(kgr)&np.isfinite(tw_at_k); a,b=(np.polyfit(kgr[v],tw_at_k[v],1) if v.sum()>=20 else (1.,0.))
    gr_all=interp_nan(hw['GR'].values.astype(float))
    ei=ev.index.to_numpy(); hgr=gr_all[ei]
    md=ev['MD'].values.astype(float); z=ev['Z'].values.astype(float)
    # expected dTVT per step from known-zone trend of the SURFACE (TVT+Z), which is ~linear
    tail=kn.tail(80)
    if len(tail)>=10:
        surf_slope=np.polyfit(tail['MD'].values, (tail['TVT_input'].values+tail['Z'].values),1)[0]
    else: surf_slope=0.0
    # candidate TVT grid around a physical prior band
    prior=last_tvt + np.concatenate([[surf_slope*(md[0]-last_MD)-(z[0]-last_Z)]])  # not used directly
    lo=min(tw_tvt.min(), last_tvt-160); hi=max(tw_tvt.max(), last_tvt+160)
    grid=np.arange(lo,hi,grid_step); G=len(grid)
    twg_grid=np.interp(grid,tw_tvt,tw_gr)   # typewell GR at each grid TVT
    n=len(ev)
    maxdev=int(round(maxdev_ft/grid_step))
    INF=1e18
    # emission for step 0
    cal0=hgr[0]*a+b
    prev=(cal0-twg_grid)**2/es + np.abs(grid-last_tvt)*0.3   # soft anchor to last_tvt at start
    back=np.zeros((n,G),np.int32); ar=np.arange(G)
    prevMD=md[0]; prevZ=z[0]
    for i in range(1,n):
        dmd=max(md[i]-prevMD,1.0); dz=z[i]-prevZ
        exp_dtvt=surf_slope*dmd - dz          # expected TVT change = dSurface - dZ
        exp_shift=int(round(exp_dtvt/grid_step))
        cal=hgr[i]*a+b
        emis=(cal-twg_grid)**2/es
        cur=np.full(G,INF); bk=np.zeros(G,np.int32)
        for d in range(exp_shift-maxdev, exp_shift+maxdev+1):
            src=ar-d; valid=(src>=0)&(src<G)
            cand=np.full(G,INF); cand[valid]=prev[src[valid]] + dev_pen*(d-exp_shift)**2
            tot=cand+emis; better=tot<cur; cur=np.where(better,tot,cur); bk=np.where(better,src,bk)
        prev=cur; back[i]=bk; prevMD=md[i]; prevZ=z[i]
    j=int(np.argmin(prev)); path=np.zeros(n,np.int32); path[n-1]=j
    for i in range(n-1,0,-1): j=back[i,j]; path[i-1]=j
    out=grid[path]
    return out, ei

def traj_ensemble(hw,tw):
    outs=[]
    for es,dp in [(80.,0.5),(150.,1.0),(60.,0.3),(120.,2.0)]:
        o,ei=traj_dp(hw,tw,es,dp); outs.append(o)
    return np.stack(outs,0).mean(0), ei

def beam_ens(hw,tw):
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    last_tvt=float(kn.iloc[-1]['TVT_input'])
    tw_tvt=tw['TVT'].values.astype(float); tw_gr=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    hgr=interp_nan(hw['GR'].values.astype(float))[ev.index.to_numpy()]
    beams=[beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r) for (bs,mc,es,r,tag) in BEAMS]
    return np.stack(beams,0).mean(0)

_hw,_tw=load(well_ids[0]); beam_ens(_hw,_tw)
NW=60; wells=[]
for wid in well_ids[-NW:]:
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells:',len(wells),flush=True)
t0=time.time(); at,bm_,tj_=[],[],[]
for k,(hw,tw,true) in enumerate(wells):
    at.append(true); bm_.append(beam_ens(hw,tw)); tj_.append(traj_ensemble(hw,tw)[0])
    if (k+1)%20==0: print('%d %.0fs'%(k+1,time.time()-t0),flush=True)
at=np.concatenate(at)
print('beam ensemble        RMSE=%.3f'%prmse(at,np.concatenate(bm_)),flush=True)
print('traj-informed DP     RMSE=%.3f'%prmse(at,np.concatenate(tj_)),flush=True)
print('beam+traj blend      RMSE=%.3f'%prmse(at,0.5*np.concatenate(bm_)+0.5*np.concatenate(tj_)),flush=True)
print('%.0fs'%(time.time()-t0),flush=True)
