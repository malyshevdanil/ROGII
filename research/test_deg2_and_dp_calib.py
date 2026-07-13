import numpy as np, pandas as pd, glob, os, time

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def prmse(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);return float(np.sqrt(np.mean((a-b)**2)))

def robfit(x,y,deg):
    x=np.asarray(x,float);y=np.asarray(y,float)
    x0=x[0]; xs=max(x.max()-x.min(),1e-6); xk=(x-x0)/xs
    if len(x)<deg+2: return y.copy()
    c=np.polyfit(xk,y,deg)
    for _ in range(4):
        r=y-np.polyval(c,xk); sc=np.median(np.abs(r))*1.4826+1e-6
        c=np.polyfit(xk,y,deg,w=1.0/(1.0+(r/(2*sc))**2))
    return np.polyval(c,xk)

# ---- PF (for deg2/deg4 projection test) ----
def run_pf(hw,tw,seed,N=150):
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
    return res, ll, ev.index.to_numpy(), z_v, md_v

def ens(hw,tw,n_seeds=10,scale=5.0):
    preds=[];liks=[];meta=None
    for s in range(n_seeds):
        r,ll,ei,zv,mv=run_pf(hw,tw,s);preds.append(r);liks.append(ll);meta=(ei,zv,mv)
    liks=np.array(liks);ln=liks-liks.max();wt=np.exp(ln/scale);wt/=wt.sum()
    return (wt[:,None]*np.stack(preds,0)).sum(0), meta

# ---- constrained DP alignment (DTW-like), raw vs heel-calibrated ----
def dp_align(gr_eval, tw_tvt, tw_gr, start_idx, es=100.0, mc=15.0, maxstep=2):
    n=len(gr_eval); nt=len(tw_gr)
    # DP over typewell index; cost = emission (gr diff)^2/es + move cost
    INF=1e18
    # to keep it feasible, restrict band around a forward-moving expectation is not needed for nt~1300, n~4000 -> nt*n*(2*maxstep+1) can be big
    # use a banded DP: allow index in [start, start + n*maxstep] bounded by nt
    lo=max(0,start_idx-50); hi=min(nt, start_idx + n + 50)  # typical: index advances ~1 per step
    idxs=np.arange(lo,hi); m=len(idxs)
    prev=np.full(m,INF);
    # init: start near start_idx
    prev[:] = ((tw_gr[idxs]-gr_eval[0])**2/es) + np.abs(idxs-start_idx)*mc
    back=np.zeros((n,m),np.int32)
    for i in range(1,n):
        emis=(tw_gr[idxs]-gr_eval[i])**2/es
        cur=np.full(m,INF)
        for d in range(0,maxstep+1):  # forward-only moves 0..maxstep (monotone, geosteering goes forward through strata mostly)
            if d==0:
                cand=prev
                src=np.arange(m)
            else:
                cand=np.full(m,INF); cand[d:]=prev[:-d]; src=np.arange(m)-d
            tot=cand + emis + (0.0 if d==0 else mc*d)
            better=tot<cur
            cur=np.where(better,tot,cur)
            back[i]=np.where(better,src,back[i])
        prev=cur
    # backtrack
    path=np.zeros(n,np.int32); j=int(np.argmin(prev)); path[n-1]=j
    for i in range(n-1,0,-1):
        j=back[i,j]; path[i-1]=j
    return tw_tvt[idxs[path]]

def load(wid):
    return (pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv')),
            pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__typewell.csv')))

NW=60
wells=[]
for wid in well_ids[-NW:]:
    hw,tw=load(wid); km=hw['TVT_input'].notna()
    if km.sum()<30 or (~km).sum()<30: continue
    wells.append((hw,tw,hw['TVT'].values[np.where(~km.values)[0]]))
print('wells:',len(wells),flush=True)

# Test A: deg2 vs deg4 projection (proxy on raw PF)
truth=[]; m_raw=[]; m_d2=[]; m_d4=[]
# Test B: DP align raw vs calibrated
b_true=[]; b_raw=[]; b_cal=[]
t0=time.time()
for k,(hw,tw,true) in enumerate(wells):
    pred,(ei,zv,mv)=ens(hw,tw)
    surf=pred+zv
    truth.append(true); m_raw.append(pred)
    m_d2.append(0.25*pred+0.75*(robfit(mv,surf,2)-zv))
    m_d4.append(0.25*pred+0.75*(robfit(mv,surf,4)-zv))
    # ---- DP alignment ----
    tw_s=tw.sort_values('TVT'); tw_tvt=tw_s['TVT'].values.astype(float); tw_gr=tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn=hw[hw['TVT_input'].notna()]
    kn_gr=kn['GR'].values.astype(float); tw_at_k=np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)
    v=np.isfinite(kn_gr)&np.isfinite(tw_at_k)
    if v.sum()>=20:
        coef=np.polyfit(kn_gr[v],tw_at_k[v],1); a,b=float(coef[0]),float(coef[1])
    else: a,b=1.0,0.0
    gr_e=hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)[np.where(~hw['TVT_input'].notna().values)[0]]
    last_tvt=float(kn['TVT_input'].iloc[-1]); start=int(np.clip(np.searchsorted(tw_tvt,last_tvt),0,len(tw_tvt)-1))
    try:
        raw_align=dp_align(gr_e, tw_tvt, tw_gr, start)
        cal_align=dp_align(gr_e*a+b, tw_tvt, tw_gr, start)
        b_true.append(true); b_raw.append(raw_align); b_cal.append(cal_align)
    except Exception as e:
        pass
    if (k+1)%15==0: print('%d wells, %.0fs'%(k+1,time.time()-t0),flush=True)

truth=np.concatenate(truth)
print('--- Test A: projection degree (proxy on raw PF) ---',flush=True)
print('raw PF          RMSE=%.3f'%prmse(truth,np.concatenate(m_raw)),flush=True)
print('proj deg-4      RMSE=%.3f'%prmse(truth,np.concatenate(m_d4)),flush=True)
print('proj deg-2      RMSE=%.3f'%prmse(truth,np.concatenate(m_d2)),flush=True)
bt=np.concatenate(b_true)
print('--- Test B: DP alignment, raw vs heel-calibrated GR ---',flush=True)
print('DP raw GR       RMSE=%.3f'%prmse(bt,np.concatenate(b_raw)),flush=True)
print('DP calibrated   RMSE=%.3f'%prmse(bt,np.concatenate(b_cal)),flush=True)
print('total %.0fs'%(time.time()-t0),flush=True)
