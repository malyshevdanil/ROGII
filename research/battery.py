"""New-angle battery: does GLOBAL alignment (not per-point) resolve the systematic per-well region error?
Cheap decisive premise-tests, run autonomously. flat=14.7, line-oracle=6.6, WARP=11, per-point-align=30."""
import sys,os,numpy as np, pandas as pd, time, traceback
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import prep
data=prep.load(); tr,va=prep.split(data); TRAIN_DIR='data/train'
def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s)))) if s else float('nan')
TG=256; L=420; KSTEPS=60; EVSTEPS=360
def load(w):
    hw=pd.read_csv(f"{TRAIN_DIR}/{w['wid']}__horizontal_well.csv"); tw=pd.read_csv(f"{TRAIN_DIR}/{w['wid']}__typewell.csv").sort_values('TVT')
    kn=hw['TVT_input'].notna().values; ev=~kn
    if kn.sum()<30 or ev.sum()<30 or len(tw)<10: return None
    MD=hw['MD'].values.astype(float); Z=hw['Z'].values.astype(float); TVT=hw['TVT'].values.astype(float)
    gr=hw['GR'].interpolate().bfill().ffill().values.astype(float); ti=hw['TVT_input'].values.astype(float)
    tt=tw['TVT'].values.astype(float); tg=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kg=gr[kn]; twk=np.interp(ti[kn],tt,tg); v=np.isfinite(kg)&np.isfinite(twk)
    a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.)); cg=gr*a+b   # calibrated GR
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg)
    return dict(cg=cg,kn=kn,ev=ev,MD=MD,Z=Z,TVT=TVT,ti=ti,g_tvt=g_tvt,g_gr=g_gr,
                last_tvt=ti[kn][-1],last_Z=Z[kn][-1])
WELLS=[x for x in (load(w) for w in va) if x]
print('battery on %d val wells | flat14.7 line6.6 WARP11 align30\n'%len(WELLS),flush=True)

def run(name,fn):
    t=time.time();
    try:
        r=fn(); print('[%-26s] %s  (%.0fs)'%(name,r,time.time()-t),flush=True)
    except Exception as e:
        print('[%-26s] ERR %s'%(name,repr(e))); traceback.print_exc()

# ---- T2: GLOBAL cross-correlation registration (whole known GR vs typewell -> best region offset) ----
def t2_global_reg():
    e=[]
    for w in WELLS:
        cg=w['cg']; kn=w['kn']; ev=w['ev']; g_tvt=w['g_tvt']; g_gr=w['g_gr']
        # known calibrated GR at known TVT positions; slide against typewell to find best global TVT shift
        kt=w['ti'][kn]; kgr=cg[kn]
        # candidate global shifts of the known TVT -> match GR
        shifts=np.linspace(-60,60,121); best=None
        for s in shifts:
            eg=np.interp(kt+s,g_tvt,g_gr); c=-np.mean((kgr-eg)**2)
            if best is None or c>best[0]: best=(c,s)
        s=best[1]   # global region shift (should be ~0 if calibration perfect)
        # apply: eval TVT = flat(last) but region-corrected... predict via continuity from shifted anchor
        # simplest: TVT_pred = last_tvt + (shift already in calibration) -> flat with corrected level
        pred=(w['last_tvt']+s)-0*w['Z'][ev]  # region-shifted flat
        # better: keep flat but this tests if global shift ~ true region error
        e.append((pred-w['TVT'][ev])**2)
    return 'region-shifted-flat RMSE=%.2f (vs flat 14.7 -> global reg %s)'%(prmse(e),'helps' if prmse(e)<14.7 else 'no')

# ---- T1: classical DTW align (calibrated GR resampled) to typewell -> TVT ----
def dtw_path(cost):  # cost (n,m) -> monotone path indices j for each i
    n,m=cost.shape; INF=1e18; D=np.full((n,m),INF); D[0]=np.cumsum(cost[0]); D[:,0]=np.cumsum(cost[:,0])
    bp=np.zeros((n,m),np.int8)
    for i in range(1,n):
        prev=D[i-1]; row=cost[i]; cur=D[i]
        # allow j to stay or advance (monotone); vectorized min over (i-1,j),(i-1,j-1),(i,j-1)
        d_up=prev.copy()
        d_diag=np.empty(m); d_diag[0]=INF; d_diag[1:]=prev[:-1]
        cur[0]=row[0]+prev[0]
        for j in range(1,m):
            a=cur[j-1]; c=d_up[j]; d=d_diag[j]; mn=min(a,c,d); cur[j]=row[j]+mn
        D[i]=cur
    # backtrack
    j=int(np.argmin(D[-1])); path=np.zeros(n,np.int32); path[-1]=j
    for i in range(n-1,0,-1):
        c=D[i-1,j]; dd=D[i-1,j-1] if j>0 else INF; a=D[i,j-1] if j>0 else INF
        if dd<=c and dd<=a and j>0: j=j-1
        elif a<=c and j>0: pass  # stay i? this simple bt: move to j-1
        path[i-1]=j
    return path
def t1_dtw():
    e=[]; cnt=0
    for w in WELLS:
        cg=w['cg']; ev=w['ev']; g_tvt=w['g_tvt']; g_gr=w['g_gr']
        # resample the full calibrated GR to L points
        idx=np.linspace(0,len(cg)-1,L).astype(int); seq=cg[idx]
        cost=(seq[:,None]-g_gr[None,:])**2/ (g_gr.std()**2+1e-6)
        path=dtw_path(cost); tvt_full=g_tvt[path]                  # TVT per resampled point
        # map back to eval points
        evmask=w['ev'][idx]; e.append((tvt_full[evmask>0]-w['TVT'][idx][evmask>0])**2)
        cnt+=1
        if cnt>=80: break   # 80 wells for speed
    return 'DTW-align RMSE=%.2f (vs per-point-align 30; <30 => global consistency helps)'%prmse(e)

# ---- T3: is the true region in top-K global matches? (multi-hypothesis hedge potential) ----
def t3_topk():
    hits={1:0,3:0,5:0}; n=0
    for w in WELLS:
        cg=w['cg']; kn=w['kn']; g_tvt=w['g_tvt']; g_gr=w['g_gr']; kt=w['ti'][kn]; kgr=cg[kn]
        shifts=np.linspace(-80,80,161); sc=np.array([-np.mean((kgr-np.interp(kt+s,g_tvt,g_gr))**2) for s in shifts])
        order=shifts[np.argsort(-sc)]
        for K in hits:
            if np.min(np.abs(order[:K]))<10: hits[K]+=1   # true region ~ shift 0
        n+=1
    return 'true-region in top-K (K=1/3/5): %.0f%%/%.0f%%/%.0f%% (hedge potential if top3>>top1)'%(100*hits[1]/n,100*hits[3]/n,100*hits[5]/n)

# ---- T4: known-zone GR->TVT nonparametric map + flat continuity ----
def t4_knownmap():
    e=[]
    for w in WELLS:
        kn=w['kn']; ev=w['ev']; cg=w['cg']; g_tvt=w['g_tvt']; g_gr=w['g_gr']
        # map eval GR -> nearest typewell TVT within a continuity band around last_tvt
        eg=cg[ev]; band=(np.abs(g_tvt-w['last_tvt'])<50)
        gt=g_tvt[band]; gg=g_gr[band]
        pred=np.array([gt[np.argmin(np.abs(gg-v))] for v in eg]) if band.sum()>3 else np.full(ev.sum(),w['last_tvt'])
        e.append((pred-w['TVT'][ev])**2)
    return 'known-band GR->TVT nearest RMSE=%.2f'%prmse(e)

run('T2 global-registration', t2_global_reg)
run('T3 top-K region hedge', t3_topk)
run('T4 known-band GR->TVT', t4_knownmap)
run('T1 DTW global align', t1_dtw)
print('\nDONE battery',flush=True)
