"""Battery v2: correct DTW + ORACLE-region-ceiling (how much would resolving the systematic per-well
error help?) + eval spectral phase. flat=14.7 line=6.6 WARP=11 align=30."""
import sys,os,numpy as np, pandas as pd, time, traceback
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import prep
data=prep.load(); tr,va=prep.split(data); TRAIN_DIR='data/train'
def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s)))) if s else float('nan')
TG=256; L=380
def load(w):
    hw=pd.read_csv(f"{TRAIN_DIR}/{w['wid']}__horizontal_well.csv"); tw=pd.read_csv(f"{TRAIN_DIR}/{w['wid']}__typewell.csv").sort_values('TVT')
    kn=hw['TVT_input'].notna().values; ev=~kn
    if kn.sum()<30 or ev.sum()<30 or len(tw)<10: return None
    Z=hw['Z'].values.astype(float); TVT=hw['TVT'].values.astype(float)
    gr=hw['GR'].interpolate().bfill().ffill().values.astype(float); ti=hw['TVT_input'].values.astype(float)
    tt=tw['TVT'].values.astype(float); tg=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kg=gr[kn]; twk=np.interp(ti[kn],tt,tg); v=np.isfinite(kg)&np.isfinite(twk)
    a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.)); cg=gr*a+b
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg)
    return dict(cg=cg,kn=kn,ev=ev,Z=Z,TVT=TVT,ti=ti,g_tvt=g_tvt,g_gr=g_gr,last_tvt=ti[kn][-1])
WELLS=[x for x in (load(w) for w in va) if x]
print('battery2 on %d wells | flat14.7 line6.6 WARP11 align30\n'%len(WELLS),flush=True)
def run(n,f):
    t=time.time()
    try: print('[%-24s] %s (%.0fs)'%(n,f(),time.time()-t),flush=True)
    except Exception as e: print('[%-24s] ERR %r'%(n,e)); traceback.print_exc()

def perpoint_align(w):  # baseline per-point argmax within a wide band -> gives ~30
    eg=w['cg'][w['ev']]; g_tvt=w['g_tvt']; g_gr=w['g_gr']
    return np.array([g_tvt[np.argmin((g_gr-v)**2)] for v in eg])

# T5: ORACLE region — take per-point align, then shift the whole EVAL prediction to the true mean (removes
# a per-well CONSTANT systematic offset). If RMSE drops a lot -> the systematic error is a constant (fixable).
def t5_oracle_const():
    e0=[];e1=[]
    for w in WELLS:
        p=perpoint_align(w); y=w['TVT'][w['ev']]
        e0.append((p-y)**2); e1.append((p-(p.mean()-y.mean())-y)**2)  # oracle-correct the mean
    return 'align=%.2f | align+oracle-mean-shift=%.2f (gap => systematic const offset share)'%(prmse(e0),prmse(e1))

# T6: ORACLE region+slope — remove per-well best linear (offset+slope) of the align error (oracle)
def t6_oracle_linear():
    e=[]
    for w in WELLS:
        p=perpoint_align(w); y=w['TVT'][w['ev']]; x=np.arange(len(y)); err=p-y
        c=np.polyfit(x,err,1); e.append((p-np.polyval(c,x)-y)**2)   # remove oracle linear trend of error
    return 'align + oracle-remove-linear-error=%.2f (if ~single digit => error is low-order/fixable)'%prmse(e)

# T1b: correct subsequence DTW (monotone up/diag/left) align full GR -> typewell
def dtw_path(seq,ref):
    n=len(seq); m=len(ref); cost=(seq[:,None]-ref[None,:])**2
    D=np.full((n,m),1e18); B=np.zeros((n,m),np.uint8); D[0]=cost[0]
    for i in range(1,n):
        Dm1=D[i-1]; ci=cost[i]; cur=D[i]
        cur[0]=ci[0]+Dm1[0]; B[i,0]=1
        for j in range(1,m):
            up=Dm1[j]; dg=Dm1[j-1]; lf=cur[j-1]; b=0; mn=dg
            if up<mn: mn=up; b=1
            if lf<mn: mn=lf; b=2
            cur[j]=ci[j]+mn; B[i,j]=b
    j=int(np.argmin(D[-1])); path=np.zeros(n,np.int32); i=n-1; path[i]=j
    while i>0 or j>0:
        b=B[i,j]
        if b==0: i-=1; j-=1
        elif b==1: i-=1
        else: j-=1
        if i>=0: path[i]=j
        if i<=0 and j<=0: break
    return path
def t1b_dtw():
    e=[]
    for w in WELLS[:80]:
        cg=w['cg']; g_tvt=w['g_tvt']; g_gr=w['g_gr']
        idx=np.linspace(0,len(cg)-1,L).astype(int); seq=cg[idx]
        gs=g_gr.std()+1e-6; path=dtw_path(seq/gs,g_gr/gs); tvt=g_tvt[path]
        evm=w['ev'][idx]; e.append((tvt[evm>0]-w['TVT'][idx][evm>0])**2)
    return 'DTW-align=%.2f (vs per-point 30; <30 => global consistency helps)'%prmse(e)

run('T5 oracle const-shift', t5_oracle_const)
run('T6 oracle linear-error', t6_oracle_linear)
run('T1b correct DTW', t1b_dtw)
print('\nDONE battery2',flush=True)
