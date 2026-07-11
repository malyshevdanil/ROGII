"""Battery3 (clean): is the BANDED-align systematic error low-order (fixable) or high-order (wall)?
+ cheap oracle of the low-pass-surface breakthrough. flat=14.7 line=6.6 WARP=11 align=30 oracle=3.9."""
import sys,os,numpy as np, pandas as pd, time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import prep
data=prep.load(); tr,va=prep.split(data); TRAIN_DIR='data/train'
def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s))))
TG=256
def lp(a,k):
    if len(a)<k or k<3: return a.copy()
    pad=k//2; ap=np.pad(a,pad,mode='edge'); return np.convolve(ap,np.ones(k)/k,mode='valid')[:len(a)]
WELLS=[]
for w in va:
    hw=pd.read_csv(f"{TRAIN_DIR}/{w['wid']}__horizontal_well.csv"); tw=pd.read_csv(f"{TRAIN_DIR}/{w['wid']}__typewell.csv").sort_values('TVT')
    kn=hw['TVT_input'].notna().values; ev=~kn
    if kn.sum()<30 or ev.sum()<30 or len(tw)<10: continue
    Z=hw['Z'].values.astype(float); TVT=hw['TVT'].values.astype(float)
    gr=hw['GR'].interpolate().bfill().ffill().values.astype(float); ti=hw['TVT_input'].values.astype(float)
    tt=tw['TVT'].values.astype(float); tg=tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kg=gr[kn]; twk=np.interp(ti[kn],tt,tg); v=np.isfinite(kg)&np.isfinite(twk)
    a,b=(np.polyfit(kg[v],twk[v],1) if v.sum()>=20 else (1.,0.)); cg=gr*a+b
    g_tvt=np.linspace(tt.min(),tt.max(),TG); g_gr=np.interp(g_tvt,tt,tg)
    WELLS.append(dict(cg=cg,ev=ev,Z=Z,TVT=TVT,g_tvt=g_tvt,g_gr=g_gr,last_tvt=ti[kn][-1],last_Z=Z[kn][-1]))
print('battery3 on %d wells | flat14.7 line6.6 WARP11 align30 oracle3.9\n'%len(WELLS),flush=True)

def banded_align(w,band=60):
    eg=w['cg'][w['ev']]; g_tvt=w['g_tvt']; g_gr=w['g_gr']; m=np.abs(g_tvt-w['last_tvt'])<band
    gt=g_tvt[m]; gg=g_gr[m]
    if len(gt)<3: return np.full(w['ev'].sum(),w['last_tvt'])
    return gt[np.argmin((gg[None,:]-eg[:,None])**2,axis=1)]

t0=time.time()
# 1) banded align at several bands (sanity ~30)
for band in [40,60,100]:
    e=[(banded_align(w,band)-w['TVT'][w['ev']])**2 for w in WELLS]
    print('  banded-align band=%3d -> RMSE %.2f'%(band,prmse(e)),flush=True)
# 2) decompose the banded-align ERROR: remove oracle const / linear / low-pass -> residual
al={id(w):banded_align(w,60) for w in WELLS}
e_raw=[];e_c=[];e_l=[];e_hp=[]
for w in WELLS:
    p=al[id(w)]; y=w['TVT'][w['ev']]; x=np.arange(len(y)); err=p-y
    e_raw.append(err**2)
    e_c.append((err-err.mean())**2)                       # remove const (oracle)
    e_l.append((err-np.polyval(np.polyfit(x,err,1),x))**2) # remove linear (oracle)
    e_hp.append((err-lp(err,61))**2)                       # keep only HF of error (i.e. remove LF)
print('\n  align err: raw=%.2f | -const=%.2f | -linear=%.2f | -lowfreq(HFonly)=%.2f'%(
    prmse(e_raw),prmse(e_c),prmse(e_l),prmse(e_hp)),flush=True)
print('  (if -linear or -lowfreq -> single digit: systematic error is LOW-ORDER/LF => a learned per-well correction can fix it => breakthrough path)',flush=True)
# 3) cheap oracle of low-pass-surface breakthrough on banded align
for k in [31,61,121]:
    e=[]
    for w in WELLS:
        p=al[id(w)]; ze=w['Z'][w['ev']]; surf=p+ze; e.append((lp(surf,k)-ze-w['TVT'][w['ev']])**2)
    print('  lowpass-surface(align) k=%3d -> RMSE %.2f'%(k,prmse(e)),flush=True)
print('\nDONE battery3 %.0fs'%(time.time()-t0),flush=True)
