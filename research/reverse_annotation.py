import numpy as np, pandas as pd, glob, os

DATA_DIR='data'; TRAIN_DIR=os.path.join(DATA_DIR,'train')
well_ids=sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR,'*__horizontal_well.csv'))})

def pl_fit_resid(md, y, nseg):
    # continuous piecewise-linear fit with nseg segments (equal MD breakpoints), return residual RMSE
    x=np.asarray(md,float); yy=np.asarray(y,float); n=len(x)
    if n<nseg*3: return 0.0
    bps=np.linspace(x.min(),x.max(),nseg+1)
    cols=[np.ones(n),x]+[np.maximum(x-k,0) for k in bps[1:-1]]
    Xd=np.stack(cols,1)
    coef,*_=np.linalg.lstsq(Xd,yy,rcond=None)
    return float(np.sqrt(np.mean((yy-Xd@coef)**2)))

FORMS=['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA']
print('=== Is true TVT piecewise-linear (human-drawn straight lines)? ===')
print('Per-well: residual RMSE (ft) of fitting the TRUE TVT vs MD with k straight segments')
print(f"{'k segments':>12}: {'mean resid':>10} {'median':>8} {'p90':>8}   (if ~0 for small k -> straight lines)")
resid_by_k={k:[] for k in [1,2,3,4,6,8]}
d2_sparsity=[]  # fraction of |2nd diff| below tiny threshold
for wid in well_ids[:200]:
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    if 'TVT' not in hw.columns: continue
    md=hw['MD'].values.astype(float); tvt=hw['TVT'].values.astype(float)
    if len(md)<50: continue
    for k in resid_by_k:
        resid_by_k[k].append(pl_fit_resid(md,tvt,k))
    d2=np.diff(tvt,2)
    d2_sparsity.append(float(np.mean(np.abs(d2)<0.01)))
for k in sorted(resid_by_k):
    r=np.array(resid_by_k[k]); print(f"{k:>12}: {r.mean():>10.3f} {np.median(r):>8.3f} {np.percentile(r,90):>8.3f}")
print()
print('2nd-difference sparsity: fraction of points with |d2(TVT)|<0.01 ft = %.1f%%'%(100*np.mean(d2_sparsity)))
print('  (high % -> long straight segments with few breakpoints = piecewise-linear)')

# Where are the breakpoints? Do they align with formation-column changes?
print('\n=== Do dip-change points align with formation structure? ===')
# detect breakpoints as local maxima of |2nd diff|, check if formation columns change slope there too
align=[]
for wid in well_ids[:150]:
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    if 'TVT' not in hw.columns or 'ANCC' not in hw.columns: continue
    tvt=hw['TVT'].values.astype(float); md=hw['MD'].values.astype(float)
    if len(md)<200: continue
    d2=np.abs(np.diff(tvt,2))
    # top breakpoints
    thr=np.percentile(d2,99.5)
    bp=np.where(d2>thr)[0]
    if len(bp)==0: continue
    # is ANCC surface (TVT+Z) also piecewise-linear? check its d2 correlates
    surf=tvt+hw['Z'].values.astype(float)
    d2s=np.abs(np.diff(surf,2))
    # correlation of breakpoint locations between tvt and surface
    c=np.corrcoef(d2,d2s)[0,1] if len(d2)==len(d2s) else np.nan
    align.append(c)
print('corr(|d2 TVT|, |d2 surface(TVT+Z)|) mean = %.3f  (high -> breaks are structural, shared)'%np.nanmean(align))

# Quantization check: are dip slopes "round"?
print('\n=== Are segment dips quantized (human round numbers)? ===')
slopes=[]
for wid in well_ids[:200]:
    hw=pd.read_csv(os.path.join(TRAIN_DIR,f'{wid}__horizontal_well.csv'))
    if 'TVT' not in hw.columns: continue
    md=hw['MD'].values.astype(float); tvt=hw['TVT'].values.astype(float)
    dd=np.diff(tvt)/np.maximum(np.diff(md),1e-6)
    # local slope over 50-ft windows
    for i in range(0,len(dd)-50,50):
        slopes.append(np.median(dd[i:i+50]))
slopes=np.array(slopes)
print('local dip (dTVT/dMD) distribution: median=%.4f'%np.median(np.abs(slopes)))
# check clustering near round values
for v in [0.0,0.01,0.02,0.05,0.1]:
    print('  within 0.005 of %.3f: %.1f%%'%(v,100*np.mean(np.abs(np.abs(slopes)-v)<0.005)))
