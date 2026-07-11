"""Measure the GR<->typewell coherence spectrum on OUR data, section-resolved (heel vs toe vs full),
replicating the methodology described in a competitor writeup (Welch, 1-ft sampling, nperseg=256,
noverlap=128). This is an independent measurement on our own 773 wells, not copied from anyone."""
import pandas as pd, numpy as np, glob, os
from scipy.signal import coherence, welch

TR = 'd:/ROGII/data/train'
wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TR}/*__horizontal_well.csv')})

def inan(a):
    a = a.copy(); m = np.isnan(a); i = np.arange(len(a))
    if m.all(): return np.zeros(len(a))
    a[m] = np.interp(i[m], i[~m], a[~m]); return a

FS = 1.0  # 1 sample per foot
NPERSEG = 256
bands = {'125ft': 1/125, '60ft': 1/60, '32ft': 1/32, '14ft': 1/14, '8ft': 1/8, '5ft': 1/5}

def coh_at_bands(x, y):
    if len(x) < NPERSEG or len(y) < NPERSEG: return None
    f, Cxy = coherence(x, y, fs=FS, nperseg=min(NPERSEG, len(x)), noverlap=min(NPERSEG, len(x))//2)
    out = {}
    for name, target_f in bands.items():
        idx = np.argmin(np.abs(f - target_f))
        out[name] = Cxy[idx]
    return out

def corr_at_true(gr, tw_at_true):
    v = np.isfinite(gr) & np.isfinite(tw_at_true)
    if v.sum() < 30: return np.nan
    return np.corrcoef(gr[v], tw_at_true[v])[0, 1]

results = {'full': [], 'heel': [], 'toe': []}
corrs = {'full': [], 'heel': [], 'toe': []}
n_used = 0
for wid in wids:
    try:
        hw = pd.read_csv(f'{TR}/{wid}__horizontal_well.csv')
        tw = pd.read_csv(f'{TR}/{wid}__typewell.csv').sort_values('TVT')
    except Exception:
        continue
    if 'TVT' not in hw.columns: continue
    tt = tw['TVT'].values.astype(float); tg = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt) < 10: continue
    gr = inan(hw['GR'].values.astype(float))
    true_tvt = hw['TVT'].values.astype(float)
    tw_at_true = np.interp(true_tvt, tt, tg)   # typewell GR looked up at the TRUE depth (ground truth, train only)
    ev_mask = hw['TVT_input'].isna().values
    kn_mask = ~ev_mask
    if kn_mask.sum() < NPERSEG or ev_mask.sum() < NPERSEG: continue

    c_full = coh_at_bands(gr, tw_at_true)
    c_heel = coh_at_bands(gr[kn_mask], tw_at_true[kn_mask])
    c_toe = coh_at_bands(gr[ev_mask], tw_at_true[ev_mask])
    if c_full is None or c_heel is None or c_toe is None: continue
    results['full'].append(c_full); results['heel'].append(c_heel); results['toe'].append(c_toe)
    corrs['full'].append(corr_at_true(gr, tw_at_true))
    corrs['heel'].append(corr_at_true(gr[kn_mask], tw_at_true[kn_mask]))
    corrs['toe'].append(corr_at_true(gr[ev_mask], tw_at_true[ev_mask]))
    n_used += 1

print('wells used:', n_used)
print()
print('=== correlation(GR, typewell@true depth), section-resolved ===')
for k in ['full', 'heel', 'toe']:
    print(f'  {k:5s}: median corr = {np.nanmedian(corrs[k]):.3f}   mean = {np.nanmean(corrs[k]):.3f}')
print()
print('=== median magnitude-squared coherence by wavelength band, section-resolved ===')
print(f'{"band":8s} {"full":>8s} {"heel":>8s} {"toe":>8s}')
for band in bands:
    f_ = np.median([r[band] for r in results['full']])
    h_ = np.median([r[band] for r in results['heel']])
    t_ = np.median([r[band] for r in results['toe']])
    print(f'{band:8s} {f_:8.3f} {h_:8.3f} {t_:8.3f}')
