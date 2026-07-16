"""Stage 4: does track6 (GBM-residual-on-pf_ancc, Stage 3b) add real blend value on top of our current
production best (v7blend = 0.7*full_combo + 0.3*gru_v7, real LB 6.742)? Check decorrelation first
(the first gbm_track6_style attempt failed here: corr 0.83-0.88, too high because it was framed as a
residual FROM sp45 itself; this version is built on pf_ancc, a genuinely independent backbone -- Stage 1
found raw pf_ancc has corr 0.70 with v7blend, better but not proven sufficient), then paired bootstrap
on the TRUE 160-well holdout (only true-holdout rows are used for GBM oof here since GroupKFold OOF
predictions are valid for every well including holdout ones).
"""
import numpy as np, pickle
from scipy.signal import savgol_filter
from paired_bootstrap import paired_bootstrap_gain

proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
warp_on_proxy = pickle.load(open('warp_on_proxy_cache.pkl', 'rb'))
gru_v7 = pickle.load(open('gru_v7_kfold_oof_preds.pkl', 'rb'))
VA_WIDS = pickle.load(open('warp_true_holdout_160.pkl', 'rb'))

stage3b = pickle.load(open('track6_stage6_oof.pkl', 'rb'))
feat = pickle.load(open('track6_stage6_features.pkl', 'rb'))
DATA2 = feat['DATA']

WIDS = [w for w in gru_v7 if w in proxy and w in warp_on_proxy and w in VA_WIDS and w in DATA2]
print('n wells (true holdout, all present):', len(WIDS))

def robust_poly_fit(x, y, deg=4, n_iter=4):
    wt = np.ones_like(y)
    coef = np.polyfit(x, y, deg, w=wt)
    for _ in range(n_iter):
        resid = y - np.polyval(coef, x)
        s = np.median(np.abs(resid)) * 1.4826 + 1e-6
        u = resid / (4.685 * s)
        wt = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        coef = np.polyfit(x, y, deg, w=wt + 1e-6)
    return coef

def physics_pp(wid, base_track, beta=0.75, warmup=500, smooth=51):
    px = proxy[wid]; md = px['md']; z = px['z']
    U_raw = base_track + z
    coef = robust_poly_fit(md, U_raw, deg=4)
    U_fit = np.polyval(coef, md)
    md0 = md.min()
    ramp = np.clip((md - md0) / max(warmup, 1e-6), 0, 1) * beta
    U_blend = (1 - ramp) * U_raw + ramp * U_fit
    if smooth >= 5 and smooth % 2 == 1 and len(U_blend) > smooth:
        U_blend = savgol_filter(U_blend, smooth, 2)
    return U_blend - z

def full_combo_track(wid):
    px = proxy[wid]
    base = (1 - 0.30) * px['sp45'] + 0.30 * warp_on_proxy[wid]
    return physics_pp(wid, base)

# reconstruct track6 per-well abs predictions by slicing stage3b's oof_abs using the same per-well
# ordering/lengths as track6_stage2_features.pkl (WIDS order there defines the slice offsets)
s3b_WIDS = stage3b['WIDS']
oof_abs = stage3b['oof_abs']
start = 0
track6_pred = {}
for wid in s3b_WIDS:
    n = len(DATA2[wid]['true'])
    if wid in WIDS:
        track6_pred[wid] = oof_abs[start:start + n]
    start += n

per_well = {}
for w in WIDS:
    px = proxy[w]
    v7blend = 0.7 * full_combo_track(w) + 0.3 * gru_v7[w]
    per_well[w] = dict(v7blend=v7blend, track6=track6_pred[w], true=px['true'])

def pooled(key, wl):
    sq = [(per_well[w][key] - per_well[w]['true']) ** 2 for w in wl]
    return float(np.sqrt(np.mean(np.concatenate(sq))))

print('v7blend solo:', pooled('v7blend', WIDS))
print('track6 solo:', pooled('track6', WIDS))

v7_err = np.concatenate([per_well[w]['v7blend'] - per_well[w]['true'] for w in WIDS])
t6_err = np.concatenate([per_well[w]['track6'] - per_well[w]['true'] for w in WIDS])
print('corr(v7blend_err, track6_err):', np.corrcoef(v7_err, t6_err)[0, 1])

def blend(wid, w):
    d = per_well[wid]
    return (1 - w) * d['v7blend'] + w * d['track6']

print('\n=== full-sample sweep ===')
for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    sq = [(blend(wid, w) - per_well[wid]['true']) ** 2 for wid in WIDS]
    print(f'  w={w:.2f}  pooled={np.sqrt(np.mean(np.concatenate(sq))):.4f}')

sq_base = {w: (per_well[w]['v7blend'] - per_well[w]['true']) ** 2 for w in WIDS}
print()
for w_test in [0.05, 0.1, 0.15, 0.2, 0.3]:
    sq_cand = {w: (blend(w, w_test) - per_well[w]['true']) ** 2 for w in WIDS}
    mean_gain, lo, hi, frac_pos = paired_bootstrap_gain(sq_base, sq_cand, WIDS, n_boot=3000, seed=0)
    print(f'paired bootstrap (w={w_test}): mean_gain={mean_gain:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  frac_positive={frac_pos:.3f}')
