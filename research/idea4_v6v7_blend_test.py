"""Idea 4: does adding v6 (own-only, 16 feat, no aug) alongside v7 (rich 24 feat, prefix-cut aug) inside
the GRU-refiner blend give extra decorrelated value over v7 alone? Both are the same architecture family
(bidirectional GRU refiner) but different feature sets/training -- test whether that's enough
diversity. Base: v7blend = 0.7*full_combo + 0.3*gru_v7 (current best, real LB 6.742 at this weight,
6.725 real at gru weight 0.35). Test candidate: replace gru_v7 with a blend of (gru_v6, gru_v7).
"""
import numpy as np, pickle
from scipy.signal import savgol_filter
from paired_bootstrap import paired_bootstrap_gain

proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
warp_on_proxy = pickle.load(open('warp_on_proxy_cache.pkl', 'rb'))
gru_v7 = pickle.load(open('gru_v7_kfold_oof_preds.pkl', 'rb'))
gru_v6 = pickle.load(open('gru_v4_kfold_oof_preds.pkl', 'rb'))
VA_WIDS = pickle.load(open('warp_true_holdout_160.pkl', 'rb'))

WIDS = [w for w in gru_v7 if w in proxy and w in warp_on_proxy and w in gru_v6 and w in VA_WIDS]
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

per_well = {}
for w in WIDS:
    px = proxy[w]
    per_well[w] = dict(full_combo=full_combo_track(w), gru_v6=gru_v6[w], gru_v7=gru_v7[w], true=px['true'])

def pooled(pred_dict, wl):
    sq = [(pred_dict[w] - per_well[w]['true']) ** 2 for w in wl]
    return float(np.sqrt(np.mean(np.concatenate(sq))))

def v7blend(wid, gw=0.35):
    d = per_well[wid]
    return (1 - gw) * d['full_combo'] + gw * d['gru_v7']

def v6v7blend(wid, gw, v6_frac):
    """gw = total GRU weight (same as v7blend's gw), v6_frac = fraction of that weight given to v6."""
    d = per_well[wid]
    gru_mix = (1 - v6_frac) * d['gru_v7'] + v6_frac * d['gru_v6']
    return (1 - gw) * d['full_combo'] + gw * gru_mix

base_pred = {w: v7blend(w) for w in WIDS}
print('v7blend (gw=0.35) solo:', pooled(base_pred, WIDS))

print('\n=== v6_frac sweep (gw fixed at 0.35) ===')
for v6f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    pred = {w: v6v7blend(w, 0.35, v6f) for w in WIDS}
    print(f'  v6_frac={v6f:.2f}  pooled={pooled(pred, WIDS):.4f}')

sq_base = {w: (base_pred[w] - per_well[w]['true']) ** 2 for w in WIDS}
print()
for v6f in [0.1, 0.2, 0.3, 0.4, 0.5]:
    cand_pred = {w: v6v7blend(w, 0.35, v6f) for w in WIDS}
    sq_cand = {w: (cand_pred[w] - per_well[w]['true']) ** 2 for w in WIDS}
    mean_gain, lo, hi, frac_pos = paired_bootstrap_gain(sq_base, sq_cand, WIDS, n_boot=3000, seed=0)
    print(f'paired bootstrap (v6_frac={v6f}): mean_gain={mean_gain:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  frac_positive={frac_pos:.3f}')

v7_err = np.concatenate([per_well[w]['gru_v7'] - per_well[w]['true'] for w in WIDS])
v6_err = np.concatenate([per_well[w]['gru_v6'] - per_well[w]['true'] for w in WIDS])
print('\ncorr(gru_v6_err, gru_v7_err):', np.corrcoef(v6_err, v7_err)[0, 1])
