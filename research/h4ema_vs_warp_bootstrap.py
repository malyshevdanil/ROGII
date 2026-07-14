"""Test the 'architectural diversity' hypothesis from today's post-mortem: does a SECOND, architecturally
different neural net (H4+EMA: 8 multi-scale channels + EMA weights, vs the production WARP's 4-channel raw
weights) blended into the full combo do better than our failed classical-on-classical attempts (beam-HSMM,
fwd-bwd trellis)? Uses the proper paired well-bootstrap (paired_bootstrap.py) instead of a handful of fixed
folds, since that was flagged as a real methodological gap today. Both nets share the identical true
160-well seed-42 holdout (kaggle_h4.py mirrors kaggle_warp.py's split construction), so this is honestly
out-of-sample for both.
"""
import numpy as np, pickle
from scipy.signal import savgol_filter
from paired_bootstrap import paired_bootstrap_gain

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_on_proxy_cache.pkl'
H4_CACHE = 'h4ema_on_proxy_cache.pkl'
HOLDOUT_PATH = 'warp_true_holdout_160.pkl'

proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
h4_on_proxy = pickle.load(open(H4_CACHE, 'rb'))
VA_WIDS = pickle.load(open(HOLDOUT_PATH, 'rb'))
WIDS = [w for w in VA_WIDS if w in proxy and w in warp_on_proxy and w in h4_on_proxy]
print('wells available for all three (proxy, warp, h4ema):', len(WIDS))

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
for wid in WIDS:
    px = proxy[wid]
    per_well[wid] = dict(full_combo=full_combo_track(wid), h4ema=h4_on_proxy[wid],
                          warp=warp_on_proxy[wid], sp45=px['sp45'], true=px['true'])

full_err = np.concatenate([per_well[w]['full_combo'] - per_well[w]['true'] for w in WIDS])
h4_err = np.concatenate([per_well[w]['h4ema'] - per_well[w]['true'] for w in WIDS])
warp_err = np.concatenate([per_well[w]['warp'] - per_well[w]['true'] for w in WIDS])
print('h4ema solo RMSE:', np.sqrt(np.mean(h4_err ** 2)))
print('warp solo RMSE:', np.sqrt(np.mean(warp_err ** 2)))
print('full_combo solo RMSE:', np.sqrt(np.mean(full_err ** 2)))
print('corr(full_combo_err, h4ema_err):', np.corrcoef(full_err, h4_err)[0, 1])
print('corr(warp_err, h4ema_err):', np.corrcoef(warp_err, h4_err)[0, 1], '  (WARP vs H4+EMA -- two neural nets, different features+EMA)')

def pooled(pred_fn, wl):
    sq = [(pred_fn(w) - per_well[w]['true']) ** 2 for w in wl]
    return float(np.sqrt(np.mean(np.concatenate(sq))))

def blend(wid, w):
    d = per_well[wid]
    return (1 - w) * d['full_combo'] + w * d['h4ema']

print('\n=== full-sample sweep: full_combo + h4ema ===')
for w in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    print(f'  w={w:.2f}  pooled={pooled(lambda wid, w=w: blend(wid, w), WIDS):.4f}')

# paired bootstrap at the best-looking weight from the sweep + a modest default
sq_base = {w: (per_well[w]['full_combo'] - per_well[w]['true']) ** 2 for w in WIDS}
for w_test in [0.1, 0.2, 0.3]:
    sq_cand = {w: (blend(w, w_test) - per_well[w]['true']) ** 2 for w in WIDS}
    mean_gain, lo, hi, frac_pos = paired_bootstrap_gain(sq_base, sq_cand, WIDS, n_boot=3000, seed=0)
    print(f'\npaired bootstrap (w={w_test}): mean_gain={mean_gain:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  frac_positive={frac_pos:.3f}')
