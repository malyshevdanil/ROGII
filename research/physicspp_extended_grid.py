"""kfold_cv_new_best.py checked (a, beta, warm, sm) but A_GRID stopped exactly at 0.30 (== production
value) and warm/degree were never swept at all -- the same "grid-edge truncation" mistake this project
specifically learned to avoid (WARP weight 0.15 looked optimal until the grid was extended to 0.30 and
a real further gain showed up on real LB). This does a coordinate-wise sweep (cheap, avoids the combinatorial
blowup of a full 5-way grid) using the CACHED WARP predictions (warp_on_proxy_cache.pkl, no need to rerun
the net), then a proper 5-fold CV confirmation around whatever the sweep finds.
"""
import numpy as np, pickle
from scipy.signal import savgol_filter

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_on_proxy_cache.pkl'
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
WIDS = [wid for wid in warp_on_proxy if wid in proxy]
print('wells:', len(WIDS))

def robust_poly_fit(x, y, deg, n_iter=4):
    wt = np.ones_like(y)
    coef = np.polyfit(x, y, deg, w=wt)
    for _ in range(n_iter):
        resid = y - np.polyval(coef, x)
        s = np.median(np.abs(resid)) * 1.4826 + 1e-6
        u = resid / (4.685 * s)
        wt = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        coef = np.polyfit(x, y, deg, w=wt + 1e-6)
    return coef

def blended_track(wid, a):
    px = proxy[wid]
    return (1 - a) * px['sp45'] + a * warp_on_proxy[wid]

def physics_pp(wid, base_track, beta, warmup, smooth, deg):
    px = proxy[wid]; md = px['md']; z = px['z']
    U_raw = base_track + z
    coef = robust_poly_fit(md, U_raw, deg)
    U_fit = np.polyval(coef, md)
    md0 = md.min()
    ramp = np.clip((md - md0) / max(warmup, 1e-6), 0, 1) * beta
    U_blend = (1 - ramp) * U_raw + ramp * U_fit
    if smooth >= 5 and smooth % 2 == 1 and len(U_blend) > smooth:
        U_blend = savgol_filter(U_blend, smooth, 2)
    return U_blend - z

def combo_pred(wid, a, beta, warm, sm, deg):
    base = blended_track(wid, a)
    return physics_pp(wid, base, beta, warm, sm, deg)

def pooled(pred_fn, wid_list):
    s = []
    for wid in wid_list:
        p = pred_fn(wid); s.append((p - proxy[wid]['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(s))))

# current production defaults
cur = dict(a=0.30, beta=0.75, warm=500, sm=51, deg=4)

def score(params, wl=WIDS):
    p = dict(cur); p.update(params)
    return pooled(lambda wid, p=p: combo_pred(wid, p['a'], p['beta'], p['warm'], p['sm'], p['deg']), wl)

print('baseline (current production params) full-set:', score({}))

print('\n=== coordinate sweep: a (WARP blend weight), extended past 0.30 ===')
for a in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.85, 1.0]:
    print(f'  a={a:.2f}  pooled={score({"a": a}):.4f}')

print('\n=== coordinate sweep: polyfit degree (never swept before, hardcoded 4) ===')
for deg in [2, 3, 4, 5, 6, 7, 8]:
    print(f'  deg={deg}  pooled={score({"deg": deg}):.4f}')

print('\n=== coordinate sweep: warmup (fixed at 500 before) ===')
for warm in [150, 250, 350, 500, 700, 1000, 1500, 2500]:
    print(f'  warm={warm}  pooled={score({"warm": warm}):.4f}')

print('\n=== coordinate sweep: smooth window, extended past 51 ===')
for sm in [0, 25, 51, 71, 91, 121, 151]:
    print(f'  sm={sm}  pooled={score({"sm": sm}):.4f}')

print('\n=== coordinate sweep: beta (sanity re-check) ===')
for beta in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
    print(f'  beta={beta:.2f}  pooled={score({"beta": beta}):.4f}')
