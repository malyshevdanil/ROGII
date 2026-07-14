"""Expand the trellis-vs-full_combo test to a larger sample (250 wells, not restricted to WARP's true
160-well holdout) for more statistical power in the paired bootstrap. This is legitimate despite the
WARP-contamination lesson: the trellis itself has no train/holdout concept (its priors come from
known-zone stats already treated as legal info everywhere in this project), and we're comparing
full_combo vs full_combo+trellis on the SAME base -- WARP's own memorization bias (if any) affects both
sides of the comparison equally as a shared component, so the DELTA should be largely unaffected.
"""
import numpy as np, pickle, time
from trellis_fwdbwd import run_one as trellis_run_one, load_well as trellis_load_well
from scipy.signal import savgol_filter

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_on_proxy_cache.pkl'
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
WIDS_ALL = [wid for wid in warp_on_proxy if wid in proxy]

rng = np.random.default_rng(99)
sample = list(rng.choice(WIDS_ALL, size=250, replace=False))
print('sample size:', len(sample))

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
t0 = time.time()
for i, wid in enumerate(sample):
    px = proxy[wid]
    r = trellis_run_one(wid)
    if r is None: continue
    pred, true_tvt = r
    r2 = trellis_load_well(wid)
    if r2 is None: continue
    hw, tw_tvt, tw_gr, kn, ev = r2
    md_v = ev.MD.values.astype(np.float64)
    md_proxy = px['md']
    trellis_on_proxy = np.interp(md_proxy, md_v, pred)
    per_well[wid] = dict(trellis=trellis_on_proxy, full_combo=full_combo_track(wid), sp45=px['sp45'], true=px['true'])
    if (i + 1) % 25 == 0:
        print(f'  {i+1}/{len(sample)}  {time.time()-t0:.0f}s')

print(f'\nn wells ok: {len(per_well)}/{len(sample)}  total time: {time.time()-t0:.0f}s')
pickle.dump(per_well, open('trellis_per_well_cache_250.pkl', 'wb'))
print('saved trellis_per_well_cache_250.pkl')
