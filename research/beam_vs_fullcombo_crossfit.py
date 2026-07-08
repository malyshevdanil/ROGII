"""Does the beam-HSMM decoder decorrelate from the FULL current-best combo (sp45 + WARP-blend(0.30) +
physics-pp), not just raw sp45? Earlier test (beam_crossfit_test.py) blended against sp45 alone and found
no real cross-fit gain. This is a stricter, more relevant bar since it's the closest local proxy to what
we'd actually submit. Also checks the two EXISTING cached tracks in proxy.pkl ('beam', 'pf8') that were
never explicitly cross-fit-blend-tested against the full combo in this session's history.
"""
import numpy as np, pickle, time
from beam_decoder_v1 import run_one as hsmm_run_one, load_well as hsmm_load_well
from scipy.signal import savgol_filter

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_on_proxy_cache.pkl'
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
WIDS = [wid for wid in warp_on_proxy if wid in proxy]

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

rng = np.random.default_rng(11)
sample = list(rng.choice(WIDS, size=50, replace=False))

per_well = {}
t0 = time.time()
for i, wid in enumerate(sample):
    px = proxy[wid]
    r = hsmm_run_one(wid, seed=1, K=800, G=9, weighted=False)
    if r is None: continue
    pts, true_tvt = r
    r2 = hsmm_load_well(wid)
    if r2 is None: continue
    hw, tw_tvt, tw_gr, kn, ev = r2
    md_v = ev.MD.values.astype(np.float64)
    md_proxy = px['md']
    beam_on_proxy = np.interp(md_proxy, md_v, pts)
    per_well[wid] = dict(
        hsmm=beam_on_proxy,
        full_combo=full_combo_track(wid),
        beam_cached=px['beam'],
        pf8=px['pf8'],
        true=px['true'],
    )
    if (i + 1) % 10 == 0:
        print(f'  {i+1}/{len(sample)}  {time.time()-t0:.0f}s')

print(f'\nn wells ok: {len(per_well)}/{len(sample)}  total time: {time.time()-t0:.0f}s')
OK_WIDS = list(per_well.keys())

def pooled(pred_key_or_fn, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        pred = pred_key_or_fn(d) if callable(pred_key_or_fn) else d[pred_key_or_fn]
        sq.append((pred - d['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

print('\n=== solo quality on this sample ===')
for k in ['full_combo', 'hsmm', 'beam_cached', 'pf8']:
    print(f'  {k}: {pooled(k, OK_WIDS):.4f}')

full_err = np.concatenate([per_well[w]['full_combo'] - per_well[w]['true'] for w in OK_WIDS])
for other in ['hsmm', 'beam_cached', 'pf8']:
    other_err = np.concatenate([per_well[w][other] - per_well[w]['true'] for w in OK_WIDS])
    print(f'corr(full_combo_err, {other}_err):', np.corrcoef(full_err, other_err)[0, 1])

def pooled_blend(other_key, w, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        pred = (1 - w) * d['full_combo'] + w * d[other_key]
        sq.append((pred - d['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

W_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

def best_w_on(other_key, wl):
    best = (1e9, None)
    for w in W_GRID:
        v = pooled_blend(other_key, w, wl)
        if v < best[0]: best = (v, w)
    return best

rng2 = np.random.default_rng(13)
order = np.array(OK_WIDS); rng2.shuffle(order)
H1, H2 = list(order[:len(order)//2]), list(order[len(order)//2:])

for other in ['hsmm', 'beam_cached', 'pf8']:
    print(f'\n=== blend full_combo + {other}: full-sample sweep ===')
    for w in W_GRID:
        print(f'  w={w:.2f}  pooled={pooled_blend(other, w, OK_WIDS):.4f}')
    print(f'--- 2-fold cross-fit ({other}) ---')
    b1 = best_w_on(other, H1)
    base_h2 = pooled_blend(other, 0.0, H2)
    v_h2 = pooled_blend(other, b1[1], H2)
    print(f'fit H1: best w={b1[1]} (train={b1[0]:.4f}) -> H2 test={v_h2:.4f} vs baseline={base_h2:.4f}')
    b2 = best_w_on(other, H2)
    base_h1 = pooled_blend(other, 0.0, H1)
    v_h1 = pooled_blend(other, b2[1], H1)
    print(f'fit H2: best w={b2[1]} (train={b2[0]:.4f}) -> H1 test={v_h1:.4f} vs baseline={base_h1:.4f}')
