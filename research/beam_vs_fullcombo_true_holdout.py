"""Clean re-run of beam_vs_fullcombo_crossfit.py restricted to the TRUE 160-well WARP holdout
(warp_true_holdout_160.pkl), since 'full_combo' includes the WARP blend and the earlier test sampled
50 random wells from the full 773 (79% memorized by WARP's own training) -- that made full_combo look
artificially strong there, which could have masked a real blend gain from beam-HSMM. This is the clean,
final answer.
"""
import numpy as np, pickle, time
from beam_decoder_v1 import run_one as hsmm_run_one, load_well as hsmm_load_well
from scipy.signal import savgol_filter

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_on_proxy_cache.pkl'
HOLDOUT_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_true_holdout_160.pkl'
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
VA_WIDS = pickle.load(open(HOLDOUT_PATH, 'rb'))
WIDS = [wid for wid in warp_on_proxy if wid in proxy and wid in VA_WIDS]
print('true-holdout wells available:', len(WIDS))

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
for i, wid in enumerate(WIDS):
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
    per_well[wid] = dict(hsmm=beam_on_proxy, full_combo=full_combo_track(wid), true=px['true'])
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(WIDS)}  {time.time()-t0:.0f}s')

print(f'\nn wells ok: {len(per_well)}/{len(WIDS)}  total time: {time.time()-t0:.0f}s')
OK_WIDS = list(per_well.keys())

def pooled(key_or_fn, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        pred = key_or_fn(d) if callable(key_or_fn) else d[key_or_fn]
        sq.append((pred - d['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

print('\nfull_combo solo (true holdout):', pooled('full_combo', OK_WIDS))
print('hsmm solo (true holdout):', pooled('hsmm', OK_WIDS))
full_err = np.concatenate([per_well[w]['full_combo'] - per_well[w]['true'] for w in OK_WIDS])
hsmm_err = np.concatenate([per_well[w]['hsmm'] - per_well[w]['true'] for w in OK_WIDS])
print('corr(full_combo_err, hsmm_err):', np.corrcoef(full_err, hsmm_err)[0, 1])

def pooled_blend(w, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        pred = (1 - w) * d['full_combo'] + w * d['hsmm']
        sq.append((pred - d['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

W_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
print('\n=== full true-holdout sweep ===')
for w in W_GRID:
    print(f'  w={w:.2f}  pooled={pooled_blend(w, OK_WIDS):.4f}')

def best_w_on(wl):
    best = (1e9, None)
    for w in W_GRID:
        v = pooled_blend(w, wl)
        if v < best[0]: best = (v, w)
    return best

K = 4
rng = np.random.default_rng(13)
order = np.array(OK_WIDS); rng.shuffle(order)
folds = np.array_split(order, K)
print(f'\n=== {K}-fold cross-fit within true holdout ===')
gains = []
for i in range(K):
    test_wl = folds[i]; train_wl = np.concatenate([folds[j] for j in range(K) if j != i])
    base = pooled_blend(0.0, test_wl)
    _, w_star = best_w_on(train_wl)
    v = pooled_blend(w_star, test_wl)
    gains.append(base - v)
    print(f'fold {i}: n={len(test_wl)}  baseline={base:.4f}  w={w_star}  test={v:.4f}  gain={base-v:+.4f}')
print('mean gain:', np.mean(gains), 'positive in', sum(g>0 for g in gains), f'/{K} folds')
