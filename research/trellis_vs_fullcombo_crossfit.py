"""Does the forward-backward trellis (weak solo, ~28-30 on a quick sample vs the competitor's own
13.42) still add blend value against our full production combo, the way the competitor's equally-weak
trellis added value against THEIR strong ensemble (13.42 alone, but 7.762->7.699 blended)? Solo quality
matters less than error decorrelation for this question -- test directly rather than keep tuning solo.
"""
import numpy as np, pickle, time
from trellis_fwdbwd import run_one as trellis_run_one, load_well as trellis_load_well
from scipy.signal import savgol_filter

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_on_proxy_cache.pkl'
HOLDOUT_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_true_holdout_160.pkl'
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
VA_WIDS = pickle.load(open(HOLDOUT_PATH, 'rb'))
WIDS = [wid for wid in warp_on_proxy if wid in proxy and wid in VA_WIDS]

rng = np.random.default_rng(21)
sample = list(rng.choice(WIDS, size=60, replace=False))
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
    if (i + 1) % 10 == 0:
        print(f'  {i+1}/{len(sample)}  {time.time()-t0:.0f}s')

print(f'\nn wells ok: {len(per_well)}/{len(sample)}  total time: {time.time()-t0:.0f}s')
pickle.dump(per_well, open('trellis_per_well_cache.pkl', 'wb'))
OK_WIDS = list(per_well.keys())

def pooled(key_or_fn, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        pred = key_or_fn(d) if callable(key_or_fn) else d[key_or_fn]
        sq.append((pred - d['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

print('\ntrellis solo:', pooled('trellis', OK_WIDS))
print('sp45 solo:', pooled('sp45', OK_WIDS))
print('full_combo solo:', pooled('full_combo', OK_WIDS))

full_err = np.concatenate([per_well[w]['full_combo'] - per_well[w]['true'] for w in OK_WIDS])
sp45_err = np.concatenate([per_well[w]['sp45'] - per_well[w]['true'] for w in OK_WIDS])
trellis_err = np.concatenate([per_well[w]['trellis'] - per_well[w]['true'] for w in OK_WIDS])
print('corr(full_combo_err, trellis_err):', np.corrcoef(full_err, trellis_err)[0, 1])
print('corr(sp45_err, trellis_err):', np.corrcoef(sp45_err, trellis_err)[0, 1])

def pooled_blend(base_key, w, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        pred = (1 - w) * d[base_key] + w * d['trellis']
        sq.append((pred - d['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

W_GRID = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]

def best_w_on(base_key, wl):
    best = (1e9, None)
    for w in W_GRID:
        v = pooled_blend(base_key, w, wl)
        if v < best[0]: best = (v, w)
    return best

for base_key in ['full_combo', 'sp45']:
    print(f'\n=== blend {base_key} + trellis: full-sample sweep ===')
    for w in W_GRID:
        print(f'  w={w:.2f}  pooled={pooled_blend(base_key, w, OK_WIDS):.4f}')

    rng2 = np.random.default_rng(13)
    order = np.array(OK_WIDS); rng2.shuffle(order)
    K = 4
    folds = np.array_split(order, K)
    print(f'--- {K}-fold cross-fit ({base_key}) ---')
    gains = []
    for i in range(K):
        test_wl = folds[i]; train_wl = np.concatenate([folds[j] for j in range(K) if j != i])
        base = pooled_blend(base_key, 0.0, test_wl)
        _, w_star = best_w_on(base_key, train_wl)
        v = pooled_blend(base_key, w_star, test_wl)
        gains.append(base - v)
        print(f'fold {i}: n={len(test_wl)}  baseline={base:.4f}  w={w_star}  test={v:.4f}  gain={base-v:+.4f}')
    print('mean gain:', np.mean(gains), 'positive in', sum(g > 0 for g in gains), f'/{K} folds')
