"""CORRECTED version of warp_weight_extended_crossfit.py: the earlier extended-`a` sweep (a=0.85 giving
7.06 vs production a=0.30's 8.58) was run on ALL 773 wells, but best_warp.pt was TRAINED on 613 of those
773 (only 160 were genuinely held out, seed=42, per kaggle_warp.py). So 79% of that "holdout" evaluation
let WARP lean on memorized wells -- explains the suspiciously huge gain. This restricts EVERYTHING to the
true 160-well holdout (warp_true_holdout_160.pkl) that best_warp.pt never trained on, and further splits
THAT into folds for the blend-weight cross-fit, so both WARP's own quality and the blend-weight choice are
evaluated honestly.
"""
import numpy as np, pickle
from scipy.signal import savgol_filter

PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'
WARP_CACHE = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_on_proxy_cache.pkl'
HOLDOUT_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/research/warp_true_holdout_160.pkl'
proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
warp_on_proxy = pickle.load(open(WARP_CACHE, 'rb'))
VA_WIDS = pickle.load(open(HOLDOUT_PATH, 'rb'))
WIDS_ALL = [wid for wid in warp_on_proxy if wid in proxy]
WIDS = [wid for wid in WIDS_ALL if wid in VA_WIDS]
print('true WARP holdout wells available in proxy:', len(WIDS), '/', len(VA_WIDS))

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

def blended_track(wid, a):
    px = proxy[wid]
    return (1 - a) * px['sp45'] + a * warp_on_proxy[wid]

def physics_pp(wid, base_track, beta=0.75, warmup=500, smooth=51, deg=4):
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

def combo_pred(wid, a):
    return physics_pp(wid, blended_track(wid, a))

def pooled(pred_fn, wid_list):
    s = []
    for wid in wid_list:
        p = pred_fn(wid); s.append((p - proxy[wid]['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(s))))

A_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print('\n=== full true-holdout sweep (160 wells WARP never trained on) ===')
for a in A_GRID:
    print(f'  a={a:.2f}  pooled={pooled(lambda wid, a=a: combo_pred(wid, a), WIDS):.4f}')

print('\n=== sp45-only baseline on true holdout ===')
print('  sp45 solo:', pooled(lambda wid: proxy[wid]['sp45'], WIDS))
print('  warp solo (raw, no physics-pp):', pooled(lambda wid: warp_on_proxy[wid], WIDS))

def best_a_on(wl):
    best = (1e9, None)
    for a in A_GRID:
        v = pooled(lambda wid, a=a: combo_pred(wid, a), wl)
        if v < best[0]: best = (v, a)
    return best

K = 4
rng = np.random.default_rng(13)
order = np.array(WIDS); rng.shuffle(order)
folds = np.array_split(order, K)

print(f'\n=== {K}-fold CV WITHIN the true 160-well holdout: production (a=0.30) vs cross-fit `a` ===')
base_scores, best_scores, chosen = [], [], []
for i in range(K):
    test_wl = folds[i]
    train_wl = np.concatenate([folds[j] for j in range(K) if j != i])
    b_prod = pooled(lambda wid: combo_pred(wid, 0.30), test_wl)
    v_train, a_star = best_a_on(train_wl)
    v_test = pooled(lambda wid, a=a_star: combo_pred(wid, a), test_wl)
    base_scores.append(b_prod); best_scores.append(v_test); chosen.append(a_star)
    print(f'fold {i}: n={len(test_wl):3d}  production(a=0.30)={b_prod:.4f}  '
          f'chosen a={a_star:.2f} -> test={v_test:.4f}  gain={b_prod - v_test:+.4f}  (train score: {v_train:.4f})')

base_scores = np.array(base_scores); best_scores = np.array(best_scores)
print(f'\nmean production: {base_scores.mean():.4f}  mean cross-fit-a: {best_scores.mean():.4f}  '
      f'mean gain: {(base_scores - best_scores).mean():+.4f}')
print(f'gain positive in {int((base_scores > best_scores).sum())}/{K} folds')
print('chosen a per fold:', chosen)
