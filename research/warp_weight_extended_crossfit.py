"""The extended coordinate sweep (physicspp_extended_grid.py) found a=0.85 giving 7.06 pooled RMSE
vs the current production a=0.30 giving 8.58 -- a huge, suspicious jump. Before trusting this, run a
proper 5-fold grouped cross-fit: select the best `a` on the OTHER folds only, evaluate on the held-out
fold, exactly like kfold_cv_new_best.py did for the original grid. If this survives cross-fit as cleanly
as the a=0.15->0.30 real-LB-confirmed lever did, it is a much bigger version of the same validated lever.
"""
import numpy as np, pickle
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

A_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

def best_a_on(wl):
    best = (1e9, None)
    for a in A_GRID:
        v = pooled(lambda wid, a=a: combo_pred(wid, a), wl)
        if v < best[0]: best = (v, a)
    return best

print('=== full-set sweep (sanity re-check) ===')
for a in A_GRID:
    print(f'  a={a:.2f}  pooled={pooled(lambda wid, a=a: combo_pred(wid, a), WIDS):.4f}')

K = 5
rng = np.random.default_rng(13)
order = np.array(WIDS); rng.shuffle(order)
folds = np.array_split(order, K)

print(f'\n=== {K}-fold grouped CV: current production (a=0.30) vs cross-fit-selected `a` ===')
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
