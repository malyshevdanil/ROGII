"""Uses the cached per_well trellis/full_combo/sp45 predictions (trellis_per_well_cache.pkl) to test
whether WINSORIZING the trellis's per-row contribution (clip how far it can pull the blend away from
the base prediction) rescues the blend from the fold-3 catastrophic-well contamination seen in the plain
linear blend (trellis_vs_fullcombo_crossfit.py: corr=0.077 with full_combo, promising full-sweep gain,
but cross-fit mean gain went NEGATIVE because a couple of catastrophically-wrong trellis wells dominate
squared error even at small blend weight). Trellis solo RMSE is 39 (heavy-tailed); a capped blend should
let the good wells contribute while not letting the bad ones blow up squared error.
"""
import numpy as np, pickle

per_well = pickle.load(open('trellis_per_well_cache.pkl', 'rb'))
OK_WIDS = list(per_well.keys())

def pooled(pred_fn, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        sq.append((pred_fn(d) - d['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

def robust_blend(d, base_key, w, cap_ft):
    diff = d['trellis'] - d[base_key]
    diff_c = np.clip(diff, -cap_ft, cap_ft)
    return d[base_key] + w * diff_c

print('=== full-sample sweep: robust (winsorized) blend, full_combo base ===')
for cap in [3, 5, 8, 12, 20, 1e9]:
    for w in [0.05, 0.1, 0.2, 0.3, 0.5]:
        v = pooled(lambda d, w=w, cap=cap: robust_blend(d, 'full_combo', w, cap), OK_WIDS)
        print(f'  cap={cap:>6}  w={w:.2f}  pooled={v:.4f}')
    print()

print('baseline (w=0):', pooled(lambda d: d['full_combo'], OK_WIDS))

# cross-fit over (w, cap) jointly
W_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]
CAP_GRID = [3, 5, 8, 12, 20]

def best_on(base_key, wl):
    best = (1e9, None)
    for cap in CAP_GRID:
        for w in W_GRID:
            v = pooled(lambda d, w=w, cap=cap: robust_blend(d, base_key, w, cap), wl)
            if v < best[0]: best = (v, (w, cap))
    return best

rng = np.random.default_rng(13)
order = np.array(OK_WIDS); rng.shuffle(order)
K = 4
folds = np.array_split(order, K)
print(f'\n=== {K}-fold cross-fit: robust blend vs full_combo ===')
gains = []
for i in range(K):
    test_wl = folds[i]; train_wl = np.concatenate([folds[j] for j in range(K) if j != i])
    base = pooled(lambda d: d['full_combo'], test_wl)
    _, (w_star, cap_star) = best_on('full_combo', train_wl)
    v = pooled(lambda d, w=w_star, cap=cap_star: robust_blend(d, 'full_combo', w, cap), test_wl)
    gains.append(base - v)
    print(f'fold {i}: n={len(test_wl)}  baseline={base:.4f}  w={w_star} cap={cap_star} -> test={v:.4f}  gain={base-v:+.4f}')
print('mean gain:', np.mean(gains), 'positive in', sum(g > 0 for g in gains), f'/{K} folds')
