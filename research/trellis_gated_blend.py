"""Well-level gate: mean|trellis - sp45| is a LEGAL (no true-label leakage) diagnostic that correlates
0.94 with the trellis's actual per-well RMSE on the cached sample -- when the trellis disagrees a lot
with the safe sp45 baseline, it is almost always because the trellis locked onto a globally wrong mode
(the same self-similar-GR failure mode diagnosed earlier this session for the beam-HSMM), not because
it found real signal sp45 missed. Gate: zero out (or downweight) the trellis's blend contribution per-well
based on this legal deviation signal, before applying the (small) global blend weight.
"""
import numpy as np, pickle

per_well = pickle.load(open('trellis_per_well_cache.pkl', 'rb'))
OK_WIDS = list(per_well.keys())

for wid, d in per_well.items():
    d['dev_from_sp45'] = float(np.mean(np.abs(d['trellis'] - d['sp45'])))

def pooled(pred_fn, wl):
    sq = []
    for wid in wl:
        d = per_well[wid]
        sq.append((pred_fn(d) - d['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq))))

def gated_blend(d, base_key, w, gate_thresh):
    if d['dev_from_sp45'] > gate_thresh:
        return d[base_key]
    return (1 - w) * d[base_key] + w * d['trellis']

print('=== full-sample sweep: gated blend, full_combo base ===')
for gate in [10, 15, 20, 25, 30, 40, 1e9]:
    for w in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        v = pooled(lambda d, w=w, g=gate: gated_blend(d, 'full_combo', w, g), OK_WIDS)
        print(f'  gate={gate:>6}  w={w:.2f}  pooled={v:.4f}')
    print()

print('baseline (no trellis):', pooled(lambda d: d['full_combo'], OK_WIDS))

GATE_GRID = [10, 15, 20, 25, 30, 40]
W_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

def best_on(base_key, wl):
    best = (1e9, None)
    for gate in GATE_GRID:
        for w in W_GRID:
            v = pooled(lambda d, w=w, g=gate: gated_blend(d, base_key, w, g), wl)
            if v < best[0]: best = (v, (w, gate))
    return best

rng = np.random.default_rng(13)
order = np.array(OK_WIDS); rng.shuffle(order)
K = 4
folds = np.array_split(order, K)
print(f'\n=== {K}-fold cross-fit: gated blend vs full_combo ===')
gains = []
for i in range(K):
    test_wl = folds[i]; train_wl = np.concatenate([folds[j] for j in range(K) if j != i])
    base = pooled(lambda d: d['full_combo'], test_wl)
    _, (w_star, gate_star) = best_on('full_combo', train_wl)
    v = pooled(lambda d, w=w_star, g=gate_star: gated_blend(d, 'full_combo', w, g), test_wl)
    gains.append(base - v)
    print(f'fold {i}: n={len(test_wl)}  baseline={base:.4f}  w={w_star} gate={gate_star} -> test={v:.4f}  gain={base-v:+.4f}')
print('mean gain:', np.mean(gains), 'positive in', sum(g > 0 for g in gains), f'/{K} folds')
