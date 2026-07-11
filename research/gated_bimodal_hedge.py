"""Test the GATED, GENTLE bimodal-midpoint hedge (per a competitor's writeup): whole-lateral constant
datum-shift scan J(delta) using GR vs typewell; find the two best-separated local minima; if they are
near-tied (loose trigger, ratio<=1.15) AND our own PF-ensemble disagreement (seed_std) is high, pull
GENTLY (alpha=0.2) toward the midpoint of the two minima. Honest cross-fit validation on the proxy."""
import pickle, numpy as np, pandas as pd, os
from scipy.signal import argrelextrema

TR = 'd:/ROGII/data/train'
proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))
D = proxy['DATA']
wids = list(D.keys())

def inan(a):
    a = a.copy(); m = np.isnan(a); i = np.arange(len(a))
    if m.all(): return np.zeros(len(a))
    a[m] = np.interp(i[m], i[~m], a[~m]); return a

GRID = np.arange(-40, 40.5, 1.0)

def well_shift_scan(wid):
    d = D[wid]
    hw = pd.read_csv(f'{TR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TR}/{wid}__typewell.csv').sort_values('TVT')
    tt = tw['TVT'].values.astype(float); tg = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]
    ev_mask = hw['TVT_input'].isna().values
    if kn.shape[0] < 20 or ev_mask.sum() < 20: return None
    gr = inan(hw['GR'].values.astype(float))
    kn_gr = kn['GR'].interpolate().bfill().ffill().values.astype(float)
    twk = np.interp(kn['TVT_input'].values, tt, tg)
    v = np.isfinite(kn_gr) & np.isfinite(twk)
    a, b = (np.polyfit(kn_gr[v], twk[v], 1) if v.sum() >= 20 else (1., 0.))
    cal_gr_eval = (gr[ev_mask]) * a + b
    pred_path = d['sp45']    # our pipeline's predicted TVT path (proxy), aligned to eval rows
    if len(pred_path) != ev_mask.sum():
        # proxy's arrays are ALREADY eval-only; hw's ev_mask selects raw eval rows -- lengths should match
        return None
    J = np.array([np.nanmean((cal_gr_eval - np.interp(pred_path + delta, tt, tg)) ** 2) for delta in GRID])
    disagree = float(np.mean(np.abs(pred_path - d['beam'])))
    return J, pred_path, d['true'], d['lt'], disagree

def smooth(J, k=3):
    kernel = np.ones(k) / k
    return np.convolve(J, kernel, mode='same')

def two_best_minima(J):
    Js = smooth(J, 3)
    idx_min = argrelextrema(Js, np.less, order=3)[0]
    idx_min = [i for i in idx_min if 0 < i < len(Js) - 1]
    if len(idx_min) < 2:
        i0 = int(np.argmin(Js)); return GRID[i0], Js[i0], None, None
    vals = sorted(idx_min, key=lambda i: Js[i])
    a_i = vals[0]
    # best-separated second minimum (>= 8 ft away from the first, per competitor's "8-20ft apart" spec)
    b_i = None
    for i in vals[1:]:
        if abs(GRID[i] - GRID[a_i]) >= 8:
            b_i = i; break
    if b_i is None: return GRID[a_i], Js[a_i], None, None
    return GRID[a_i], Js[a_i], GRID[b_i], Js[b_i]

print('scanning wells...')
records = []
for i, wid in enumerate(wids):
    try:
        out = well_shift_scan(wid)
    except Exception:
        out = None
    if out is None: continue
    J, pred_path, true, lt, disagree = out
    da, Ja, db, Jb = two_best_minima(J)
    records.append(dict(wid=wid, da=da, Ja=Ja, db=db, Jb=Jb, pred_path=pred_path, true=true, lt=lt, disagree=disagree))
    if (i + 1) % 200 == 0: print(' ', i + 1, '/', len(wids))
print('usable wells:', len(records))

# gate: two-minima ratio <= 1.15 (loose trigger) AND disagreement in the top quartile (per competitor's
# JOINT gate: "J(Δb) ≤ 1.15·J(Δa) and pf_vs_beam_disagreement(w) is high")
disagree_vals = np.array([r['disagree'] for r in records])
disagree_thresh = np.percentile(disagree_vals, 75)
for r in records:
    ratio_ok = bool(r['db'] is not None and r['Jb'] <= 1.15 * r['Ja'])
    r['triggered'] = bool(ratio_ok and r['disagree'] >= disagree_thresh)
n_trig = sum(r['triggered'] for r in records)
print('disagreement threshold (75th pct):', disagree_thresh)
print('triggered (ratio AND top-quartile disagreement) wells:', n_trig, '/', len(records))

def pooled(preds_map):
    s = []
    for r in records:
        p = preds_map(r)
        s.append((p - r['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(s))))

print('\n=== baseline (sp45, no hedge) ===')
print('pooled:', pooled(lambda r: r['pred_path']))

print('\n=== hard-commit to better minimum (on triggered wells only) ===')
def hardcommit(r):
    if r['triggered']:
        return r['pred_path'] + r['da']  # da is already the BEST (lowest J) minimum
    return r['pred_path']
print('pooled:', pooled(hardcommit))

for alpha in [0.1, 0.2, 0.3, 0.5]:
    def gentle(r, a=alpha):
        if r['triggered']:
            mid = (r['da'] + r['db']) / 2.0
            return (1 - a) * r['pred_path'] + a * (r['pred_path'] + mid)
        return r['pred_path']
    print(f'gentle midpoint hedge alpha={alpha}: pooled=', pooled(gentle))
