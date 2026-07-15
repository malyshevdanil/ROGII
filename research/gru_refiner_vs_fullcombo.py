"""Evaluate the saved best GRU-refiner checkpoint (ep6, holdout RMSE 9.85 vs sp45's 10.62) for
decorrelation and blend value against the full production combo (9.70 on the same true holdout),
using the paired well-bootstrap validator (not a handful of fixed folds).
"""
import numpy as np, pickle, torch
from scipy.signal import savgol_filter
from gru_refiner import build_dataset, GRURefiner, CFG, DEV, SEQ_LEN, normalize_features
from paired_bootstrap import paired_bootstrap_gain

DATA = build_dataset()
WIDS = list(DATA.keys())
rng = np.random.default_rng(CFG['seed']); idx = np.arange(len(WIDS)); rng.shuffle(idx)
VA = [WIDS[i] for i in idx[:CFG['n_val']]]; TR = [WIDS[i] for i in idx[CFG['n_val']:]]
mean, std = normalize_features(DATA, TR)  # must match training normalization (fit on TR only)

net = GRURefiner(21, CFG['d'], CFG['layers'], CFG['drop']).to(DEV)
net.load_state_dict(torch.load('gru_refiner_best.pt', map_location=DEV))
net.eval()

gru_pred = {}
with torch.no_grad():
    for i in range(0, len(VA), 32):
        ws = VA[i:i + 32]
        X = torch.tensor(np.stack([DATA[w]['Xn'] for w in ws]), dtype=torch.float32, device=DEV)
        pred_resid = net(X).cpu().numpy()
        for j, w in enumerate(ws):
            d = DATA[w]; n = d['n']
            src_dst = np.linspace(0, n - 1, SEQ_LEN)
            pred_full = np.interp(np.arange(n), src_dst, pred_resid[j])
            gru_pred[w] = d['src_sp45'] + pred_full

proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
warp_on_proxy = pickle.load(open('warp_on_proxy_cache.pkl', 'rb'))

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
for w in VA:
    px = proxy[w]
    per_well[w] = dict(gru=gru_pred[w], full_combo=full_combo_track(w), true=px['true'])

def pooled(key, wl):
    sq = [(per_well[w][key] - per_well[w]['true']) ** 2 for w in wl]
    return float(np.sqrt(np.mean(np.concatenate(sq))))

print('gru solo:', pooled('gru', VA))
print('full_combo solo:', pooled('full_combo', VA))
full_err = np.concatenate([per_well[w]['full_combo'] - per_well[w]['true'] for w in VA])
gru_err = np.concatenate([per_well[w]['gru'] - per_well[w]['true'] for w in VA])
print('corr(full_combo_err, gru_err):', np.corrcoef(full_err, gru_err)[0, 1])

def blend(wid, w):
    d = per_well[wid]
    return (1 - w) * d['full_combo'] + w * d['gru']

print('\n=== full-sample sweep ===')
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    sq = [(blend(wid, w) - per_well[wid]['true']) ** 2 for wid in VA]
    print(f'  w={w:.2f}  pooled={np.sqrt(np.mean(np.concatenate(sq))):.4f}')

sq_base = {w: (per_well[w]['full_combo'] - per_well[w]['true']) ** 2 for w in VA}
for w_test in [0.1, 0.2, 0.3, 0.5]:
    sq_cand = {w: (blend(w, w_test) - per_well[w]['true']) ** 2 for w in VA}
    mean_gain, lo, hi, frac_pos = paired_bootstrap_gain(sq_base, sq_cand, VA, n_boot=3000, seed=0)
    print(f'paired bootstrap (w={w_test}): mean_gain={mean_gain:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  frac_positive={frac_pos:.3f}')
