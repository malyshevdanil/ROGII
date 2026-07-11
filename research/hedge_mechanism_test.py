"""Tests the 'hedge is base-dependent, not a universal lever' mechanism on our own data:
does wall-hedge (shrink toward last-known TVT) help a bare sp45 base (as we found before, alpha~0.10-0.15),
but become redundant/harmful once stacked on the already-tuned WARP+physics-pp combo (our real 6.836 best)?
Cross-fit (5-fold grouped) both ways so the answer isn't a lucky split.
"""
import numpy as np, pandas as pd, glob, os, pickle, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import savgol_filter

torch.manual_seed(0); np.random.seed(0); DEV = 'cpu'
CFG = dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.2, maxstep=1.5)
TG = CFG['TWGRID']; KS = CFG['KSTEPS']; ES = CFG['EVSTEPS']
TRAIN_DIR = 'd:/ROGII/data/train'
PROXY_PATH = 'C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl'

def inn(a):
    a = a.copy(); n = len(a); idx = np.arange(n); m = np.isnan(a)
    if m.all(): return np.zeros(n)
    a[m] = np.interp(idx[m], idx[~m], a[~m]); return a

def build_well(wid):
    try:
        hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
        tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    except Exception:
        return None
    if 'TVT' not in hw.columns: return None
    tt = tw['TVT'].values.astype(float); tg = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt) < 10: return None
    kn = hw[hw['TVT_input'].notna()]
    if len(kn) < 20 or hw['TVT_input'].isna().sum() < 20: return None
    last = kn.iloc[-1]; last_tvt = float(last['TVT_input'])
    n = len(hw); gr = inn(hw['GR'].values.astype(float))
    kn_gr = kn['GR'].interpolate().bfill().ffill().values.astype(float)
    twk = np.interp(kn['TVT_input'].values, tt, tg); v = np.isfinite(kn_gr) & np.isfinite(twk)
    a, b = (np.polyfit(kn_gr[v], twk[v], 1) if v.sum() >= 20 else (1., 0.)); cal = gr * a + b
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float); mdd = np.gradient(MD); mdd[mdd == 0] = 1
    grad = np.gradient(gr); rstd = pd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
    dz = np.gradient(Z) / mdd
    ev = hw['TVT_input'].isna().values.astype(float); ei = np.where(ev > 0.5)[0]
    if len(ei) < 5: return None
    e0 = ei[0]; ks = np.arange(max(0, e0 - 400), e0)
    if len(ks) < 5: return None
    g_tvt = np.linspace(tt.min(), tt.max(), TG); g_gr = np.interp(g_tvt, tt, tg); gm, gs = float(g_gr.mean()), float(g_gr.std() + 1e-6)
    caln = (cal - gm) / gs; gradn = grad / gs; rstdn = rstd / gs; true = hw['TVT'].values.astype(float)
    kd = np.linspace(ks[0], ks[-1], KS); ed = np.linspace(e0, n - 1, ES); dst = np.concatenate([kd, ed]); src = np.arange(n)
    R = lambda x: np.interp(dst, src, x).astype(np.float32)
    evr = np.concatenate([np.zeros(KS), np.ones(ES)]).astype(np.float32)
    return dict(wid=wid, H=np.stack([R(caln), R(gradn), R(rstdn), R(dz)]).astype(np.float32),
                gn=((g_gr - gm) / gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt,
                md=R(MD), dz_raw=R(dz), md_raw=MD, ev_raw=hw['TVT_input'].isna().values)

class Enc(nn.Module):
    def __init__(s, nin, d, drop):
        super().__init__(); s.inp = nn.Conv1d(nin, d, 5, padding=2)
        s.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(d, d, 3, padding=dl, dilation=dl), nn.GroupNorm(8, d), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(d, d, 3, padding=dl, dilation=dl), nn.GroupNorm(8, d), nn.GELU()) for dl in (1, 2, 4, 8, 16)])
        s.out = nn.Conv1d(d, d, 1)
    def forward(s, x):
        h = F.gelu(s.inp(x))
        for b in s.blocks: h = h + b(h)
        return s.out(h)

class WarpNet(nn.Module):
    def __init__(s, d, drop):
        super().__init__(); s.he = Enc(4, d, drop); s.te = Enc(1, d, drop)
        s.q = nn.Conv1d(d, d, 1); s.k = nn.Conv1d(d, d, 1); s.vv = nn.Conv1d(d, d, 1); s.sc = d ** -0.5
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
    def forward(s, H, G, lt):
        h = s.he(H); t = s.te(G)
        Q = s.q(h).transpose(1, 2); K = s.k(t).transpose(1, 2); V = s.vv(t).transpose(1, 2)
        att = torch.softmax(Q @ K.transpose(1, 2) * s.sc, dim=2); ctx = att @ V
        x = torch.cat([h.transpose(1, 2), ctx], dim=2)
        dt = torch.tanh(s.head(x)[..., 0]) * CFG['maxstep']; tvt = torch.cumsum(dt, 1)
        return tvt - tvt[:, KS - 1:KS] + lt[:, None]

print('building well features (all train wells)...')
wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA = [b for b in (build_well(w) for w in wids) if b is not None]
d3 = np.concatenate([w['dz_raw'] for w in DATA]); m3, s3 = d3.mean(), d3.std() + 1e-6
for w in DATA: w['H'][3] = (w['dz_raw'] - m3) / s3

net = WarpNet(CFG['d'], CFG['drop']).to(DEV)
sd = torch.load('d:/ROGII/best_warp.pt', map_location=DEV)
net.load_state_dict(sd); net.eval()
print('running WARP inference...')
warp_pred = {}
with torch.no_grad():
    for i in range(0, len(DATA), 16):
        ws = DATA[i:i + 16]
        H = torch.tensor(np.stack([w['H'] for w in ws]), dtype=torch.float32, device=DEV)
        G = torch.tensor(np.stack([w['gn'][:TG] for w in ws]), dtype=torch.float32, device=DEV)[:, None]
        lt = torch.tensor([w['last_tvt'] for w in ws], dtype=torch.float32, device=DEV)
        p = net(H, G, lt).cpu().numpy()
        for w, pp in zip(ws, p): warp_pred[w['wid']] = pp

proxy = pickle.load(open(PROXY_PATH, 'rb'))['DATA']
common = [w for w in DATA if w['wid'] in proxy]
warp_on_proxy = {}
for w in common:
    wid = w['wid']; ev_mask = w['ev'] > 0.5
    md_ev = w['md'][ev_mask]; warp_ev = warp_pred[wid][ev_mask]
    md_raw_ev = w['md_raw'][w['ev_raw']]
    warp_on_proxy[wid] = np.interp(md_raw_ev, md_ev, warp_ev)

WIDS = [w['wid'] for w in common]
print('usable wells:', len(WIDS))

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

def physics_pp(wid, base_track, beta, warmup, smooth):
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

FROZEN = (0.30, 0.75, 500, 51)
def combo_track(wid):
    return physics_pp(wid, blended_track(wid, FROZEN[0]), *FROZEN[1:])

def hedge(base_track, wid, alpha):
    lt = proxy[wid]['lt']
    return (1 - alpha) * base_track + alpha * lt

def pooled(pred_fn, wid_list):
    s = []
    for wid in wid_list:
        p = pred_fn(wid); s.append((p - proxy[wid]['true']) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(s))))

ALPHA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

def best_alpha_on(wl, base_fn):
    best = (99.0, None)
    for a in ALPHA_GRID:
        v = pooled(lambda wid, a=a: hedge(base_fn(wid), wid, a), wl)
        if v < best[0]: best = (v, a)
    return best

K = 5
rng = np.random.default_rng(13)  # SAME seed as the earlier k-fold script, for comparable folds
order = np.array(WIDS); rng.shuffle(order)
folds = np.array_split(order, K)

print('\n=== Does wall-hedge help bare sp45? (cross-fit, 5-fold) ===')
gains_sp45 = []
for i in range(K):
    test_wl = folds[i]; train_wl = np.concatenate([folds[j] for j in range(K) if j != i])
    b = pooled(lambda wid: proxy[wid]['sp45'], test_wl)
    _, a = best_alpha_on(train_wl, lambda wid: proxy[wid]['sp45'])
    h = pooled(lambda wid, a=a: hedge(proxy[wid]['sp45'], wid, a), test_wl)
    gains_sp45.append(b - h)
    print(f'fold {i}: baseline={b:.4f}  hedge(alpha={a:.2f})={h:.4f}  gain={b-h:+.4f}')
print('mean gain (hedge on bare sp45):', np.mean(gains_sp45))

print('\n=== Does wall-hedge help on TOP of the already-tuned WARP+physics-pp combo? (cross-fit, 5-fold) ===')
gains_combo = []
for i in range(K):
    test_wl = folds[i]; train_wl = np.concatenate([folds[j] for j in range(K) if j != i])
    b = pooled(lambda wid: combo_track(wid), test_wl)
    _, a = best_alpha_on(train_wl, lambda wid: combo_track(wid))
    h = pooled(lambda wid, a=a: hedge(combo_track(wid), wid, a), test_wl)
    gains_combo.append(b - h)
    print(f'fold {i}: combo baseline={b:.4f}  combo+hedge(alpha={a:.2f})={h:.4f}  gain={b-h:+.4f}')
print('mean gain (hedge on top of WARP+pp combo):', np.mean(gains_combo))

print(f'\n=== summary ===')
print(f'hedge on bare sp45:        mean gain {np.mean(gains_sp45):+.4f}, positive in {sum(g>0 for g in gains_sp45)}/{K} folds')
print(f'hedge on WARP+pp combo:    mean gain {np.mean(gains_combo):+.4f}, positive in {sum(g>0 for g in gains_combo)}/{K} folds')
