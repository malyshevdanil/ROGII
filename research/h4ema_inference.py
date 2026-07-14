"""Run inference with the H4+EMA checkpoint (best_ema.pt) -- a genuinely different neural architecture
from the production WARP net: 8 multi-scale/band-pass GR channels (vs WARP's 4) + EMA-averaged weights
(implicit weight-space ensembling). Both were trained with the IDENTICAL n_val=160, seed=42 holdout split
(kaggle_h4.py mirrors kaggle_warp.py's construction exactly), so predictions on warp_true_holdout_160.pkl
are honestly out-of-sample for BOTH nets -- no re-derivation of the split needed, just reuse it.

Purpose: test whether two ARCHITECTURALLY DIFFERENT neural nets (not two classical trellis/PF variants)
decorrelate well enough to add real blend value -- the working hypothesis from today's post-mortem is that
our failed blend attempts (beam-HSMM, fwd-bwd trellis) failed because they were "classical on top of
classical" (our combo is already 70% classical PF + 30% WARP-neural); true decorrelation needs a second
voice from a DIFFERENT paradigm, and the two neural nets we already have (different feature sets + EMA vs
raw weights) are the cheapest available test of that hypothesis -- no new training needed.
"""
import numpy as np, pandas as pd, glob, os, pickle, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cpu'
TRAIN_DIR = 'd:/ROGII/data/train'
CFG = dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.2, maxstep=1.5)
TG = CFG['TWGRID']; KSTEPS = CFG['KSTEPS']; EVSTEPS = CFG['EVSTEPS']

def interp_nan(a):
    a = a.copy(); n = len(a); idx = np.arange(n); m = np.isnan(a)
    if m.all(): return np.zeros(n)
    a[m] = np.interp(idx[m], idx[~m], a[~m]); return a

def build_well_h4(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tw_tvt) < 10: return None
    kn = hw[hw['TVT_input'].notna()]
    if len(kn) < 20 or hw['TVT_input'].isna().sum() < 20: return None
    last = kn.iloc[-1]; last_tvt = float(last['TVT_input'])
    n = len(hw); gr = interp_nan(hw['GR'].values.astype(float))
    kn_gr = kn['GR'].interpolate().bfill().ffill().values.astype(float)
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr); v = np.isfinite(kn_gr) & np.isfinite(tw_at_k)
    a, b = (np.polyfit(kn_gr[v], tw_at_k[v], 1) if v.sum() >= 20 else (1., 0.)); cal_gr = gr * a + b
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float); mdd = np.gradient(MD); mdd[mdd == 0] = 1
    grad = np.gradient(gr); rstd = pd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
    dzdmd = np.gradient(Z) / mdd
    ev = hw['TVT_input'].isna().values.astype(float); ei = np.where(ev > 0.5)[0]
    if len(ei) < 5: return None
    e0 = ei[0]; ksrc = np.arange(max(0, e0 - 400), e0)
    if len(ksrc) < 5: return None
    g_tvt = np.linspace(tw_tvt.min(), tw_tvt.max(), TG); g_gr = np.interp(g_tvt, tw_tvt, tw_gr)
    gm, gs = float(g_gr.mean()), float(g_gr.std() + 1e-6); caln = (cal_gr - gm) / gs; gradn = grad / gs; rstdn = rstd / gs
    cs = pd.Series(caln)
    sm5 = cs.rolling(5, center=True, min_periods=1).mean().values
    sm15 = cs.rolling(15, center=True, min_periods=1).mean().values
    sm41 = cs.rolling(41, center=True, min_periods=1).mean().values
    dog1 = sm5 - sm15; dog2 = sm15 - sm41
    true = hw['TVT'].values.astype(float)
    kd = np.linspace(ksrc[0], ksrc[-1], KSTEPS); ed = np.linspace(e0, n - 1, EVSTEPS)
    dst = np.concatenate([kd, ed]); src = np.arange(n)
    R = lambda x: np.interp(dst, src, x).astype(np.float32)
    evr = np.concatenate([np.zeros(KSTEPS), np.ones(EVSTEPS)]).astype(np.float32)
    return dict(wid=wid, H=np.stack([R(caln), R(sm15), R(sm41), R(dog1), R(dog2), R(gradn), R(rstdn), R(dzdmd)]).astype(np.float32),
                gn=((g_gr - gm) / gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt,
                md=R(MD), md_raw=MD, ev_raw=hw['TVT_input'].isna().values)

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

class WarpNetH4(nn.Module):
    def __init__(s, d, drop):
        super().__init__(); s.he = Enc(8, d, drop); s.te = Enc(1, d, drop)
        s.q = nn.Conv1d(d, d, 1); s.k = nn.Conv1d(d, d, 1); s.vv = nn.Conv1d(d, d, 1); s.sc = d ** -0.5
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))
    def forward(s, H, G, lt):
        h = s.he(H); t = s.te(G)
        Q = s.q(h).transpose(1, 2); K = s.k(t).transpose(1, 2); V = s.vv(t).transpose(1, 2)
        att = torch.softmax(Q @ K.transpose(1, 2) * s.sc, dim=2); ctx = att @ V
        x = torch.cat([h.transpose(1, 2), ctx], dim=2)
        dt = torch.tanh(s.head(x)[..., 0]) * CFG['maxstep']; tvt = torch.cumsum(dt, 1)
        return tvt - tvt[:, KSTEPS - 1:KSTEPS] + lt[:, None]

if __name__ == '__main__':
    import time
    proxy = pickle.load(open('C:/Users/user/AppData/Local/Temp/claude/d--ROGII/6761a874-94c4-40fb-a605-d49c9b58b718/scratchpad/proxy.pkl', 'rb'))['DATA']
    VA_WIDS = pickle.load(open('warp_true_holdout_160.pkl', 'rb'))

    print('building H4 features for the true holdout wells...')
    t0 = time.time()
    DATA = [b for b in (build_well_h4(w) for w in VA_WIDS if w in proxy) if b is not None]
    d3 = np.concatenate([w['H'][7] for w in DATA]); m3, s3 = d3.mean(), d3.std() + 1e-6
    for w in DATA: w['H'][7] = (w['H'][7] - m3) / s3
    print(f'built {len(DATA)} wells  {time.time()-t0:.0f}s')

    net = WarpNetH4(CFG['d'], CFG['drop']).to(DEV)
    sd = torch.load('d:/ROGII/best_ema.pt', map_location=DEV)
    net.load_state_dict(sd); net.eval()

    h4_on_proxy = {}
    with torch.no_grad():
        for i in range(0, len(DATA), 16):
            ws = DATA[i:i + 16]
            H = torch.tensor(np.stack([w['H'] for w in ws]), dtype=torch.float32, device=DEV)
            G = torch.tensor(np.stack([w['gn'][:TG] for w in ws]), dtype=torch.float32, device=DEV)[:, None]
            lt = torch.tensor([w['last_tvt'] for w in ws], dtype=torch.float32, device=DEV)
            p = net(H, G, lt).cpu().numpy()
            for w, pp in zip(ws, p):
                wid = w['wid']
                ev_mask = w['ev'] > 0.5
                md_ev = w['md'][ev_mask]; pred_ev = pp[ev_mask]
                md_raw_ev = w['md_raw'][w['ev_raw']]
                h4_on_proxy[wid] = np.interp(md_raw_ev, md_ev, pred_ev)

    print('h4+ema predictions computed for', len(h4_on_proxy), 'wells')
    pickle.dump(h4_on_proxy, open('h4ema_on_proxy_cache.pkl', 'wb'))
    print('saved to h4ema_on_proxy_cache.pkl')
