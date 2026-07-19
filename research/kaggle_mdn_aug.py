"""MDN + PREFIX-CUT AUGMENTATION (no synthetic pretraining). After 3 straight synthetic-generator
failures this session (range bug -> flat-collapse -> flat-collapse again even with real-anchored
pairing -- see rogii_nn_research memory), user chose to drop synthetic pretraining and instead push
the ONE augmentation mechanism with a proven real-LB track record in this whole project: prefix-cut
augmentation (GRU-refiner v7: "8.404->8.159" per Tucker's writeup, and our own "-0.187 real gain").
Adds prefix-cut to kaggle_mdn.py's MDN+cross-attention+Viterbi architecture: for TRAIN wells only,
also train on 1-2 additional cuts where only a `cut_frac` PREFIX of the known zone is used to compute
the last-known anchor / calibration / known-zone tail window -- the real eval targets (e0:n-1) are
UNCHANGED, so this is 100% real supervision, just with an artificially staler/poorer anchor context
(exactly what many genuinely-hard real held-out wells look like). VA (validation) wells always use
cut_frac=1.0 (the real, unmodified anchor), matching "validation uses only the original hidden zones."
"""
import numpy as np, pandas as pd, glob, os, time, copy, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CFG = dict(d=96, K=3, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.20, wd=3e-4, lr=1.5e-3,
           epochs=int(os.environ.get('EPOCHS', '120')), bs=24,
           max_off=90., min_sig=2., trans=0.02, aug_caljit=0.12, aug_noise=0.07, aug_warp=0.35, aug_hflip=0.5,
           w_known=0.5, ema_decay=0.995, n_val=160, seed=42, tag='mdn_aug', n_cuts=4, cut_lo=0.3, cut_hi=0.9)
TG = CFG['TWGRID']; KSTEPS = CFG['KSTEPS']; EVSTEPS = CFG['EVSTEPS']; K = CFG['K']
c = glob.glob('/kaggle/input/**/*__horizontal_well.csv', recursive=True); _t = [p for p in c if 'train' in p.lower()]; c = _t if _t else c
TRAIN_DIR = os.path.dirname(c[0]) if c else next((d for d in ('data/train', 'd:/ROGII/data/train') if os.path.isdir(d)), 'data/train')
print('DEV=%s data=%s' % (DEV, TRAIN_DIR), flush=True)


def inan(a):
    a = a.copy(); m = np.isnan(a); i = np.arange(len(a))
    if m.all(): return np.zeros(len(a))
    a[m] = np.interp(i[m], i[~m], a[~m]); return a


def build(wid, cut_frac=1.0):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tt = tw['TVT'].values.astype(float); tg = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt) < 10: return None
    kn_idx = np.where(hw['TVT_input'].notna().values)[0]
    if len(kn_idx) < 20 or hw['TVT_input'].isna().sum() < 20: return None
    n_use = max(20, int(len(kn_idx) * cut_frac)); kn_idx_use = kn_idx[:n_use]
    last_i = int(kn_idx_use[-1]); last_tvt = float(hw['TVT_input'].iloc[last_i])
    n = len(hw); gr = inan(hw['GR'].values.astype(float))
    kg = gr[kn_idx_use]; ktvt = hw['TVT_input'].values[kn_idx_use]
    twk = np.interp(ktvt, tt, tg); v = np.isfinite(kg) & np.isfinite(twk)
    a, b = (np.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.)); cg = gr * a + b
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float); mdd = np.gradient(MD); mdd[mdd == 0] = 1; dzdmd = np.gradient(Z) / mdd
    ev = hw['TVT_input'].isna().values.astype(float); ei = np.where(ev > 0.5)[0]
    if len(ei) < 5: return None
    e0 = ei[0]
    ks = np.arange(max(0, last_i + 1 - 400), last_i + 1)
    if len(ks) < 5: return None
    g_tvt = np.linspace(tt.min(), tt.max(), TG); g_gr = np.interp(g_tvt, tt, tg); gm, gs = float(g_gr.mean()), float(g_gr.std() + 1e-6)
    caln = (cg - gm) / gs; true = hw['TVT'].values.astype(float)
    kd = np.linspace(ks[0], ks[-1], KSTEPS); ed = np.linspace(e0, n - 1, EVSTEPS); dst = np.concatenate([kd, ed]); src = np.arange(n)
    R = lambda x: np.interp(dst, src, x).astype(np.float32); evr = np.concatenate([np.zeros(KSTEPS), np.ones(EVSTEPS)]).astype(np.float32)
    return dict(H=np.stack([R(caln), R(np.gradient(caln)), R(dzdmd)]).astype(np.float32),
                gn=((g_gr - gm) / gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt)


print('building...', flush=True); t0 = time.time()
wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
BASE = {}
for w in wids:
    b = build(w, 1.0)
    if b is not None: BASE[w] = b
rng0 = np.random.default_rng(CFG['seed']); idx0 = np.arange(len(BASE)); rng0.shuffle(idx0)
wids_ok = list(BASE.keys())
VA_wids = set(wids_ok[i] for i in idx0[:CFG['n_val']])
TR_wids = [wids_ok[i] for i in idx0[CFG['n_val']:]]
VA = [BASE[w] for w in VA_wids]
DATA_TR = {w: BASE[w] for w in TR_wids}
rng = np.random.default_rng(CFG['seed'] + 1)
for w in TR_wids:
    for _ in range(CFG['n_cuts']):
        f = rng.uniform(CFG['cut_lo'], CFG['cut_hi'])
        b = build(w, float(f))
        if b is not None: DATA_TR[f'{w}__cut{len(DATA_TR)}'] = b
TR = list(DATA_TR.values())
d2 = np.concatenate([w['H'][2] for w in TR]); mm, ss = d2.mean(), d2.std() + 1e-6
for w in TR: w['H'][2] = (w['H'][2] - mm) / ss
for w in VA: w['H'][2] = (w['H'][2] - mm) / ss
print('built VA=%d TR_orig=%d TR_with_cuts=%d %.0fs' % (len(VA), len(TR_wids), len(TR), time.time() - t0), flush=True)


def prmse(s): return float(np.sqrt(np.mean(np.concatenate(s))))


print('BASELINES flat=%.3f' % prmse([(w['last_tvt'] - w['tvt'][w['ev'] > 0.5]) ** 2 for w in VA]), flush=True)


class Enc(nn.Module):
    def __init__(s, nin, d, drop):
        super().__init__(); s.inp = nn.Conv1d(nin, d, 5, padding=2)
        s.bl = nn.ModuleList([nn.Sequential(nn.Conv1d(d, d, 3, padding=dl, dilation=dl), nn.GroupNorm(8, d), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(d, d, 3, padding=dl, dilation=dl), nn.GroupNorm(8, d), nn.GELU()) for dl in (1, 2, 4, 8, 16)]); s.o = nn.Conv1d(d, d, 1)

    def forward(s, x):
        h = F.gelu(s.inp(x))
        for b in s.bl: h = h + b(h)
        return s.o(h)


class MDN(nn.Module):
    def __init__(s, d, drop):
        super().__init__(); s.he = Enc(3, d, drop); s.te = Enc(1, d, drop); s.q = nn.Conv1d(d, d, 1); s.k = nn.Conv1d(d, d, 1); s.vv = nn.Conv1d(d, d, 1); s.sc = d ** -0.5
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 3 * K))
        s.head[-1].weight.data *= 0.01; s.head[-1].bias.data.zero_()

    def forward(s, H, G, last_tvt):
        h = s.he(H); t = s.te(G); Q = s.q(h).transpose(1, 2); Kk = s.k(t).transpose(1, 2); V = s.vv(t).transpose(1, 2)
        ctx = torch.softmax(Q @ Kk.transpose(1, 2) * s.sc, 2) @ V; x = torch.cat([h.transpose(1, 2), ctx], 2)
        o = s.head(x); B, L, _ = o.shape; o = o.view(B, L, K, 3)
        base = torch.linspace(-30., 30., K, device=o.device)[None, None, :]
        mu = last_tvt[:, None, None] + base + torch.tanh(o[..., 0]) * CFG['max_off']
        sig = CFG['min_sig'] + (18. - CFG['min_sig']) * torch.sigmoid(o[..., 1])
        logpi = F.log_softmax(o[..., 2], dim=2)
        return mu, sig, logpi


def mdn_nll(mu, sig, logpi, y):
    z = (y[..., None] - mu) / sig; comp = logpi - 0.5 * z * z - torch.log(sig) - 0.9189
    return -torch.logsumexp(comp, dim=2)


def viterbi(mu, sig, logpi, trans):
    L, Kk = mu.shape; em = logpi
    D = em[0].copy(); bp = np.zeros((L, Kk), np.int32)
    for i in range(1, L):
        tc = -trans * (mu[i][None, :] - mu[i - 1][:, None]) ** 2
        tot = D[:, None] + tc; bp[i] = np.argmax(tot, 0); D = em[i] + tot[bp[i], np.arange(Kk)]
    path = np.zeros(L, np.int32); path[-1] = np.argmax(D)
    for i in range(L - 1, 0, -1): path[i - 1] = bp[i, path[i]]
    return mu[np.arange(L), path]


def make(seed):
    torch.manual_seed(seed); net = MDN(CFG['d'], CFG['drop']).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=CFG['epochs'] * ((len(TR) + CFG['bs'] - 1) // CFG['bs']))
    return net, opt, sch


def batch(ws, train=False):
    H = np.stack([w['H'] for w in ws]).copy(); gn = np.stack([w['gn'][:TG] for w in ws]).copy()
    tvt = np.stack([w['tvt'] for w in ws]).copy(); ev = np.stack([w['ev'] for w in ws])
    if train:
        sc = 1 + np.random.randn(len(ws), 1, 1) * CFG['aug_caljit']; sh = np.random.randn(len(ws), 1, 1) * CFG['aug_caljit']
        H[:, 0:1] = H[:, 0:1] * sc + sh; gn = gn * sc[:, :, 0] + sh[:, :, 0]; H = H + np.random.randn(*H.shape) * CFG['aug_noise']
        if CFG['aug_warp'] > 0:
            m = EVSTEPS; src = np.arange(m)
            for bi in range(len(ws)):
                d = np.cumsum(np.abs(1 + np.random.randn(m) * CFG['aug_warp'])); d = (d - d[0]) / (d[-1] - d[0]) * (m - 1)
                for ch in range(H.shape[1]): H[bi, ch, KSTEPS:] = np.interp(d, src, H[bi, ch, KSTEPS:])
                tvt[bi, KSTEPS:] = np.interp(d, src, tvt[bi, KSTEPS:])
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    return to(H), to(gn)[:, None], to(np.array([w['last_tvt'] for w in ws])), to(tvt), to(ev)


def evaluate(net):
    net.eval(); e = []
    with torch.no_grad():
        for i in range(0, len(VA), CFG['bs']):
            ws = VA[i:i + CFG['bs']]; H, G, lt, tvt, ev = batch(ws); mu, sig, lp = net(H, G, lt)
            mu = mu.cpu().numpy(); lp = lp.cpu().numpy(); sg = sig.cpu().numpy(); tv = tvt.cpu().numpy(); em = ev.cpu().numpy() > 0.5
            for b in range(len(ws)):
                path = viterbi(mu[b], sg[b], lp[b], CFG['trans']); e.append((path[em[b]] - tv[b][em[b]]) ** 2)
    net.train(); return float(np.sqrt(np.mean(np.concatenate(e))))


net, opt, sch = make(CFG['seed']); ema = copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]; edec = CFG['ema_decay']
print('training MDN+prefix-cut-aug K=%d | params=%.0fk | epochs=%d' % (K, sum(p.numel() for p in net.parameters()) / 1e3, CFG['epochs']), flush=True)
t0 = time.time(); best = (99, 0); bestema = (99, 0); order = np.arange(len(TR))
best_ema_state = None
for ep in range(CFG['epochs']):
    np.random.shuffle(order)
    for i in range(0, len(TR), CFG['bs']):
        H, G, lt, tvt, ev = batch([TR[j] for j in order[i:i + CFG['bs']]], True); mu, sig, lp = net(H, G, lt)
        m = ev + CFG['w_known'] * (1 - ev); nll = mdn_nll(mu, sig, lp, tvt); loss = (nll * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(edec).add_(pn, alpha=1 - edec)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
    if (ep + 1) % 2 == 0 or ep == CFG['epochs'] - 1:
        r = evaluate(net); re = evaluate(ema)
        if r < best[0]: best = (r, ep + 1)
        if re < bestema[0]:
            bestema = (re, ep + 1)
            best_ema_state = {k: v.clone() for k, v in ema.state_dict().items()}
        print('  ep%2d | raw %.3f (best %.3f) | EMA %.3f (best %.3f@%d) %.0fs' % (ep + 1, r, best[0], re, bestema[0], bestema[1], time.time() - t0), flush=True)
if best_ema_state is not None:
    ema.load_state_dict(best_ema_state)
    torch.save(best_ema_state, f"{CFG['tag']}_best_ema.pt")
print('DONE raw %.3f | EMA %.3f@%d (RESTORED to best checkpoint, saved %s_best_ema.pt) (flat~15, WARP~11, PF~7, Tucker~5.4) -- MDN+prefix-cut-aug' % (best[0], bestema[0], bestema[1], CFG['tag']), flush=True)
