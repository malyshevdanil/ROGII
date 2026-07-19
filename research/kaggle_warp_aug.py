"""WARP (4-channel, cross-attention, cumsum-integration) + PREFIX-CUT AUGMENTATION, no synthetic data.
WARP alone (real-only, no augmentation) already reaches ~11.3-11.7 on the true holdout -- notably
stronger than MDN's from-scratch ~14 -- so applying the SAME prefix-cut augmentation mechanism that
gave MDN a real ~0.6-0.7ft gain (~14->13.35/13.42) to this already-stronger base architecture is the
next natural test: does augmentation push WARP meaningfully below its unaugmented ~11.3 ceiling?
Prefix-cut: for TRAIN wells only, also train on `n_cuts` additional cuts where only a `cut_frac`
PREFIX of the known zone is used to compute the last-known anchor / calibration / known-zone tail
window -- the real eval targets stay UNCHANGED (100% real supervision, forces robustness to a
staler/poorer anchor). VA always uses cut_frac=1.0.
"""
import numpy as np, pandas as pd, glob, os, time, copy, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CFG = dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.20, wd=3e-4, lr=1.2e-3,
           epochs=int(os.environ.get('EPOCHS', '120')), bs=24, maxstep=1.5, smooth=0.06, w_known=0.5,
           ema_decay=0.995, n_val=160, seed=42, tag='warp_aug_tw', n_cuts=3, cut_lo=0.3, cut_hi=0.9)
TG = CFG['TWGRID']; KS = CFG['KSTEPS']; ES = CFG['EVSTEPS']
TRAIN_DIR = next((d for d in ('data/train', 'd:/ROGII/data/train') if os.path.isdir(d)), 'data/train')


def inn(a):
    a = a.copy(); n = len(a); idx = np.arange(n); m = np.isnan(a)
    if m.all(): return np.zeros(n)
    a[m] = np.interp(idx[m], idx[~m], a[~m]); return a


def build_well(wid, cut_frac=1.0):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tt = tw['TVT'].values.astype(float); tg = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    if len(tt) < 10: return None
    kn_idx = np.where(hw['TVT_input'].notna().values)[0]
    if len(kn_idx) < 20 or hw['TVT_input'].isna().sum() < 20: return None
    n_use = max(20, int(len(kn_idx) * cut_frac)); kn_idx_use = kn_idx[:n_use]
    last_i = int(kn_idx_use[-1]); last_tvt = float(hw['TVT_input'].iloc[last_i])
    n = len(hw); gr = inn(hw['GR'].values.astype(float))
    kg = gr[kn_idx_use]; ktvt = hw['TVT_input'].values[kn_idx_use]
    twk = np.interp(ktvt, tt, tg); v = np.isfinite(kg) & np.isfinite(twk)
    a, b = (np.polyfit(kg[v], twk[v], 1) if v.sum() >= 20 else (1., 0.)); cal = gr * a + b
    Z = hw['Z'].values.astype(float); MD = hw['MD'].values.astype(float); mdd = np.gradient(MD); mdd[mdd == 0] = 1
    grad = np.gradient(gr); rstd = pd.Series(gr).rolling(21, center=True, min_periods=1).std().fillna(0).values
    dz = np.gradient(Z) / mdd
    ev = hw['TVT_input'].isna().values.astype(float); ei = np.where(ev > 0.5)[0]
    if len(ei) < 5: return None
    e0 = ei[0]
    ks = np.arange(max(0, last_i + 1 - 400), last_i + 1)
    if len(ks) < 5: return None
    g_tvt = np.linspace(tt.min(), tt.max(), TG); g_gr = np.interp(g_tvt, tt, tg); gm, gs = float(g_gr.mean()), float(g_gr.std() + 1e-6)
    caln = (cal - gm) / gs; gradn = grad / gs; rstdn = rstd / gs; true = hw['TVT'].values.astype(float)
    kd = np.linspace(ks[0], ks[-1], KS); ed = np.linspace(e0, n - 1, ES); dst = np.concatenate([kd, ed]); src = np.arange(n)
    R = lambda x: np.interp(dst, src, x).astype(np.float32)
    evr = np.concatenate([np.zeros(KS), np.ones(ES)]).astype(np.float32)
    return dict(H=np.stack([R(caln), R(gradn), R(rstdn), R(dz)]).astype(np.float32),
                gn=((g_gr - gm) / gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt)


print('building...', flush=True); t0 = time.time()
wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
BASE = {}
for w in wids:
    b = build_well(w, 1.0)
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
        b = build_well(w, float(f))
        if b is not None: DATA_TR[f'{w}__cut{len(DATA_TR)}'] = b
TR = list(DATA_TR.values())
d3 = np.concatenate([w['H'][3] for w in TR]); m3, s3 = d3.mean(), d3.std() + 1e-6
for w in TR: w['H'][3] = (w['H'][3] - m3) / s3
for w in VA: w['H'][3] = (w['H'][3] - m3) / s3
print('built VA=%d TR_orig=%d TR_with_cuts=%d %.0fs' % (len(VA), len(TR_wids), len(TR), time.time() - t0), flush=True)


def prmse(sqs): return float(np.sqrt(np.mean(np.concatenate(sqs))))


print('flat=%.3f' % prmse([(w['last_tvt'] - w['tvt'][w['ev'] > 0.5]) ** 2 for w in VA]), flush=True)


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
        s.head[-1].weight.data *= 0.01; s.head[-1].bias.data.zero_()

    def forward(s, H, G, lt):
        h = s.he(H); t = s.te(G)
        Q = s.q(h).transpose(1, 2); K = s.k(t).transpose(1, 2); V = s.vv(t).transpose(1, 2)
        att = torch.softmax(Q @ K.transpose(1, 2) * s.sc, dim=2); ctx = att @ V
        x = torch.cat([h.transpose(1, 2), ctx], dim=2)
        dt = torch.tanh(s.head(x)[..., 0]) * CFG['maxstep']; tvt = torch.cumsum(dt, 1)
        return tvt - tvt[:, KS - 1:KS] + lt[:, None]


AUG_WARP = 0.35  # time-warp strength: historically took WARP v1(12.0)->v2(11.45) in an earlier session;
# stretches/compresses the eval-zone MD sequence via a random cumulative warp -- complementary to
# prefix-cut (which varies the KNOWN-side anchor/context, not the eval sequence's own spatial scale).


def batch(ws, train=False):
    H = np.stack([w['H'] for w in ws]).copy(); gn = np.stack([w['gn'][:TG] for w in ws]).copy()
    tvt = np.stack([w['tvt'] for w in ws]).copy(); ev = np.stack([w['ev'] for w in ws])
    if train:
        sc = 1 + np.random.randn(len(ws), 1, 1) * 0.12; sh = np.random.randn(len(ws), 1, 1) * 0.12
        H[:, 0:1] = H[:, 0:1] * sc + sh; gn = gn * sc[:, :, 0] + sh[:, :, 0]; H = H + np.random.randn(*H.shape) * 0.07
        if AUG_WARP > 0:
            m = ES; src = np.arange(m)
            for bi in range(len(ws)):
                d = np.cumsum(np.abs(1 + np.random.randn(m) * AUG_WARP)); d = (d - d[0]) / (d[-1] - d[0]) * (m - 1)
                for ch in range(H.shape[1]): H[bi, ch, KS:] = np.interp(d, src, H[bi, ch, KS:])
                tvt[bi, KS:] = np.interp(d, src, tvt[bi, KS:])
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    return to(H), to(gn)[:, None], to([w['last_tvt'] for w in ws]), to(tvt), to(ev)


net = WarpNet(CFG['d'], CFG['drop']).to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
STEPS_PER_EPOCH = (len(TR) + CFG['bs'] - 1) // CFG['bs']
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=CFG['epochs'] * STEPS_PER_EPOCH)
ema = copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]; edec = CFG['ema_decay']


def evaluate(m):
    m.eval(); e = []
    with torch.no_grad():
        for i in range(0, len(VA), CFG['bs']):
            H, G, lt, tvt, ev = batch(VA[i:i + CFG['bs']]); p = m(H, G, lt); e.append(((p - tvt)[ev > 0.5]).cpu().numpy())
    m.train(); return float(np.sqrt((np.concatenate(e) ** 2).mean()))


print('training WARP+prefix-cut-aug | params=%.0fk | epochs=%d' % (sum(p.numel() for p in net.parameters()) / 1e3, CFG['epochs']), flush=True)
t0 = time.time(); best = (99, 0); bestema = (99, 0); order = np.arange(len(TR)); best_ema_state = None
for ep in range(CFG['epochs']):
    np.random.shuffle(order)
    for i in range(0, len(TR), CFG['bs']):
        H, G, lt, tvt, ev = batch([TR[j] for j in order[i:i + CFG['bs']]], True); p = net(H, G, lt)
        m = ev + CFG['w_known'] * (1 - ev)
        loss = (F.smooth_l1_loss(p, tvt, reduction='none', beta=8.0) * m).sum() / m.sum()
        d2 = p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]; loss = loss + CFG['smooth'] * (d2 * d2 * m[:, 2:]).sum() / m[:, 2:].sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(edec).add_(pn, alpha=1 - edec)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
    if (ep + 1) % 3 == 0 or ep == CFG['epochs'] - 1:
        r = evaluate(net); re = evaluate(ema)
        if r < best[0]: best = (r, ep + 1)
        if re < bestema[0]:
            bestema = (re, ep + 1)
            best_ema_state = {k: v.clone() for k, v in ema.state_dict().items()}
        print('  ep%3d | raw %.3f (best %.3f) | EMA %.3f (best %.3f@%d) %.0fs' %
              (ep + 1, r, best[0], re, bestema[0], bestema[1], time.time() - t0), flush=True)
if best_ema_state is not None:
    torch.save(best_ema_state, f"{CFG['tag']}_best_ema.pt")
print('DONE raw %.3f | EMA %.3f@%d (flat~15, WARP-no-aug~11.3-11.7, PF~7, Tucker~5.4) -- WARP+prefix-cut-aug' %
      (best[0], bestema[0], bestema[1]), flush=True)
