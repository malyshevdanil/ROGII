"""DIAGNOSTIC: why did synthetic pretrain (kaggle_synth2.py, NOISE=boot/gauss) fail to transfer to real
(finetune ~13, worse than real-only WARP's 11.3)? Two competing hypotheses:
  (A) synthetic task is TOO EASY (not enough self-similarity/ambiguity) -> model overfits/memorizes a
      clean GR->TVT inversion that doesn't match real's genuine ambiguity -> should show SYNTH-HOLDOUT
      error dropping much lower than REAL-holdout error ever gets (e.g. synth->2-3 while real stays ~15).
  (B) synthetic task is roughly as hard as real, but there's a DISTRIBUTION SHIFT (typewell/noise
      statistics differ enough that the learned features don't transfer) -> synth-holdout error would
      stay in a similar ~11-15 range as real, but real-holdout still wouldn't improve during pretraining.
This script tracks BOTH synth-holdout and real-holdout error every few epochs during synthetic-only
training (no finetuning yet) to distinguish the two failure modes before redesigning the generator.
"""
import numpy as np, pandas as pd, glob, os, time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NOISE = os.environ.get('NOISE', 'boot')
EPOCHS = int(os.environ.get('EPOCHS', '24'))
CFG = dict(d=128, KSTEPS=60, EVSTEPS=360, TWGRID=256, drop=0.2, wd=3e-4, lr=1.2e-3, bs=24,
           maxstep=1.5, smooth=0.06, w_known=0.5, n_val=160, seed=42)
TG = CFG['TWGRID']; KS = CFG['KSTEPS']; ES = CFG['EVSTEPS']; L = KS + ES
TRAIN_DIR = next((d for d in ('data/train', 'd:/ROGII/data/train') if os.path.isdir(d)), 'data/train')


def inn(a):
    a = a.copy(); n = len(a); idx = np.arange(n); m = np.isnan(a)
    if m.all(): return np.zeros(n)
    a[m] = np.interp(idx[m], idx[~m], a[~m]); return a


def build_well(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
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
    perf = (np.interp(true, tt, tg) - gm) / gs
    resid = (R(caln) - R(perf)).astype(np.float32)
    return dict(H=np.stack([R(caln), R(gradn), R(rstdn), R(dz)]).astype(np.float32),
                gn=((g_gr - gm) / gs).astype(np.float32), tvt=R(true), ev=evr, last_tvt=last_tvt,
                resid=resid, tt=tt, tg=tg, gm=gm, gs=gs, dzr=R(dz).astype(np.float32))


print('building real wells...', flush=True); t0 = time.time()
wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
DATA = [b for b in (build_well(w) for w in wids) if b is not None]
d3 = np.concatenate([w['H'][3] for w in DATA]); m3, s3 = d3.mean(), d3.std() + 1e-6
for w in DATA:
    w['H'][3] = (w['H'][3] - m3) / s3; w['dzr'] = (w['dzr'] - m3) / s3
rng = np.random.default_rng(CFG['seed']); idx = np.arange(len(DATA)); rng.shuffle(idx)
VA = [DATA[i] for i in idx[:CFG['n_val']]]; TRr = [DATA[i] for i in idx[CFG['n_val']:]]
print('built %d (%d/%d) %.0fs | NOISE=%s EPOCHS=%d' % (len(DATA), len(TRr), len(VA), time.time() - t0, NOISE, EPOCHS), flush=True)


def prmse(sqs): return float(np.sqrt(np.mean(np.concatenate(sqs))))


print('flat=%.3f' % prmse([(w['last_tvt'] - w['tvt'][w['ev'] > 0.5]) ** 2 for w in VA]), flush=True)

RESID = [w['resid'] for w in TRr]
TWS = [(w['tt'], w['tg'], w['gm'], w['gs']) for w in TRr]
DZR = [w['dzr'] for w in TRr]
_dt = np.concatenate([np.diff(w['tvt'][w['ev'] > 0.5]) for w in TRr]); DTSD = float(np.std(_dt))
print('real eval dTVT std=%.4f' % DTSD, flush=True)


def gen_synth(nb, rng_local=None):
    rr = rng_local if rng_local is not None else np.random
    out = []
    for _ in range(nb):
        ti = rr.randint(len(TWS)); tt, tg, gm, gs = TWS[ti]
        last_tvt = float(rr.uniform(tt.min() + 50, tt.max() - 50))
        dips = np.zeros(L)
        nseg = rr.randint(2, 7); bps = np.sort(rr.choice(np.arange(KS, L), nseg, replace=False))
        cur = rr.randn() * DTSD * 1.3; p = 0
        for bp in list(bps) + [L]:
            dips[p:bp] = cur; cur = cur + rr.randn() * DTSD * 1.1
            if rr.rand() < 0.25: cur += rr.choice([-1, 1]) * rr.uniform(2, 6) * DTSD
            p = bp
        path = np.cumsum(dips); path = path - path[KS - 1] + last_tvt
        path = np.clip(path, tt.min() + 5, tt.max() - 5)
        perf = (np.interp(path, tt, tg) - gm) / gs
        if NOISE == 'boot':
            nz = RESID[rr.randint(len(RESID))].copy()
            if rr.rand() < 0.5: nz = nz[::-1].copy()
            nz = nz * rr.uniform(0.7, 1.2)
        else:
            sd = np.median([np.std(r) for r in RESID[:50]]); nz = rr.randn(L).astype(np.float32) * sd
        caln = (perf + nz).astype(np.float32)
        gradn = np.gradient(caln).astype(np.float32)
        rstdn = pd.Series(caln).rolling(21, center=True, min_periods=1).std().fillna(0).values.astype(np.float32)
        dz = DZR[rr.randint(len(DZR))].astype(np.float32)
        H = np.stack([caln, gradn, rstdn, dz]).astype(np.float32)
        evr = np.concatenate([np.zeros(KS), np.ones(ES)]).astype(np.float32)
        out.append(dict(H=H, gn=((np.interp(np.linspace(tt.min(), tt.max(), TG), tt, tg) - gm) / gs).astype(np.float32),
                         tvt=path.astype(np.float32), ev=evr, last_tvt=last_tvt))
    return out


class _RngShim:
    def __init__(self, rng): self.rng = rng
    def randint(self, a, b=None):
        if b is None: a, b = 0, a
        return int(self.rng.integers(a, b))
    def uniform(self, a, b): return self.rng.uniform(a, b)
    def randn(self): return float(self.rng.standard_normal())
    def choice(self, arr, size=None, replace=True):
        return self.rng.choice(arr, size=size, replace=replace)
    def rand(self): return float(self.rng.random())


# FIXED synthetic holdout (generated once, held constant across all epochs, to measure generalization
# on the synthetic distribution itself, analogous to VA for real)
_synth_rng = _RngShim(np.random.default_rng(999))
SYNTH_VA = gen_synth(160, _synth_rng)
print('built fixed synthetic holdout: 160 wells', flush=True)


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


def batch(ws):
    to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)
    H = np.stack([w['H'] for w in ws]); gn = np.stack([w['gn'][:TG] for w in ws])
    return to(H), to(gn)[:, None], to([w['last_tvt'] for w in ws]), to(np.stack([w['tvt'] for w in ws])), to(np.stack([w['ev'] for w in ws]))


net = WarpNet(CFG['d'], CFG['drop']).to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
STEPS_PER = 200
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG['lr'], total_steps=EPOCHS * STEPS_PER)


def evaluate(ws):
    net.eval(); e = []
    with torch.no_grad():
        for i in range(0, len(ws), CFG['bs']):
            H, G, lt, tvt, ev = batch(ws[i:i + CFG['bs']]); p = net(H, G, lt); e.append(((p - tvt)[ev > 0.5]).cpu().numpy())
    net.train(); return float(np.sqrt((np.concatenate(e) ** 2).mean()))


print('training WARP on %s synthetic | tracking BOTH synth-holdout and REAL-holdout' % NOISE, flush=True)
t0 = time.time(); best_real = (99, 0)
for ep in range(EPOCHS):
    tot = 0
    for _ in range(STEPS_PER):
        ws = gen_synth(CFG['bs'])
        H, G, lt, tvt, ev = batch(ws); p = net(H, G, lt)
        m = ev + CFG['w_known'] * (1 - ev)
        loss = (F.smooth_l1_loss(p, tvt, reduction='none', beta=8.0) * m).sum() / m.sum()
        d2 = p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]; loss = loss + CFG['smooth'] * (d2 * d2 * m[:, 2:]).sum() / m[:, 2:].sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
    if (ep + 1) % 3 == 0 or ep == EPOCHS - 1:
        r_synth = evaluate(SYNTH_VA)
        r_real = evaluate(VA)
        if r_real < best_real[0]: best_real = (r_real, ep + 1)
        print('  ep%2d loss%.3f | SYNTH-holdout %.3f | REAL-holdout %.3f (best %.3f@%d) %.0fs' %
              (ep + 1, tot / STEPS_PER, r_synth, r_real, best_real[0], best_real[1], time.time() - t0), flush=True)
print('DONE. If SYNTH-holdout << REAL-holdout throughout -> hypothesis A (task too easy). '
      'If both stay similar (~11-16) and neither improves much -> hypothesis B (distribution shift).', flush=True)
