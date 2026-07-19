"""v3 generator: PAIRED, REAL-ANCHORED counterfactual-eval augmentation.

Control-run finding that motivated this: a model trained on REAL data only scores 20-24 on the
v1/v2 cross-well-paired synthetic holdout (worse than flat=14.7), while the same harness reaches
~11.7-11.8 on the real holdout (confirms harness is fine). And a model trained on the v2 (range-
calibrated) synthetic collapses to flat (~14.7) on BOTH synth and real holdout without improving.
Together these say: the v1/v2 generator's cross-well pairing (random typewell + a DIFFERENT random
well's bootstrapped residual + a DIFFERENT random well's dz channel, fully-synthetic known zone) is
itself an out-of-distribution artifact, not just a noise-realism problem.

Fix tested here: keep the KNOWN zone, typewell, dz channel, and residual-noise bank ALL from the
SAME real well (paired, not cross-well) -- preserves whatever real per-well calibration/
autocorrelation consistency the model exploits. Only the EVAL-zone TVT trajectory is replaced with
an alternative, equally-plausible AR(1) continuation (anchored to continue from the well's own real
last-known dip, not dip=0), giving unlimited counterfactual "what happens next" augmentation from
the 613 real known-zones instead of inventing typewell pairings from scratch.
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

_dt = np.concatenate([np.diff(w['tvt'][w['ev'] > 0.5]) for w in TRr]); DTSD = float(np.std(_dt))
print('real eval dTVT std=%.4f' % DTSD, flush=True)

AR1_RHO = 0.83
AR1_EPS_STD = 0.1865
AR1_FAULT_P = 0.005
AR1_FAULT_SCALE = 5.0


def _gen_path_ar1(n, rho, eps_std, fault_p, fault_scale, target_step_std, rr, dip0=0.0):
    dip = dip0; cur = 0.0; path = np.zeros(n)
    for i in range(n):
        dip = rho * dip + rr.randn() * eps_std
        if fault_p > 0 and rr.rand() < fault_p:
            dip += rr.choice([-1, 1]) * rr.uniform(1, fault_scale) * target_step_std
        cur += dip; path[i] = cur
    return path


def gen_synth(nb, rng_local=None):
    rr = rng_local if rng_local is not None else np.random
    out = []
    for _ in range(nb):
        ti = rr.randint(len(TRr)); w = TRr[ti]
        tt, tg, gm, gs = w['tt'], w['tg'], w['gm'], w['gs']
        last_tvt = w['last_tvt']
        tail = w['tvt'][max(0, KS - 6):KS]
        dip0 = float(np.median(np.diff(tail))) if len(tail) >= 2 else 0.0
        raw = _gen_path_ar1(ES, AR1_RHO, AR1_EPS_STD, AR1_FAULT_P, AR1_FAULT_SCALE, DTSD, rr, dip0=dip0)
        eval_path = raw + last_tvt
        eval_path = np.clip(eval_path, tt.min() + 5, tt.max() - 5)
        perf_eval = (np.interp(eval_path, tt, tg) - gm) / gs
        own_resid_eval = w['resid'][KS:].copy()
        if rr.rand() < 0.5: own_resid_eval = own_resid_eval[::-1].copy()
        nz_eval = own_resid_eval * rr.uniform(0.85, 1.15)
        caln = np.concatenate([w['H'][0, :KS], (perf_eval + nz_eval).astype(np.float32)]).astype(np.float32)
        gradn = np.gradient(caln).astype(np.float32)
        rstdn = pd.Series(caln).rolling(21, center=True, min_periods=1).std().fillna(0).values.astype(np.float32)
        dz = w['dzr'].astype(np.float32)
        H = np.stack([caln, gradn, rstdn, dz]).astype(np.float32)
        evr = np.concatenate([np.zeros(KS), np.ones(ES)]).astype(np.float32)
        tvt_full = np.concatenate([w['tvt'][:KS], eval_path]).astype(np.float32)
        out.append(dict(H=H, gn=w['gn'].astype(np.float32), tvt=tvt_full, ev=evr, last_tvt=last_tvt))
    return out


class _RngShim:
    def __init__(self, rng): self.rng = rng
    def randint(self, a, b=None):
        if b is None: a, b = 0, a
        return int(self.rng.integers(a, b))
    def uniform(self, a, b): return self.rng.uniform(a, b)
    def randn(self, *shape):
        if not shape: return float(self.rng.standard_normal())
        return self.rng.standard_normal(shape if len(shape) > 1 else shape[0])
    def choice(self, arr, size=None, replace=True):
        return self.rng.choice(arr, size=size, replace=replace)
    def rand(self): return float(self.rng.random())


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


print('training WARP on %s synthetic (v3 real-anchored) | tracking BOTH synth-holdout and REAL-holdout' % NOISE, flush=True)
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
print('DONE v3 (real-anchored). Compare vs v2 (flat ~14.7/14.7) and real-only-control (real~11.7, synth~20-24).', flush=True)
