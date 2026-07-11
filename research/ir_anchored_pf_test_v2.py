"""New, more targeted hypothesis for 'fix physics inside the tracker' (not a transplant of pf_z, which
failed -- likely because merging two models erases the decorrelation that made blending them work).

pf_ancc's `rate` (d(surface)/dMD) currently evolves as a pure AR(1) that decays toward ZERO:
    rate[j] = ALPHA*rate[j] + RN*randn()          (ALPHA=0.998)
Fig.12 (our own paper) shows PF error grows with eval-zone length. One concrete, physically-motivated
reason: over a long eval zone this slow decay pulls `rate` away from the well's OWN known-zone dip `ir`
and toward a generic 0, even when the true local dip is persistently non-zero (we measured ir=0.30 on one
well). Fix: anchor the mean-reversion target to `ir` instead of 0:
    rate[j] = ir + ALPHA*(rate[j]-ir) + RN*randn()
This is a one-line change to an assumption that is plausibly just wrong, not a transplant of a second,
decorrelated model -- so it should not suffer the same "erases diversity" failure mode.

Tested with proper seed-averaging AND cross-fit from the start (lesson learned from the first attempt,
where a single-seed sweep showed a mirage that vanished under averaging).
"""
import numpy as np, pandas as pd, glob, os
from numba import njit

TRAIN_DIR = 'd:/ROGII/data/train'
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001
PF_RESAMP = 0.5
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.

@njit(cache=False)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=False)
def _seed_jit(seed):
    np.random.seed(seed)

@njit(cache=False)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    u0 = np.random.uniform(0., 1./N); np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0+j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci]+rp*np.random.randn(); na[j] = aux[ci]+rv*np.random.randn()
    return np2, na

@njit(cache=False)
def _pf_ancc_variant(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP,
                      ir_anchor):
    """ir_anchor=0.0 -> identical to production _pf_ancc (decays to 0).
    ir_anchor=1.0 -> decays fully toward `ir` instead of 0.
    Values in between blend the two targets."""
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls+IS*np.random.randn(); rate[j] = ir+0.01*np.random.randn()
    pts = np.empty(len(md_v)); pm = md_v[0]-1.
    target = ir_anchor * ir
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = target + ALPHA*(rate[j]-target) + RN*np.random.randn()
            pos[j] += rate[j]*dm+PN*np.random.randn()
            tvt_j = pos[j]-z_v[i]; tvt_j = max(tvt_j, vmin-50.); tvt_j = min(tvt_j, vmin+len(gg)*step+50.)
            pos[j] = tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs
                lk = max(np.exp(-0.5*d*d) if d*d < 600. else 0., 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
        tv = 0.
        for j in range(N): tv += w[j]*(pos[j]-z_v[i])
        pts[i] = tv; pm = md_v[i]
    return pts

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn.GR.values-np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))

def load_well(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(kn) < 50 or len(ev) < 50: return None
    return hw, tw_tvt, tw_gr, kn, ev

def run_one(wid, ir_anchor, seed, N=300):
    _seed_jit(seed)
    r = load_well(wid)
    if r is None: return None
    hw, tw_tvt, tw_gr, kn, ev = r
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values)
    m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    md_v = ev.MD.values.astype(np.float64); z_v = ev.Z.values.astype(np.float64); gr_v = ev.GR.values.astype(np.float64)
    true_tvt = ev.TVT.values.astype(np.float64)
    pts = _pf_ancc_variant(md_v, z_v, gr_v, gg, gmin, gst, gs, ls, ir, N,
                            ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP, ir_anchor)
    return pts, true_tvt

def pooled_rmse(wid_list, ir_anchor, seed=0, N=300):
    sq, n = 0.0, 0
    for wid in wid_list:
        r = run_one(wid, ir_anchor, seed, N)
        if r is None: continue
        pts, true_tvt = r
        sq += np.sum((pts - true_tvt) ** 2); n += len(true_tvt)
    return float(np.sqrt(sq / n)) if n > 0 else np.nan

if __name__ == '__main__':
    wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    rng = np.random.default_rng(7)
    sample = list(rng.choice(wids, size=120, replace=False))

    ANCHORS = [0.0, 0.25, 0.5, 0.75, 1.0]
    SEEDS = [1, 2, 3, 4]

    print('=== seed-averaged sweep (4 seeds x 120 wells), ir_anchor=0 is the production baseline ===')
    for a in ANCHORS:
        vals = [pooled_rmse(sample, a, seed=sd) for sd in SEEDS]
        print(f'ir_anchor={a:.2f}  per-seed={[round(v,3) for v in vals]}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}')

    print('\n=== cross-fit: fit best anchor on H1 (2 seeds), test on H2 (2 seeds), and vice versa ===')
    folds = np.array_split(np.array(sample), 2)
    H1, H2 = list(folds[0]), list(folds[1])
    FIT_SEEDS = [1, 2]; TEST_SEEDS = [3, 4]

    def best_on(wl):
        best = (1e9, None)
        for a in ANCHORS:
            v = np.mean([pooled_rmse(wl, a, seed=sd) for sd in FIT_SEEDS])
            if v < best[0]: best = (v, a)
        return best

    b1 = best_on(H1)
    v_h2 = np.mean([pooled_rmse(H2, b1[1], seed=sd) for sd in TEST_SEEDS])
    base_h2 = np.mean([pooled_rmse(H2, 0.0, seed=sd) for sd in TEST_SEEDS])
    print(f'fit on H1: best anchor={b1[1]} (train={b1[0]:.4f}) -> H2 test={v_h2:.4f} vs H2 baseline(anchor=0)={base_h2:.4f}')

    b2 = best_on(H2)
    v_h1 = np.mean([pooled_rmse(H1, b2[1], seed=sd) for sd in TEST_SEEDS])
    base_h1 = np.mean([pooled_rmse(H1, 0.0, seed=sd) for sd in TEST_SEEDS])
    print(f'fit on H2: best anchor={b2[1]} (train={b2[0]:.4f}) -> H1 test={v_h1:.4f} vs H1 baseline(anchor=0)={base_h1:.4f}')
