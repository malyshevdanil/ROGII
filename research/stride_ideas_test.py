"""Two concrete, cheap-to-test ideas extracted from the strongest competitor writeup we read (STRIDE,
~5.7 public LB, rank ~7):

(A) Cauchy (heavy-tailed) GR likelihood instead of Gaussian. Their claim: a heavy-tailed distance resists
    being dragged by a single strong-but-wrong (aliased/repeated-bed) match, because the penalty for a
    bad candidate saturates instead of growing quadratically forever. Our PF currently uses a plain Gaussian
    kernel exp(-0.5*d^2). Swap for a Cauchy-style kernel 1/(1+d^2) at a tunable scale.

(B) A "typewell reference term": recalibrate the typewell's GR level to THIS well's own known-section GR
    (an affine gain+offset fit on the known zone) BEFORE computing the likelihood, not as an external
    post-hoc position-matching scheme (which is what our own failed shift-scan test tried). Our PF currently
    compares raw GR against the raw typewell curve with only a noise-SCALE parameter (gs), no level/gain
    calibration at all -- this is a real, previously-untested gap.

Tested with seed-averaging AND cross-fit validation from the start (lesson learned this session: single-seed
sweeps produce mirages that vanish or reverse under averaging).
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
def _pf_ancc_v3(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP,
                lk_kind, cauchy_scale, gr_gain, gr_offset):
    """lk_kind: 0=Gaussian (production), 1=Cauchy.
    gr_v is calibrated as gr_v*gr_gain + gr_offset before the emission comparison (gain=1,offset=0 -> raw)."""
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls+IS*np.random.randn(); rate[j] = ir+0.01*np.random.randn()
    pts = np.empty(len(md_v)); pm = md_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA*rate[j]+RN*np.random.randn(); pos[j] += rate[j]*dm+PN*np.random.randn()
            tvt_j = pos[j]-z_v[i]; tvt_j = max(tvt_j, vmin-50.); tvt_j = min(tvt_j, vmin+len(gg)*step+50.)
            pos[j] = tvt_j+z_v[i]
        gv = gr_v[i]*gr_gain+gr_offset
        if not np.isnan(gv):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gv-eg)/gs
                if lk_kind == 0:
                    lk = max(np.exp(-0.5*d*d) if d*d < 600. else 0., 1e-300)
                else:
                    cs = cauchy_scale
                    lk = 1./(1.+(d/cs)*(d/cs))
                w[j] *= lk; ws += w[j]
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

def _fit_affine(x, y):
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 20: return 1.0, 0.0
    a, b = np.polyfit(x[v], y[v], 1)
    return float(a), float(b)

def load_well(wid):
    hw = pd.read_csv(f'{TRAIN_DIR}/{wid}__horizontal_well.csv')
    tw = pd.read_csv(f'{TRAIN_DIR}/{wid}__typewell.csv').sort_values('TVT')
    if 'TVT' not in hw.columns: return None
    tw_tvt = tw['TVT'].values.astype(float); tw_gr = tw['GR'].fillna(tw['GR'].mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(kn) < 50 or len(ev) < 50: return None
    return hw, tw_tvt, tw_gr, kn, ev

def run_one(wid, seed, lk_kind=0, cauchy_scale=1.0, recalibrate=False, N=300):
    np.random.seed(seed)
    r = load_well(wid)
    if r is None: return None
    hw, tw_tvt, tw_gr, kn, ev = r
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values)
    m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)

    gr_gain, gr_offset = 1.0, 0.0
    if recalibrate:
        kn_gr = kn['GR'].interpolate().bfill().ffill().values.astype(float)
        kn_twg = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
        # STRIDE's framing: recalibrate the TYPEWELL toward this well's own GR. Equivalent (up to inverting
        # the affine map) to calibrating this well's GR toward the typewell, which is what we implement here
        # for a minimal, single-line change to the emission comparison.
        a, b = _fit_affine(kn_gr, kn_twg)
        gr_gain, gr_offset = a, b

    md_v = ev.MD.values.astype(np.float64); z_v = ev.Z.values.astype(np.float64); gr_v = ev.GR.values.astype(np.float64)
    true_tvt = ev.TVT.values.astype(np.float64)
    pts = _pf_ancc_v3(md_v, z_v, gr_v, gg, gmin, gst, gs, ls, ir, N,
                       ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP,
                       lk_kind, cauchy_scale, gr_gain, gr_offset)
    return pts, true_tvt

def pooled_rmse(wid_list, seed, lk_kind=0, cauchy_scale=1.0, recalibrate=False, N=300):
    sq, n = 0.0, 0
    for wid in wid_list:
        r = run_one(wid, seed, lk_kind, cauchy_scale, recalibrate, N)
        if r is None: continue
        pts, true_tvt = r
        sq += np.sum((pts - true_tvt) ** 2); n += len(true_tvt)
    return float(np.sqrt(sq / n)) if n > 0 else np.nan

if __name__ == '__main__':
    wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    rng = np.random.default_rng(7)
    sample = list(rng.choice(wids, size=120, replace=False))
    SEEDS = [1, 2, 3, 4]

    print('=== baseline (Gaussian, no recalibration) ===')
    base = [pooled_rmse(sample, sd, lk_kind=0) for sd in SEEDS]
    print(f'  per-seed={[round(v,3) for v in base]}  mean={np.mean(base):.4f}  std={np.std(base):.4f}')

    print('\n=== (A) Cauchy likelihood, sweep scale ===')
    for cs in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        vals = [pooled_rmse(sample, sd, lk_kind=1, cauchy_scale=cs) for sd in SEEDS]
        print(f'  cauchy_scale={cs:.2f}  per-seed={[round(v,3) for v in vals]}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}')

    print('\n=== (B) GR/typewell affine recalibration (Gaussian likelihood, gs kept) ===')
    vals = [pooled_rmse(sample, sd, lk_kind=0, recalibrate=True) for sd in SEEDS]
    print(f'  recalibrated  per-seed={[round(v,3) for v in vals]}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}')

    print('\n=== cross-fit: best cauchy_scale fit on H1 (seeds 1,2), tested on H2 (seeds 3,4), and vice versa ===')
    folds = np.array_split(np.array(sample), 2)
    H1, H2 = list(folds[0]), list(folds[1])
    FIT_SEEDS = [1, 2]; TEST_SEEDS = [3, 4]
    SCALES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    def best_scale_on(wl):
        best = (1e9, None)
        for cs in SCALES:
            v = np.mean([pooled_rmse(wl, sd, lk_kind=1, cauchy_scale=cs) for sd in FIT_SEEDS])
            if v < best[0]: best = (v, cs)
        return best

    b1 = best_scale_on(H1)
    v_h2 = np.mean([pooled_rmse(H2, sd, lk_kind=1, cauchy_scale=b1[1]) for sd in TEST_SEEDS])
    base_h2 = np.mean([pooled_rmse(H2, sd, lk_kind=0) for sd in TEST_SEEDS])
    print(f'fit on H1: best scale={b1[1]} (train={b1[0]:.4f}) -> H2 test={v_h2:.4f} vs H2 Gaussian baseline={base_h2:.4f}')

    b2 = best_scale_on(H2)
    v_h1 = np.mean([pooled_rmse(H1, sd, lk_kind=1, cauchy_scale=b2[1]) for sd in TEST_SEEDS])
    base_h1 = np.mean([pooled_rmse(H1, sd, lk_kind=0) for sd in TEST_SEEDS])
    print(f'fit on H2: best scale={b2[1]} (train={b2[0]:.4f}) -> H1 test={v_h1:.4f} vs H1 Gaussian baseline={base_h1:.4f}')
