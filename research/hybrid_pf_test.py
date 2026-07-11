"""New idea (inspired by a competitor's biggest lever: fixing the motion assumption INSIDE the tracker,
not blending a second tracker's output): pf_ancc's `rate` (d(surface)/dMD) currently evolves by pure
momentum (ALPHA=0.998), with no coupling to the trajectory's own Z-derivative. pf_z separately shows that
dTVT/dMD correlates with dZ/dMD (fit via known-zone regression, beta/icpt). Since surface = TVT + Z,
d(surface)/dMD = dTVT/dMD + dZ/dMD =~ (beta+1)*dZ/dMD + icpt.

We build `_pf_ancc_hybrid`: identical to the production _pf_ancc, but ALSO reweights particles each step
by a soft Gaussian likelihood pulling `rate` toward this Z-derived expectation, with a tunable strength
(0 = pure baseline pf_ancc, 1 = fully trust the Z-coupling like pf_z does). This directly fixes the motion
model INSIDE the main tracker, rather than running a second tracker and blending outputs afterward (the
approach that broke when stacked as a third correction, §7.2).

Cross-fit validated (fit strength on one half, test on the other, both directions) on real single-seed
pf_ancc runs (not the proxy/sp45 cache, which already includes GBM+projection -- this test needs to isolate
the tracker's OWN motion model, so it runs the actual PF kernel from scratch).
"""
import numpy as np, pandas as pd, glob, os
from numba import njit

TRAIN_DIR = 'd:/ROGII/data/train'

PF_N = 300  # reduced from production 600 for speed; this is a relative-comparison test, not a submission
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
def _pf_ancc_base(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
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

@njit(cache=False)
def _pf_ancc_hybrid(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP,
                     beta_r, icpt_r, zsig_r, strength):
    """Same as _pf_ancc_base, but each step ALSO reweights by a soft Gaussian pulling `rate` toward the
    Z-derived expectation re=(beta_r+1)*dZ/dMD+icpt_r, at tunable `strength` (0=off, 1=full pf_z-style trust).
    """
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls+IS*np.random.randn(); rate[j] = ir+0.01*np.random.randn()
    pts = np.empty(len(md_v)); pm = md_v[0]-1.; pz = z_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.); dzd = (z_v[i]-pz)/dm; re = (beta_r+1.)*dzd+icpt_r
        for j in range(N):
            rate[j] = ALPHA*rate[j]+RN*np.random.randn(); pos[j] += rate[j]*dm+PN*np.random.randn()
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
        if strength > 0.:
            ws2 = 0.
            for j in range(N):
                dv = (rate[j]-re)/max(zsig_r*2., 0.005)
                lz = max(np.exp(-0.5*dv*dv) if dv*dv < 600. else 0., 1e-300)
                lz = lz**strength  # temper the pull: strength=0 -> lz=1 (no effect), strength=1 -> full pf_z-style
                w[j] *= lz; ws2 += w[j]
            if ws2 > 0.:
                for j in range(N): w[j] /= ws2
            else:
                for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
        tv = 0.
        for j in range(N): tv += w[j]*(pos[j]-z_v[i])
        pts[i] = tv; pm = md_v[i]; pz = z_v[i]
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

def fit_beta_icpt(kn):
    dz_k = np.diff(kn.Z.values); dvt = np.diff(kn.TVT_input.values); dmd_k = np.diff(kn.MD.values)
    m2 = dmd_k > 0
    if m2.sum() < 10: return -1., 0., 0.1
    vz = dz_k[m2]/dmd_k[m2]; vt = dvt[m2]/dmd_k[m2]; A = np.column_stack([vz, np.ones_like(vz)])
    c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
    beta, icpt = float(c[0]), float(c[1])
    zsig = max(float(np.std(vt-(c[0]*vz+c[1]))), 0.001)
    return beta, icpt, zsig

def run_one(wid, strength, seed):
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
    beta_r, icpt_r, zsig_r = fit_beta_icpt(kn)
    md_v = ev.MD.values.astype(np.float64); z_v = ev.Z.values.astype(np.float64); gr_v = ev.GR.values.astype(np.float64)
    true_tvt = ev.TVT.values.astype(np.float64)
    if strength == 0.0:
        pts = _pf_ancc_base(md_v, z_v, gr_v, gg, gmin, gst, gs, ls, ir, PF_N,
                             ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
    else:
        pts = _pf_ancc_hybrid(md_v, z_v, gr_v, gg, gmin, gst, gs, ls, ir, PF_N,
                               ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP,
                               beta_r, icpt_r, zsig_r, strength)
    return pts, true_tvt

def pooled_rmse(wid_list, strength, seed=0):
    sq, n = 0.0, 0
    for wid in wid_list:
        r = run_one(wid, strength, seed)
        if r is None: continue
        pts, true_tvt = r
        sq += np.sum((pts - true_tvt) ** 2); n += len(true_tvt)
    return float(np.sqrt(sq / n)) if n > 0 else np.nan

if __name__ == '__main__':
    wids = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(f'{TRAIN_DIR}/*__horizontal_well.csv')})
    rng = np.random.default_rng(7)
    sample = list(rng.choice(wids, size=200, replace=False))
    folds = np.array_split(np.array(sample), 2)
    H1, H2 = list(folds[0]), list(folds[1])

    STRENGTHS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print('=== full-sweep on all 200 wells (1 seed) ===')
    for s in STRENGTHS:
        v = pooled_rmse(sample, s)
        print(f'  strength={s:.1f}  pooled={v:.4f}')

    print('\n=== cross-fit: fit best strength on H1, test on H2, and vice versa ===')
    def best_on(wl):
        best = (1e9, None)
        for s in STRENGTHS:
            v = pooled_rmse(wl, s)
            if v < best[0]: best = (v, s)
        return best
    b1 = best_on(H1)
    v_h2 = pooled_rmse(H2, b1[1])
    base_h2 = pooled_rmse(H2, 0.0)
    print(f'fit on H1: best strength={b1[1]} (train={b1[0]:.4f}) -> H2 test={v_h2:.4f} vs H2 baseline(strength=0)={base_h2:.4f}')

    b2 = best_on(H2)
    v_h1 = pooled_rmse(H1, b2[1])
    base_h1 = pooled_rmse(H1, 0.0)
    print(f'fit on H2: best strength={b2[1]} (train={b2[0]:.4f}) -> H1 test={v_h1:.4f} vs H1 baseline(strength=0)={base_h1:.4f}')
